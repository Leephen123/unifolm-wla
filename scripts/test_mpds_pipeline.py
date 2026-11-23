"""Unit & integration tests for the MPDS data pipeline (packed & single_frame async)."""

from __future__ import annotations

import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from unifolm_wla.dataloader import build_dataloader
from unifolm_wla.dataloader.full_pipeline import FullPipelineDataset, pack_state
from unifolm_wla.dataloader.mpds_backend import clear_mpds_caches, resolve_mpds_schema


def _create_synthetic_mpds(path: Path, *, layout: str, num_frames: int = 80, async_state_fps: int | None = None) -> dict:
    from mpds.dataset import MPDataset
    from mpds.device import mp_use_device

    rng = np.random.default_rng(123 if layout == "packed" else 456)
    arm_states_20hz = rng.standard_normal((num_frames, 35)).astype(np.float32) * 0.2
    hand_states_20hz = rng.standard_normal((num_frames, 12)).astype(np.float32) * 0.1
    arm_actions_20hz = rng.standard_normal((num_frames + 55, 102)).astype(np.float32) * 0.1
    hand_actions_20hz = rng.standard_normal((num_frames + 55, 12)).astype(np.float32) * 0.1

    video_tb = Fraction(1, 20)
    state_fps = async_state_fps if (layout == "single_frame" and async_state_fps) else 20
    state_tb = Fraction(1, state_fps)

    # Use WLA-style keys for packed and openpi-style keys for single_frame to verify both!
    cam_keys = ("camera/front/rgb", "camera/left/rgb", "camera/right/rgb")
    if layout == "packed":
        st_arm_k, st_hand_k = "robot/g1/arm_state", "robot/g1/hand_state"
        act_arm_k, act_hand_k = "robot/g1/arm_actions", "robot/g1/hand_actions"
    else:
        st_arm_k, st_hand_k = "robot/g1/state", "robot/g1/hand_state"
        act_arm_k, act_hand_k = "robot/g1/action", "robot/g1/hand_action"

    features = {
        cam_keys[0]: {"type": "video/h264", "time_base": str(video_tb)},
        cam_keys[1]: {"type": "video/h264", "time_base": str(video_tb)},
        cam_keys[2]: {"type": "video/h264", "time_base": str(video_tb)},
        st_arm_k: {"type": "application/vnd.mpds.arraycodec", "time_base": str(state_tb)},
        st_hand_k: {"type": "application/vnd.mpds.arraycodec", "time_base": str(state_tb)},
        act_arm_k: {"type": "application/vnd.mpds.arraycodec", "time_base": str(video_tb)},
        act_hand_k: {"type": "application/vnd.mpds.arraycodec", "time_base": str(video_tb)},
        "prompt/texts": {"type": "application/json", "time_base": str(video_tb)},
    }
    if layout == "packed":
        features["robot/g1/state_valid"] = {"type": "application/vnd.mpds.arraycodec", "time_base": str(video_tb)}
        features["robot/g1/action_valid"] = {"type": "application/vnd.mpds.arraycodec", "time_base": str(video_tb)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with mp_use_device(num_threads=1):
        with MPDataset.create(str(path), metadata={"features": features}) as ds:
            for r in range(num_frames):
                t = video_tb * r
                frame = np.full((64, 96, 3), fill_value=(r * 3) % 250, dtype=np.uint8)
                data = {
                    cam_keys[0]: frame,
                    cam_keys[1]: frame,
                    cam_keys[2]: frame,
                    "prompt/texts": ["Fold the short jeans neatly."],
                }
                if layout == "packed":
                    a100 = np.zeros((100, 35), dtype=np.float32)
                    h100 = np.zeros((100, 12), dtype=np.float32)
                    sv = np.zeros((100,), dtype=bool)
                    for s in range(100):
                        back = (99 - s) * 2
                        if r - back >= 0:
                            a100[s] = arm_states_20hz[r - back]
                            h100[s] = hand_states_20hz[r - back]
                            sv[s] = True
                    data[st_arm_k] = a100
                    data[st_hand_k] = h100
                    data["robot/g1/state_valid"] = sv
                    data[act_arm_k] = arm_actions_20hz[r + 1 : r + 51]
                    data[act_hand_k] = hand_actions_20hz[r + 1 : r + 51]
                    data["robot/g1/action_valid"] = np.array([True], dtype=bool)
                else:
                    data[act_arm_k] = arm_actions_20hz[r]
                    data[act_hand_k] = hand_actions_20hz[r]
                    step_ratio = 20 // state_fps
                    if r % step_ratio == 0:
                        data[st_arm_k] = arm_states_20hz[r]
                        data[st_hand_k] = hand_states_20hz[r]
                ds.append(time=t, data=data)

            if layout == "single_frame":
                # Extra future actions so all num_frames anchors have full 50 future actions
                for r in range(num_frames, num_frames + 55):
                    t = video_tb * r
                    ds.append(
                        time=t,
                        data={
                            act_arm_k: arm_actions_20hz[r],
                            act_hand_k: hand_actions_20hz[r],
                        },
                    )

    return {
        "arm_states_20hz": arm_states_20hz,
        "hand_states_20hz": hand_states_20hz,
        "arm_actions_20hz": arm_actions_20hz,
        "hand_actions_20hz": hand_actions_20hz,
    }


def _make_cfg(manifest_path: Path, stats_path: Path, num_workers: int = 2) -> OmegaConf:
    cfg = OmegaConf.load(REPO / "unifolm_wla/config/training/fold_shorts_base_mpds.yaml")
    cfg.datasets.vla_data.manifest_path = str(manifest_path)
    cfg.datasets.vla_data.statistics_path = str(stats_path)
    cfg.datasets.vla_data.per_device_batch_size = 4
    cfg.datasets.vla_data.num_workers = num_workers
    cfg.datasets.vla_data.background_recolor = False
    return cfg


def test_packed_and_single_frame_end_to_end(tmp_path: Path):
    from scripts.prepare_mpds import prepare_mpds, write_json

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)

        # 1. Packed layout
        packed_root = tmp_path / "packed_raw" / "synth_ds"
        ref_packed = _create_synthetic_mpds(packed_root / "episode_000000.mpds", layout="packed", num_frames=80)
        manifest_p, stats_p, acc_p, _ = prepare_mpds(tmp_path / "packed_raw", prompt=None, workers=1)
        assert acc_p["status"] == "PASS" and acc_p["valid_anchors"] == 80
        prep_p = tmp_path / "packed_prep"
        prep_p.mkdir()
        write_json(prep_p / "manifest.json", manifest_p)
        write_json(prep_p / "dataset_statistics.json", stats_p)

        ds_p = FullPipelineDataset(prep_p / "manifest.json", prep_p / "dataset_statistics.json", history_frames=50, prompt_mode="full_only")
        assert len(ds_p) == 80
        ex = ds_p[60]
        assert ex["state"].shape == (60,)
        assert ex["state_history"].shape == (50, 60)
        assert ex["action"].shape == (50, 114)
        assert ex["lang"] == "Fold the short jeans neatly."
        assert len(ex["image"]) == 3 and all(im.size == (448, 320) for im in ex["image"])
        expected_act = np.concatenate((ref_packed["arm_actions_20hz"][61:111], ref_packed["hand_actions_20hz"][61:111]), axis=-1)
        got_act = ex["action"] * ds_p.asc + ds_p.ao
        assert np.max(np.abs(got_act - expected_act)) < 1e-5

        loader_p = build_dataloader(_make_cfg(prep_p / "manifest.json", prep_p / "dataset_statistics.json", num_workers=2), dataset_py="mpds")
        batch_p = next(iter(loader_p))
        assert len(batch_p) == 4

        # 2. Single-frame async 10Hz state + 20Hz video/action layout (openpi G1 schema)
        sf_root = tmp_path / "sf_raw" / "synth_ds"
        ref_sf = _create_synthetic_mpds(sf_root / "episode_000000.mpds", layout="single_frame", num_frames=80, async_state_fps=10)
        manifest_s, stats_s, acc_s, _ = prepare_mpds(tmp_path / "sf_raw", prompt=None, workers=1)
        assert acc_s["status"] == "PASS" and acc_s["valid_anchors"] == 80
        prep_s = tmp_path / "sf_prep"
        prep_s.mkdir()
        write_json(prep_s / "manifest.json", manifest_s)
        write_json(prep_s / "dataset_statistics.json", stats_s)

        ds_s = FullPipelineDataset(prep_s / "manifest.json", prep_s / "dataset_statistics.json", history_frames=50, prompt_mode="full_only")
        ex_s = ds_s[60]  # t = 3.0s => 30 valid 10Hz history slots (-3.0s .. -0.1s)
        assert ex_s["state_history_mask"].any(axis=1).sum() == 30
        # Slot 49 is t - 0.1s = 2.9s => 20Hz frame index 58
        raw_58 = np.concatenate((ref_sf["arm_states_20hz"][58], ref_sf["hand_states_20hz"][58]))
        packed_58, _ = pack_state((raw_58 - ds_s.so) / ds_s.ss)
        assert np.max(np.abs(ex_s["state_history"][49] - packed_58)) < 1e-5

        loader_s = build_dataloader(_make_cfg(prep_s / "manifest.json", prep_s / "dataset_statistics.json", num_workers=2), dataset_py="mpds")
        batch_s = next(iter(loader_s))
        assert len(batch_s) == 4

        clear_mpds_caches()
        mpds_warnings = [w for w in caught if "mpds" in str(w.message).lower() or "MPDataset" in str(w.message)]
        assert not mpds_warnings, [str(w.message) for w in mpds_warnings]


def test_custom_feature_map_resolution():
    meta = {
        "features": {
            "cam/main": {},
            "cam/l": {},
            "cam/r": {},
            "obs/state47": {},
            "act/chunk114": {},
        }
    }
    schema = resolve_mpds_schema(
        meta,
        head_camera="front_head",
        feature_map={
            "front_head": "cam/main",
            "left_hand": "cam/l",
            "right_hand": "cam/r",
            "state": "obs/state47",
            "actions": "act/chunk114",
        },
    )
    assert schema["front_head"] == "cam/main"
    assert schema["state"] == "obs/state47"
    assert schema["actions"] == "act/chunk114"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_custom_feature_map_resolution()
        test_packed_and_single_frame_end_to_end(Path(td))
    print("SYNTHETIC_UNIT_TESTS_PASSED")
