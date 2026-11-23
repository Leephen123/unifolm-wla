#!/usr/bin/env python3
"""Prepare MPDS (.mpds) datasets for UnifoLM-WLA full-pipeline training.

Discovers .mpds episodes under --mpds-root, computes exact WLA normalization
statistics (dataset_statistics.json), validates images/states/actions, and writes
manifest.json + dataset_statistics.json + data-acceptance.json under dataset/prepared/.
Optionally attaches SAM 3.1 background masks via --mask-index.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
from tempfile import NamedTemporaryFile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ACTION_STEPS = 50
PREPARED_FILES = ("manifest.json", "dataset_statistics.json", "data-acceptance.json")


def write_json(path: Path, value: object) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            json.dump(value, tmp, ensure_ascii=False)
            tmp.write("\n")
            tmp.close()
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


def parse_episode_index(path: Path, fallback: int) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else fallback


def discover_mpds_files(root: Path) -> dict[str, list[Path]]:
    """Group .mpds files by immediate parent directory (treated as logical dataset)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.rglob("*.mpds")):
        if any(part.startswith(".") or part in ("prepared", "eval") for part in p.relative_to(root).parts):
            continue
        groups[str(p.parent.resolve())].append(p.resolve())
    if not groups:
        raise ValueError(f"No .mpds files found under {root}")
    return dict(sorted(groups.items()))


def _inspect_worker(job: tuple[str, str, dict | None]) -> dict:
    from unifolm_wla.dataloader.mpds_backend import inspect_mpds_episode
    path, head_camera, feature_map = job
    return inspect_mpds_episode(path, head_camera=head_camera, feature_map=feature_map)


def prepare_mpds(
    mpds_root: Path,
    prompt: str | None,
    head_camera: str = "front_head",
    feature_map: dict[str, str] | None = None,
    workers: int = 4,
    mask_index_path: Path | None = None,
) -> tuple[dict, dict, dict, dict | None]:
    from unifolm_wla.dataloader.full_pipeline import pack_state, pad_resize
    from unifolm_wla.dataloader.mpds_backend import clear_mpds_caches, read_mpds_sample

    ds_groups = discover_mpds_files(mpds_root)
    mask_lookup: dict[tuple[str, int], str] = {}
    mask_camera = head_camera
    if mask_index_path is not None:
        mi = json.loads(mask_index_path.read_text(encoding="utf-8"))
        mask_camera = mi.get("camera") or mi.get("background_mask_camera") or head_camera
        for item in mi.get("episodes", []):
            mpath = item.get("masks_parquet") or item.get("background_masks_parquet")
            if not mpath:
                continue
            ds_key = Path(item["dataset"]).name
            mask_lookup[(ds_key, int(item["episode_index"]))] = str(Path(mpath).resolve())

    entries: list[dict] = []
    groups_stats: list[dict] = []
    all_rows = all_valid = 0
    inferred_prompt = prompt

    for ds_dir, files in ds_groups.items():
        jobs = [(str(f), head_camera, feature_map) for f in files]
        inspected = []
        if workers <= 1 or len(jobs) == 1:
            inspected = [_inspect_worker(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                futs = {pool.submit(_inspect_worker, j): idx for idx, j in enumerate(jobs)}
                results = [None] * len(jobs)
                for fut in as_completed(futs):
                    results[futs[fut]] = fut.result()
                inspected = results

        states_list = []
        hands_list = []
        samples = 0
        total = np.zeros(102, dtype=np.float64)
        total2 = np.zeros(102, dtype=np.float64)

        for idx_in_ds, (fpath, info) in enumerate(zip(files, inspected, strict=True)):
            eid = parse_episode_index(fpath, idx_in_ds)
            if inferred_prompt is None and info["prompt"]:
                inferred_prompt = info["prompt"]

            cur_s = info["current_states"]
            cur_sv = info["current_state_valid"]
            if cur_sv.any():
                states_list.append(cur_s[cur_sv].astype(np.float32))

            v_acts = info["valid_actions"]
            if len(v_acts):
                arm_a = v_acts[:, :, :102].reshape(-1, 102)
                hand_a = v_acts[:, :, 102:].reshape(-1, 12)
                samples += len(arm_a)
                total += arm_a.sum(axis=0)
                total2 += (arm_a ** 2).sum(axis=0)
                hands_list.append(hand_a.astype(np.float32))

            entry = {
                "mpds_path": info["path"],
                "mpds_layout": info["layout"],
                "dataset": ds_dir,
                "episode_index": eid,
                "valid_rows": info["valid_rows"],
                "row_timestamps": info["row_timestamps"],
            }
            if mask_lookup:
                mp = mask_lookup.get((Path(ds_dir).name, eid))
                if mp is not None:
                    entry["background_masks_parquet"] = mp
            entries.append(entry)
            all_rows += info["num_frames"]
            all_valid += len(info["valid_rows"])

        if not states_list:
            raise ValueError(f"No valid states in MPDS dataset group: {ds_dir}")
        if samples == 0:
            raise ValueError(f"No valid {ACTION_STEPS}-step action windows in MPDS dataset group: {ds_dir}")

        mean = total / samples
        groups_stats.append({
            "state_q": np.quantile(np.concatenate(states_list, axis=0), [0.01, 0.99], axis=0),
            "hand_q": np.quantile(np.concatenate(hands_list, axis=0), [0.01, 0.99], axis=0),
            "mean": mean,
            "variance": np.maximum(total2 / samples - mean ** 2, 0.0),
        })
        print(f"Prepared MPDS {Path(ds_dir).name}: {len(files)} episodes, {all_valid} valid anchors", flush=True)

    if not all_valid:
        raise ValueError("No valid training anchors found across MPDS episodes.")
    if not inferred_prompt:
        raise ValueError("No prompt provided via --prompt and no prompt/texts stream found in MPDS files.")

    # Exact same WLA normalization math as scripts/prepare_full_pipeline.py
    state_lo = np.min([g["state_q"][0] for g in groups_stats], axis=0)
    state_hi = np.max([g["state_q"][1] for g in groups_stats], axis=0)
    state_offset, state_scale = (state_lo + state_hi) / 2, (state_hi - state_lo) / 2
    state_offset[29:35], state_scale[29:35] = 0.0, 1.0  # Preserve IMU rotation representation.
    means = np.stack([g["mean"] for g in groups_stats])
    body_offset = means.mean(0)
    body_variance = np.mean([g["variance"] for g in groups_stats], 0) + (means ** 2).mean(0) - body_offset ** 2
    hand_lo = np.min([g["hand_q"][0] for g in groups_stats], axis=0)
    hand_hi = np.max([g["hand_q"][1] for g in groups_stats], axis=0)
    action_offset = np.concatenate((body_offset, (hand_lo + hand_hi) / 2))
    action_scale = np.concatenate((np.sqrt(np.maximum(body_variance, 0.0)), (hand_hi - hand_lo) / 2))
    state_scale[state_scale < 1e-6] = 1.0
    action_scale[action_scale < 1e-6] = 1.0

    # Decode-check one sample per dataset group
    checked = set()
    for entry in entries:
        if entry["dataset"] in checked or not entry["valid_rows"]:
            continue
        imgs, raw_s, s_ok, act, _, _, _ = read_mpds_sample(
            entry["mpds_path"],
            entry["valid_rows"][0],
            head_camera=head_camera,
            history_frames=0,
            feature_map=feature_map,
            row_timestamps=entry["row_timestamps"],
            layout=entry["mpds_layout"],
        )
        if s_ok:
            packed, mask = pack_state((raw_s - state_offset) / state_scale)
            if mask.sum() != 47 or not np.isfinite(packed).all():
                raise ValueError(f"Invalid packed state in {entry['mpds_path']}")
        if act.shape != (ACTION_STEPS, 114) or not np.isfinite(act).all():
            raise ValueError(f"Invalid action shape {act.shape} in {entry['mpds_path']}")
        for im in imgs:
            if pad_resize(im).size != (448, 320):
                raise ValueError(f"Unexpected resized image in {entry['mpds_path']}")
        checked.add(entry["dataset"])
    clear_mpds_caches()

    manifest = {
        "source_format": "mpds",
        "mpds_root": str(mpds_root.resolve()),
        "prompt": inferred_prompt,
        "head_camera": head_camera,
        "episodes": entries,
        "total_frames": all_rows,
        "valid_anchors": all_valid,
    }
    statistics = {
        "full_pipeline": {
            "state": {"offset": state_offset.tolist(), "scale": state_scale.tolist()},
            "action": {"offset": action_offset.tolist(), "scale": action_scale.tolist()},
        }
    }
    acceptance = {
        "status": "PASS",
        "source_format": "mpds",
        "episodes": len(entries),
        "valid_anchors": all_valid,
        "decoded_datasets": len(checked),
        "state_dim": 47,
        "packed_state_dim": 60,
        "action_shape": [ACTION_STEPS, 114],
    }

    masked_manifest = None
    if mask_lookup and all("background_masks_parquet" in e for e in entries):
        masked_manifest = dict(manifest, background_mask_camera=mask_camera)

    return manifest, statistics, acceptance, masked_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpds-root", required=True, type=Path, help="Directory containing .mpds episode files.")
    parser.add_argument("--dataset-name", required=True, help="Output folder name under dataset/prepared/ (or --output-dir).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional explicit output directory instead of dataset/prepared/<name>.")
    parser.add_argument("--prompt", default=None, help="Fallback task prompt (optional if prompt/texts exists in MPDS).")
    parser.add_argument("--head-camera", choices=("front_head", "fov_head"), default="front_head")
    parser.add_argument("--feature-map", default=None, help="Optional JSON dict mapping WLA roles to custom MPDS feature keys.")
    parser.add_argument("--mask-index", type=Path, default=None, help="Optional SAM 3.1 mask_index.json to attach masks.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    feature_map = json.loads(args.feature_map) if args.feature_map else None
    output = args.output_dir.resolve() if args.output_dir else (REPO / "dataset/prepared" / args.dataset_name)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Already exists: {output}")

    manifest, statistics, acceptance, masked_manifest = prepare_mpds(
        args.mpds_root.expanduser().resolve(strict=True),
        prompt=args.prompt,
        head_camera=args.head_camera,
        feature_map=feature_map,
        workers=args.workers,
        mask_index_path=args.mask_index.expanduser().resolve(strict=True) if args.mask_index else None,
    )
    output.mkdir(parents=True, exist_ok=False)
    for fname, value in zip(PREPARED_FILES, (manifest, statistics, acceptance), strict=True):
        write_json(output / fname, value)
    if masked_manifest is not None:
        write_json(output / "manifest_background_masks_tracked.json", masked_manifest)
    print(f"Prepared MPDS training data: {output}", flush=True)


if __name__ == "__main__":
    main()
