#!/usr/bin/env python3
"""Prepare G1 fold-shorts training data and optionally submit SAM mask jobs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

REPO = Path(__file__).resolve().parents[1]
ACTION_STEPS = 50
PREPARED_FILES = ("manifest.json", "dataset_statistics.json", "data-acceptance.json")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            json.dump(value, temporary, ensure_ascii=False)
            temporary.write("\n")
            temporary.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def episode_metadata(root):
    info = read_json(root / "meta/info.json")
    if info["codebase_version"] != "v2.1" or info["fps"] != 20:
        raise ValueError(f"Expected LeRobot v2.1 at 20 FPS: {root}")
    with (root / "meta/episodes.jsonl").open(encoding="utf-8") as records:
        episodes = [json.loads(line) for line in records if line.strip()]
    if not episodes or len({item["episode_index"] for item in episodes}) != len(episodes):
        raise ValueError(f"Expected nonempty, unique episode metadata: {root}")
    return info, episodes


def discover_selection(train, prompt):
    datasets = []
    for parent, directories, _ in os.walk(train):
        directories.sort()
        root = Path(parent)
        if not (root / "meta/info.json").is_file():
            directories[:] = [name for name in directories if name != "prepared" and not name.startswith(".")]
            continue
        directories.clear()
        _, episodes = episode_metadata(root)
        datasets.append({
            "repo_id": str(root.resolve()), "episodes": len(episodes),
            "frames": sum(episode["length"] for episode in episodes),
        })
    if not datasets:
        raise ValueError(f"No LeRobot datasets found under {train}")
    return {
        "datasets": datasets,
        "episodes": sum(dataset["episodes"] for dataset in datasets),
        "frames": sum(dataset["frames"] for dataset in datasets),
        "prompt": prompt,
    }


def prepare(selection, selection_path):
    import av
    import numpy as np
    import pyarrow.parquet as pq

    # Direct execution from scripts/ uses the checkout's training data contract.
    sys.path.insert(0, str(REPO))
    from unifolm_wla.dataloader.full_pipeline import CAMERAS, load_state_valid_episode, pack_state, pad_resize

    entries, groups = [], []
    all_rows = all_valid = 0
    roots = set()
    for dataset in selection["datasets"]:
        root = Path(dataset["repo_id"]).resolve(strict=True)
        if root in roots:
            raise ValueError(f"Dataset appears twice in the selection: {root}")
        roots.add(root)
        info, episodes = episode_metadata(root)
        if len(episodes) != dataset["episodes"] or sum(e["length"] for e in episodes) != dataset["frames"]:
            raise ValueError(f"Selection counts differ from LeRobot metadata: {root}")

        states, hands = [], []
        samples = 0
        total, total2 = np.zeros(102), np.zeros(102)
        for episode in episodes:
            eid = episode["episode_index"]
            body = root / "body" / f"episode_{eid:06d}.npz"
            file_args = dict(episode_chunk=eid // info["chunks_size"], episode_index=eid)
            parquet = root / info["data_path"].format(**file_args)
            if "state_valid" in info["features"]:
                state, actions, state_valid, valid = load_state_valid_episode(str(parquet))
                load_state_valid_episode.cache_clear()
                if len(valid) != episode["length"]:
                    raise ValueError(f"Metadata and Parquet row counts differ: {parquet}")
                states.append(state[valid & state_valid[:, -1], -1])
                windows = actions[valid].reshape(-1, 114)
                samples += len(windows)
                body_actions = windows[:, :102]
                total += body_actions.sum(0)
                total2 += (body_actions ** 2).sum(0)
                hands.append(windows[:, 102:].astype(np.float32))
                source = {"state_valid_parquet": str(parquet)}
                frame_count = len(valid)
            else:
                with np.load(body, allow_pickle=False) as z:
                    arm, hand = z["arm_state"], z["hand_state"]
                    action, hand_action, centers = z["arm_actions"], z["hand_actions"], z["centers"]
                    if (action.shape != (len(action), 102) or arm.shape != (len(action), 35)
                            or hand.shape != (len(action), 12) or hand_action.shape != hand.shape
                            or centers.shape != (episode["length"],) or not len(centers)
                            or not np.issubdtype(centers.dtype, np.integer)
                            or np.any(centers < 0) or np.any(centers >= len(action))):
                        raise ValueError(f"Unexpected state, action or center indices: {body}")
                    if not all(np.isfinite(array).all() for array in (arm, hand, action, hand_action)):
                        raise ValueError(f"Nonfinite state or action: {body}")

                    valid = centers + ACTION_STEPS < len(action)
                    table = pq.read_table(
                        parquet, columns=["action_valid", "arm_state", "hand_state", "arm_actions", "hand_actions"],
                        use_threads=False,
                    )
                    actual_valid = np.asarray(table["action_valid"].to_pylist()).reshape(-1).astype(bool)
                    np.testing.assert_array_equal(valid, actual_valid, err_msg=str(parquet))
                    for row in sorted({0, len(centers) // 2, len(centers) - 1}):
                        center = int(centers[row])
                        for field, width, expected in (("arm_state", 35, arm[center]), ("hand_state", 12, hand[center])):
                            np.testing.assert_array_equal(
                                np.asarray(table[field][row].as_py()).reshape(200, width)[-1],
                                expected, err_msg=f"{parquet} {field} row {row}",
                            )
                        if valid[row]:
                            steps = slice(center + 1, center + ACTION_STEPS + 1)
                            for field, width, expected in (
                                ("arm_actions", 102, action[steps]), ("hand_actions", 12, hand_action[steps]),
                            ):
                                np.testing.assert_array_equal(
                                    np.asarray(table[field][row].as_py()).reshape(ACTION_STEPS, width),
                                    expected, err_msg=f"{parquet} {field} row {row}",
                                )
                    del table

                    current = centers[valid]
                    states.append(np.concatenate((arm[current], hand[current]), axis=1))
                    # Count how many training windows contain each action.
                    diff = np.zeros(len(action) + 1, dtype=np.int64)
                    np.add.at(diff, current + 1, 1)
                    np.add.at(diff, current + ACTION_STEPS + 1, -1)
                    weights = np.cumsum(diff)[:-1]
                    samples += int(weights.sum())
                    total += (action * weights[:, None]).sum(0)
                    total2 += (action ** 2 * weights[:, None]).sum(0)
                    hands.append(np.repeat(hand_action, weights, axis=0).astype(np.float32))

                source = {"body": str(body)}
                frame_count = len(centers)

            videos = {
                key: str(root / info["video_path"].format(**file_args, video_key=key))
                for key, _ in CAMERAS
            }
            for path in videos.values():
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            entries.append({
                **source, "dataset": str(root), "episode_index": eid,
                "videos": videos, "valid_rows": np.flatnonzero(valid).tolist(),
            })
            all_rows += frame_count
            all_valid += int(valid.sum())

        if not any(len(array) for array in states):
            raise ValueError(f"No valid current states for normalization: {root}")
        if samples == 0:
            raise ValueError(f"No complete {ACTION_STEPS}-step action windows: {root}")
        mean = total / samples
        groups.append({
            "state_q": np.quantile(np.concatenate(states), [.01, .99], axis=0),
            "hand_q": np.quantile(np.concatenate(hands), [.01, .99], axis=0),
            "mean": mean, "variance": np.maximum(total2 / samples - mean ** 2, 0),
        })
        print(f"Prepared {root.name}: {len(entries)} episodes, {all_valid} valid frames", flush=True)

    if len(entries) != selection["episodes"] or all_rows != selection["frames"] or not all_valid:
        raise ValueError("Prepared episode/frame counts differ from the selection or no valid samples exist.")

    state_lo = np.min([group["state_q"][0] for group in groups], axis=0)
    state_hi = np.max([group["state_q"][1] for group in groups], axis=0)
    state_offset, state_scale = (state_lo + state_hi) / 2, (state_hi - state_lo) / 2
    state_offset[29:35], state_scale[29:35] = 0, 1  # Preserve the IMU rotation representation.
    means = np.stack([group["mean"] for group in groups])
    body_offset = means.mean(0)
    body_variance = np.mean([group["variance"] for group in groups], 0) + (means ** 2).mean(0) - body_offset ** 2
    hand_lo = np.min([group["hand_q"][0] for group in groups], axis=0)
    hand_hi = np.max([group["hand_q"][1] for group in groups], axis=0)
    action_offset = np.concatenate((body_offset, (hand_lo + hand_hi) / 2))
    action_scale = np.concatenate((np.sqrt(np.maximum(body_variance, 0)), (hand_hi - hand_lo) / 2))
    state_scale[state_scale < 1e-6] = 1
    action_scale[action_scale < 1e-6] = 1

    checked = set()
    for entry in entries:
        if entry["dataset"] in checked or not entry["valid_rows"]:
            continue
        if "state_valid_parquet" in entry:
            state, _, state_valid, valid = load_state_valid_episode(entry["state_valid_parquet"])
            load_state_valid_episode.cache_clear()
            valid_rows = np.flatnonzero(valid & state_valid[:, -1])
            if not len(valid_rows):
                continue
            state = state[valid_rows[0], -1]
        else:
            with np.load(entry["body"], allow_pickle=False) as z:
                center = z["centers"][entry["valid_rows"][0]]
                state = np.concatenate((z["arm_state"][center], z["hand_state"][center]))
        packed, mask = pack_state((state - state_offset) / state_scale)
        if mask.sum() != 47 or not np.isfinite(packed).all():
            raise ValueError(f"Invalid packed state: {entry['dataset']} episode {entry['episode_index']}")
        for path in entry["videos"].values():
            with av.open(path) as video:
                image = next(video.decode(video=0)).to_image()
            if pad_resize(image).size != (448, 320):
                raise ValueError(f"Unexpected resized image: {path}")
        checked.add(entry["dataset"])

    manifest = {
        "selection": str(selection_path), "prompt": selection["prompt"], "episodes": entries,
        "total_frames": all_rows, "valid_anchors": all_valid,
    }
    statistics = {"full_pipeline": {
        "state": {"offset": state_offset.tolist(), "scale": state_scale.tolist()},
        "action": {"offset": action_offset.tolist(), "scale": action_scale.tolist()},
    }}
    acceptance = {
        "status": "PASS", "episodes": len(entries), "valid_anchors": all_valid,
        "decoded_datasets": len(checked), "state_dim": 47, "packed_state_dim": 60,
        "action_shape": [ACTION_STEPS, 114],
        "parquet_comparison": "all validity rows; first/middle/last state and valid action per episode",
    }
    return manifest, statistics, acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Task directory containing train/ and eval/.")
    parser.add_argument("--selection", type=Path, help="Use an existing selection.json instead of scanning train/.")
    parser.add_argument("--prompt", help="Task prompt when generating a selection.")
    parser.add_argument("--dataset-name", required=True, help="Output folder name under dataset/prepared/.")
    parser.add_argument("--sam-repo", type=Path, help="Submit SAM 3.1 jobs using this repository after preparation.")
    parser.add_argument("--sam-deps", type=Path, help="PyAV/PyArrow directory for the SAM launcher, if needed.")
    parser.add_argument("--sam-shards", type=int, help="Number of SAM Slurm shards.")
    parser.add_argument("--sam-partition", help="SAM Slurm partition.")
    parser.add_argument("--sam-nodelist", help="SAM Slurm node list.")
    args = parser.parse_args()
    name = args.dataset_name
    if not name or name != name.strip() or name in (".", "..") or "/" in name or chr(92) in name:
        parser.error("--dataset-name must be a single folder name.")
    if not args.dataset_root and not args.selection:
        parser.error("Provide --dataset-root or --selection.")
    if not args.selection and (not args.prompt or not args.prompt.strip()):
        parser.error("--prompt is required when scanning train/.")
    if args.selection and args.prompt is not None:
        parser.error("With --selection the prompt comes from that file.")
    if args.sam_repo and not args.dataset_root:
        parser.error("--sam-repo requires --dataset-root.")
    if not args.sam_repo and any(value is not None for value in
                                 (args.sam_deps, args.sam_shards, args.sam_partition, args.sam_nodelist)):
        parser.error("SAM options require --sam-repo.")
    if args.sam_shards is not None and args.sam_shards < 1:
        parser.error("--sam-shards must be positive.")

    root = args.dataset_root.expanduser().resolve(strict=True) if args.dataset_root else None
    output = REPO / "dataset/prepared" / name
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Already exists: {output}. Choose a new --dataset-name.")
    selection_path = args.selection.expanduser().resolve(strict=True) if args.selection else output / "selection.json"
    selection = read_json(selection_path) if args.selection else discover_selection(root / "train", args.prompt)
    if args.sam_repo:
        sam = args.sam_repo.expanduser().resolve(strict=True)
        launcher = sam / "scripts/run_segment_lerobot.py"
        if not launcher.is_file():
            raise FileNotFoundError(launcher)
        if any(not Path(dataset["repo_id"]).resolve().is_relative_to(root) for dataset in selection["datasets"]):
            raise ValueError("SAM requires every selected LeRobot dataset to be under --dataset-root.")
        sam_command = [
            sys.executable, "-B", str(launcher), "--no-media", "--dataset-root", str(root),
            "--manifest", str(output / "manifest.json"), "--output", str(sam / "output" / root.name),
        ]
        for flag, value in (("--deps", args.sam_deps), ("--shards", args.sam_shards),
                            ("--partition", args.sam_partition), ("--nodelist", args.sam_nodelist)):
            if value is not None:
                sam_command.extend((flag, str(value.expanduser().resolve(strict=True) if isinstance(value, Path) else value)))

    manifest, statistics, acceptance = prepare(selection, selection_path)
    output.mkdir(parents=True, exist_ok=False)
    if not args.selection:
        write_json(selection_path, selection)
    for name, value in zip(PREPARED_FILES, (manifest, statistics, acceptance), strict=True):
        write_json(output / name, value)
    print(f"Prepared training data: {output}", flush=True)
    if args.sam_repo:
        subprocess.run(sam_command, cwd=sam, check=True)


if __name__ == "__main__":
    main()
