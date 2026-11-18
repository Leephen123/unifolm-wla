#!/usr/bin/env python3
"""Prepare fixed open-loop references from a fold-shorts LeRobot validation set."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from prepare_full_pipeline import ACTION_STEPS, REPO, episode_metadata, write_json

VIDEO_FPS = 20


def evaluation_channels():
    def channel(index, group, label, unit, prefix="v"):
        return dict(key=f"{prefix}{index}", index=index, group=group, label=label, unit=unit)

    channels = [channel(i, "root", f"Root Δ{axis}", "m") for i, axis in enumerate("XYZ")]
    channels += [
        channel(3 + 9 * joint + i, "xyz", f"身体点 {joint} · {axis}", "m")
        for joint in range(11) for i, axis in enumerate("XYZ")
    ]
    channels += [
        channel(102 + 6 * side + i, "hand", f"{hand} · 通道 {i}", "0–1000")
        for side, hand in enumerate(("左手", "右手")) for i in range(6)
    ]
    channels += [channel(joint, "rotation", f"身体点 {joint} · 旋转误差", "°", "r") for joint in range(11)]
    channels += [
        channel(6 + 9 * joint + i, "rot6d", f"身体点 {joint} · rot6D {i}", "")
        for joint in range(11) for i in range(6)
    ]
    return channels


def prepare_episode(dataset, info, episode, output, scene, canonical, cameras):
    import numpy as np
    import pyarrow.parquet as pq

    eid = episode["episode_index"]
    body = dataset / "body" / f"episode_{eid:06d}.npz"
    file_args = dict(episode_chunk=eid // info["chunks_size"], episode_index=eid)
    parquet = dataset / info["data_path"].format(**file_args)
    # The current evaluator opens these standard three-camera paths directly.
    for camera, _ in cameras:
        video = dataset / info["video_path"].format(**file_args, video_key=camera)
        expected = dataset / f"videos/chunk-000/{camera}/episode_{eid:06d}.mp4"
        if not expected.is_file() or not video.is_file() or not video.samefile(expected):
            raise ValueError(f"Expected evaluator video path {expected}, found {video}")

    with np.load(body, allow_pickle=False) as data:
        centers = data["centers"]
        arm, hand = data["arm_state"], data["hand_state"]
        arm_actions, hand_actions = data["arm_actions"], data["hand_actions"]
        timestamps = data["source_timestamps"]
    body_frames = len(arm_actions)
    if (arm.shape != (body_frames, 35) or hand.shape != (body_frames, 12)
            or arm_actions.shape != (body_frames, 102) or hand_actions.shape != hand.shape
            or timestamps.shape != (body_frames,)
            or centers.shape != (episode["length"],) or not len(centers)
            or not np.issubdtype(centers.dtype, np.integer)
            or np.any(centers < 0) or np.any(centers >= body_frames)
            or np.any(np.diff(centers) < 0)):
        raise ValueError(f"Invalid state, action or frame mapping: {body}")
    if not all(np.isfinite(array).all() for array in (arm, hand, arm_actions, hand_actions, timestamps)):
        raise ValueError(f"Nonfinite state, action or timestamp: {body}")

    states = np.concatenate((arm[centers], hand[centers]), axis=-1)
    valid = centers + ACTION_STEPS < body_frames
    anchor_rows = []
    previous = -ACTION_STEPS
    for row in np.flatnonzero(valid):
        if centers[row] >= previous + ACTION_STEPS:
            anchor_rows.append(int(row))
            previous = centers[row]
    if not anchor_rows:
        raise ValueError(f"No complete {ACTION_STEPS}-step windows: {body}")
    anchors = np.asarray(anchor_rows, dtype=np.int64)
    indices = centers[anchors, None] + np.arange(1, ACTION_STEPS + 1)
    native = np.concatenate((arm_actions[indices], hand_actions[indices]), axis=-1)

    table = pq.read_table(parquet, columns=["action_valid", "arm_state", "hand_state", "arm_actions", "hand_actions"])
    np.testing.assert_array_equal(
        valid, np.asarray(table["action_valid"].to_pylist()).reshape(-1).astype(bool), err_msg=str(parquet),
    )
    expected_actions = np.concatenate((
        np.asarray([table["arm_actions"][row].as_py() for row in anchor_rows]).reshape(-1, ACTION_STEPS, 102),
        np.asarray([table["hand_actions"][row].as_py() for row in anchor_rows]).reshape(-1, ACTION_STEPS, 12),
    ), axis=-1)
    np.testing.assert_array_equal(native, expected_actions, err_msg=str(parquet))
    for row in sorted({0, len(centers) // 2, len(centers) - 1}):
        expected_state = np.concatenate((
            np.asarray(table["arm_state"][row].as_py()).reshape(200, 35)[-1],
            np.asarray(table["hand_state"][row].as_py()).reshape(200, 12)[-1],
        ))
        np.testing.assert_array_equal(states[row], expected_state, err_msg=f"{parquet} row {row}")

    targets = canonical(native).reshape(-1, 114)
    time_s = (timestamps[indices] - timestamps[0]).reshape(-1)
    video_times = np.concatenate([
        row / VIDEO_FPS + timestamps[steps] - timestamps[centers[row]]
        for row, steps in zip(anchors, indices, strict=True)
    ])
    duration = len(centers) / VIDEO_FPS
    group = dict(
        id=eid, frames=len(targets), video_frames=len(centers), video_duration=duration,
        times=video_times.tolist(),
        chunks=[dict(id=index, anchor=row, anchor_video_s=row / VIDEO_FPS, segment=0,
                     scene=scene, start=index * ACTION_STEPS, stop=(index + 1) * ACTION_STEPS)
                for index, row in enumerate(anchor_rows)],
        segments=[dict(index=0, scene=scene, video_start=0., video_stop=duration)],
    )
    segments = [dict(segment_index=0, scene=scene, frame_start=0,
                     frame_stop_exclusive=len(centers), body_frames=body_frames)]
    write_json(output / f"episode-{eid}.json", dict(group=group, segments=segments))
    np.savez(output / f"episode-{eid}.npz", states=states, targets=targets,
             anchors=anchors, centers=centers[anchors],
             segment_ids=np.zeros(len(targets), dtype=np.int64), time_s=time_s)
    return group


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path, help="Task directory containing eval/.")
    source.add_argument("--dataset", type=Path, help="Path to the validation LeRobot dataset.")
    parser.add_argument("--output", type=Path, help="New reference directory (default: eval/prepared).")
    parser.add_argument("--episodes", type=int, nargs="+", help="Episode IDs to prepare (default: all).")
    parser.add_argument("--scene", default="fold shorts", help="Scene label in the evaluation report.")
    parser.add_argument("--head-camera", choices=("front_head", "fov_head"), default="front_head")
    args = parser.parse_args()

    if args.dataset_root:
        eval_dir = args.dataset_root.expanduser().resolve(strict=True) / "eval"
        candidates = sorted(path for path in eval_dir.iterdir() if path.name != "prepared"
                            and (path / "meta/info.json").is_file())
        if len(candidates) != 1:
            parser.error("Expected one validation LeRobot dataset under eval/. Choose it with --dataset.")
        dataset = candidates[0]
    else:
        dataset = args.dataset.expanduser().absolute()
        eval_dir = dataset.parent
    output = (args.output or eval_dir / "prepared").expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Already exists: {output}. Choose a new --output directory.")

    info, episodes = episode_metadata(dataset)
    by_id = {episode["episode_index"]: episode for episode in episodes}
    selected = args.episodes if args.episodes is not None else sorted(by_id)
    if len(set(selected)) != len(selected) or any(eid not in by_id for eid in selected):
        parser.error("--episodes must contain distinct IDs present in meta/episodes.jsonl.")

    sys.path.insert(0, str(REPO))
    from examples.unifolm_wla.open_loop_eval import canonical
    from unifolm_wla.dataloader.full_pipeline import cameras_for_head

    cameras = cameras_for_head(args.head_camera)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as scratch:
        stage = Path(scratch) / output.name
        stage.mkdir()
        groups = [prepare_episode(dataset, info, by_id[eid], stage, args.scene, canonical, cameras)
                  for eid in selected]
        chunks = sum(len(group["chunks"]) for group in groups)
        write_json(stage / "manifest.json",
                   dict(groups=groups, channels=evaluation_channels(), head_camera=args.head_camera))
        write_json(stage / "acceptance.json", dict(status="PASS", episodes=len(groups), chunks=chunks,
                                                 parquet_targets_equal=True))
        stage.rename(output)
    print(f"Prepared validation data: {len(groups)} episodes, {chunks} windows, {output}", flush=True)


if __name__ == "__main__":
    main()
