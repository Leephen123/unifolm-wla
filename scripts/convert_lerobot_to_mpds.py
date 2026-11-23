#!/usr/bin/env python3
"""Convert LeRobot v2.1 G1 datasets (or a prepared WLA manifest) into MPDS (.mpds) files."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import json
from pathlib import Path
import sys

import av
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from unifolm_wla.dataloader.full_pipeline import cameras_for_head, load_body, load_state_valid_episode


CAM_STREAM_NAMES = {
    "front_head": "camera/front/rgb",
    "fov_head": "camera/fov/rgb",
    "left_hand": "camera/left/rgb",
    "right_hand": "camera/right/rgb",
}

VIDEO_MIME = {
    "h264": "video/h264",
    "h265": "video/h265",
    "av1": "video/av1",
    "ffv1": "video/ffv1",
}


def _decode_all_rgb(video_path: str) -> list[np.ndarray]:
    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _convert_one(job: dict) -> tuple[str, int, int, int]:
    from mpds.dataset import MPDataset
    from mpds.device import mp_use_device

    entry = job["entry"]
    out_path = Path(job["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layout = job["layout"]
    head_camera = job["head_camera"]
    video_mime = VIDEO_MIME[job["video_codec"]]
    prompt = job["prompt"]
    async_state_fps = job.get("async_state_fps")

    cameras = cameras_for_head(head_camera)
    cam_frames = {key: _decode_all_rgb(entry["videos"][key]) for key, _ in cameras}
    num_frames = len(cam_frames[head_camera])
    for key, frs in cam_frames.items():
        if len(frs) != num_frames:
            raise ValueError(f"Camera {key} length {len(frs)} != {num_frames} in {entry['dataset']} ep {entry['episode_index']}")

    if "state_valid_parquet" in entry:
        states_all, actions_all, sv_all, av_all = load_state_valid_episode(entry["state_valid_parquet"])
        load_state_valid_episode.cache_clear()
        if len(states_all) != num_frames:
            raise ValueError(f"Parquet rows {len(states_all)} != video frames {num_frames}")
        source_has_packed = True
    else:
        z = load_body(entry["body"])
        load_body.cache_clear()
        centers = z["centers"]
        if len(centers) != num_frames:
            raise ValueError(f"Body centers {len(centers)} != video frames {num_frames}")
        source_has_packed = False

    cam_tb = Fraction(1, 20)
    state_tb = Fraction(1, int(async_state_fps)) if (layout == "single_frame" and async_state_fps) else cam_tb

    features = {
        CAM_STREAM_NAMES[key]: {"type": video_mime, "time_base": str(cam_tb)}
        for key, _ in cameras
    }
    features.update({
        "robot/g1/arm_state": {"type": "application/vnd.mpds.arraycodec", "time_base": str(state_tb)},
        "robot/g1/hand_state": {"type": "application/vnd.mpds.arraycodec", "time_base": str(state_tb)},
        "robot/g1/arm_actions": {"type": "application/vnd.mpds.arraycodec", "time_base": str(cam_tb)},
        "robot/g1/hand_actions": {"type": "application/vnd.mpds.arraycodec", "time_base": str(cam_tb)},
        "prompt/texts": {"type": "application/json", "time_base": str(cam_tb)},
    })
    if layout == "packed":
        features["robot/g1/state_valid"] = {"type": "application/vnd.mpds.arraycodec", "time_base": str(cam_tb)}
        features["robot/g1/action_valid"] = {"type": "application/vnd.mpds.arraycodec", "time_base": str(cam_tb)}
    elif source_has_packed:
        features["robot/g1/state_valid"] = {"type": "application/vnd.mpds.arraycodec", "time_base": str(state_tb)}
        features["robot/g1/action_valid"] = {"type": "application/vnd.mpds.arraycodec", "time_base": str(cam_tb)}

    tmp_path = out_path.with_suffix(".mpds.tmp")
    with mp_use_device(num_threads=1):
        with MPDataset.create(str(tmp_path), metadata={"features": features}) as ds:
            if layout == "packed":
                if not source_has_packed:
                    raise ValueError("--layout packed requires state_valid_parquet source episodes.")
                for i in range(num_frames):
                    t = cam_tb * i
                    data = {CAM_STREAM_NAMES[key]: cam_frames[key][i] for key, _ in cameras}
                    data["robot/g1/arm_state"] = states_all[i, :, :35].astype(np.float32, copy=False)
                    data["robot/g1/hand_state"] = states_all[i, :, 35:].astype(np.float32, copy=False)
                    data["robot/g1/arm_actions"] = actions_all[i, :, :102].astype(np.float32, copy=False)
                    data["robot/g1/hand_actions"] = actions_all[i, :, 102:].astype(np.float32, copy=False)
                    data["robot/g1/state_valid"] = sv_all[i].astype(bool, copy=False)
                    data["robot/g1/action_valid"] = np.asarray([bool(av_all[i])], dtype=bool)
                    data["prompt/texts"] = [prompt]
                    ds.append(time=t, data=data)
            else:
                # single_frame layout: per-frame state (35 + 12) + packed 50-step or single-frame actions
                state_step_stride = 1
                if async_state_fps is not None:
                    if 20 % int(async_state_fps) != 0:
                        raise ValueError("--async-state-fps must divide 20 (e.g. 10 or 20).")
                    state_step_stride = 20 // int(async_state_fps)

                for i in range(num_frames):
                    t = cam_tb * i
                    data = {CAM_STREAM_NAMES[key]: cam_frames[key][i] for key, _ in cameras}
                    if source_has_packed:
                        if i % state_step_stride == 0:
                            data["robot/g1/arm_state"] = states_all[i, -1, :35].astype(np.float32, copy=False)
                            data["robot/g1/hand_state"] = states_all[i, -1, 35:].astype(np.float32, copy=False)
                            data["robot/g1/state_valid"] = np.asarray([bool(sv_all[i, -1])], dtype=bool)
                        # Store per-frame 50-step packed actions (or single-frame if body)
                        data["robot/g1/arm_actions"] = actions_all[i, :, :102].astype(np.float32, copy=False)
                        data["robot/g1/hand_actions"] = actions_all[i, :, 102:].astype(np.float32, copy=False)
                        data["robot/g1/action_valid"] = np.asarray([bool(av_all[i])], dtype=bool)
                    else:
                        c = int(centers[i])
                        if i % state_step_stride == 0:
                            data["robot/g1/arm_state"] = z["arm_state"][c].astype(np.float32, copy=False)
                            data["robot/g1/hand_state"] = z["hand_state"][c].astype(np.float32, copy=False)
                        data["robot/g1/arm_actions"] = z["arm_actions"][c].astype(np.float32, copy=False)
                        data["robot/g1/hand_actions"] = z["hand_actions"][c].astype(np.float32, copy=False)
                    data["prompt/texts"] = [prompt]
                    ds.append(time=t, data=data)

    tmp_path.replace(out_path)
    return str(out_path), entry["episode_index"], num_frames, out_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Input prepared WLA manifest.json.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for converted .mpds episodes.")
    parser.add_argument("--layout", choices=("packed", "single_frame"), default="packed")
    parser.add_argument("--video-codec", choices=("h264", "h265", "av1", "ffv1"), default="h264")
    parser.add_argument("--async-state-fps", type=int, default=None,
                        help="Optional lower state FPS (e.g. 10) for single_frame async testing.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N episodes.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    head_camera = manifest.get("head_camera", "front_head")
    prompt = manifest["prompt"]
    episodes = manifest["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for entry in episodes:
        ds_name = Path(entry["dataset"]).name
        eid = int(entry["episode_index"])
        out_path = out_root / ds_name / f"episode_{eid:06d}.mpds"
        jobs.append({
            "entry": entry,
            "out_path": str(out_path),
            "layout": args.layout,
            "head_camera": head_camera,
            "video_codec": args.video_codec,
            "prompt": prompt,
            "async_state_fps": args.async_state_fps,
        })

    total_bytes = 0
    total_frames = 0
    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            p, eid, nf, sz = _convert_one(job)
            total_bytes += sz
            total_frames += nf
            print(f"[{i}/{len(jobs)}] {Path(p).parent.name}/{Path(p).name}: {nf} frames, {sz/1e6:.2f} MB", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_convert_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                p, eid, nf, sz = fut.result()
                total_bytes += sz
                total_frames += nf
                print(f"[{i}/{len(jobs)}] {Path(p).parent.name}/{Path(p).name}: {nf} frames, {sz/1e6:.2f} MB", flush=True)

    print(f"Converted {len(jobs)} episodes ({total_frames} frames, {total_bytes/1e6:.2f} MB) -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
