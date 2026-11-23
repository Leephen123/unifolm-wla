#!/usr/bin/env python3
"""Prepare frame-aligned surface masks and texture coordinates for training."""
import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from unifolm_wla.dataloader.background_recolor import _decode_mask
from unifolm_wla.dataloader.table_appearance import CACHE_VERSION, table_cache_path

MASKS = tuple(f"front_head_{part}_mask" for part in ("table", "floor", "clothes", "pants", "robot_arm"))


def _motion(previous, current, source, target):
    points = cv2.goodFeaturesToTrack(previous, 100, 0.006, 6, mask=source, blockSize=5)
    if points is None or len(points) < 8:
        return None
    settings = dict(winSize=(21, 21), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.04))
    following, forward, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **settings)
    if following is None:
        return None
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, following, None, **settings)
    if returned is None:
        return None
    first, second = points[:, 0], following[:, 0]
    keep = forward.ravel().astype(bool) & backward.ravel().astype(bool)
    keep &= np.linalg.norm(returned[:, 0] - first, axis=1) < 1.25
    x, y = np.rint(second[keep]).astype(np.int32).T
    height, width = target.shape
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid &= target[y.clip(0, height-1), x.clip(0, width-1)] != 0
    first, second = first[keep][valid], second[keep][valid]
    if len(first) < 8:
        return None
    affine, inliers = cv2.estimateAffinePartial2D(first, second, ransacReprojThreshold=1.5, maxIters=300)
    if affine is None:
        return None
    supported = inliers.ravel().astype(bool)
    if supported.sum() < 7 or supported.mean() < 0.5:
        return None
    if np.linalg.norm(np.ptp(first[supported], axis=0)) < 32:
        return None
    scale = np.linalg.norm(affine[:2, 0])
    rotation = abs(np.arctan2(affine[1, 0], affine[0, 0]))
    center = np.array((width / 2, height / 2))
    center_motion = np.linalg.norm(affine[:, :2] @ center + affine[:, 2] - center)
    if not (0.90 <= scale <= 1.11 and rotation <= 0.22 and center_motion <= 48):
        return None
    return np.vstack((affine, (0, 0, 1)))


def _reference_projection(table):
    y, x = np.nonzero(table)
    height, width = table.shape
    left, right = np.percentile(x, (2, 98))
    top, bottom = np.percentile(y, (2, 98))
    dx = max(right - left, 0.36 * width)
    dy = max(bottom - top, 0.30 * height)
    x_center, y_center = (left + right) / 2, (top + bottom) / 2
    far, near = y_center - dy / 2, y_center + dy / 2
    footprint = np.float32(((x_center - 0.4 * dx, far), (x_center + 0.4 * dx, far),
                            (x_center + 0.5 * dx, near), (x_center - 0.5 * dx, near)))
    return cv2.getPerspectiveTransform(footprint, np.float32(((0, 0), (1, 0), (1, 1), (0, 1))))


class SurfaceTracking:
    def __init__(self, surface):
        self.surface = surface
        self.reference = None
        self.pose = np.eye(3)
        self.previous = None
        self.stats = Counter()
        self.inset = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def project(self, gray, visible, background):
        if self.reference is None and visible.any():
            self.reference = _reference_projection(visible)
            self.previous = None
        current = (gray, cv2.erode(visible.astype(np.uint8), self.inset),
                   cv2.erode(background.astype(np.uint8), self.inset))
        if self.previous is not None and self.reference is not None:
            before, old_surface, old_background = self.previous
            step = _motion(before, gray, old_surface, current[1])
            status = self.surface
            if step is None:
                step = _motion(before, gray, old_background, current[2])
                status = "scene"
            if step is None:
                status = "held"
            else:
                self.pose = step @ self.pose
                scale = np.linalg.norm(self.pose[:2, 0])
                self.pose[:2, :2] *= np.clip(scale, 0.48, 2.8) / scale
            self.stats[status] += 1
        self.previous = current
        if visible.any():
            return (self.reference @ np.linalg.inv(self.pose)).astype("<f4").tobytes()
        return None


def _iter_gray_frames(video_source, frames):
    kind, path, feature_map = video_source
    if kind == "mpds":
        from mpds.dataset import MPDataset
        from mpds.device import mp_use_device
        from unifolm_wla.dataloader.mpds_backend import resolve_mpds_schema
        with mp_use_device(num_threads=1):
            with MPDataset.open(path) as ds:
                schema = resolve_mpds_schema(ds.metadata, head_camera="front_head", feature_map=feature_map)
                head_key = schema["front_head"]
                count = 0
                for row, (_ts, item) in enumerate(ds.select([head_key]).enumerate()):
                    if row >= frames:
                        raise ValueError(f"MPDS frame count exceeds mask count ({frames}): {path}.")
                    rgb = np.asarray(item[head_key], dtype=np.uint8)
                    size = (rgb.shape[1], rgb.shape[0])
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    yield row, size, gray
                    count += 1
                if count != frames:
                    raise ValueError(f"Decoded {count} of {frames} MPDS frames: {path}.")
        return
    with av.open(path) as video:
        stream, = video.streams.video
        stream.codec_context.thread_count = 1
        if stream.frames != frames or stream.average_rate != 20:
            raise ValueError(f"Video and mask metadata do not agree: {path}.")
        size = stream.width, stream.height
        count = 0
        for row, frame in enumerate(video.decode(stream)):
            if row >= frames or frame.pts * frame.time_base != row / stream.average_rate:
                raise ValueError(f"Video timestamp mismatch at {row}: {path}.")
            yield row, size, frame.to_ndarray(format="gray")
            count += 1
        if count != frames:
            raise ValueError(f"Decoded {count} of {frames} frames: {path}.")


def _write_episode(db, episode, video_source, masks_path, stats, surface):
    masks = pq.read_table(masks_path, columns=["frame_index", *MASKS], use_threads=False)
    frames = len(masks)
    if not np.array_equal(masks["frame_index"].to_numpy(), np.arange(frames)):
        raise ValueError(f"Mask frame indices are not aligned: {masks_path}.")
    columns = [masks[name] for name in MASKS]
    tracking = SurfaceTracking(surface)
    trims = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    records = []
    for row, size, gray in _iter_gray_frames(video_source, frames):
        table, floor, clothes, pants, arm = [_decode_mask(col[row].as_py(), size) for col in columns]
        foreground = clothes | pants | arm
        selected = table & ~floor if surface == "table" else floor & ~table
        selected &= ~foreground
        if selected.any():
            if surface == "table":
                # Remove narrow mask extensions onto table legs.
                opened = cv2.morphologyEx(selected.astype(np.uint8), cv2.MORPH_OPEN, trims).astype(bool)
                if opened.sum() >= 0.88 * selected.sum():
                    selected = opened
            packed = zlib.compress(np.packbits(selected).tobytes(), 1)
        else:
            packed = None
            stats[f"empty_{surface}_mask"] += 1
        projection = tracking.project(gray, selected, ~foreground)
        records.append((episode, row, packed, projection))
    if len(records) != frames:
        raise ValueError(f"Decoded {len(records)} of {frames} frames: {video_source[1]}.")
    db.executemany("INSERT INTO frames VALUES (?, ?, ?, ?)", records)
    stats.update(tracking.stats)
    stats.update(episodes=1, frames=frames)


def _existing(path, signature):
    if not path.is_file():
        return None
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
    if metadata["signature"] != signature:
        raise ValueError(f"Existing table cache has a different manifest: {path}.")
    return Counter(json.loads(metadata["counts"]))


def _prepare_group(job):
    dataset, entries, root, signature, surface = job
    path = table_cache_path(root, dataset)
    if (old := _existing(path, signature)) is not None:
        return dataset, old
    stats = Counter()
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix=".building-", dir=root) as temporary:
        staged = Path(temporary) / path.name
        with closing(sqlite3.connect(staged)) as db, db:
            db.execute("CREATE TABLE frames(episode INTEGER, row INTEGER, mask BLOB, projection BLOB, "
                       "PRIMARY KEY(episode, row)) WITHOUT ROWID")
            db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            for episode, video_path, masks_path in entries:
                _write_episode(db, episode, video_path, masks_path, stats, surface)
            db.executemany("INSERT INTO metadata VALUES (?, ?)",
                           (("signature", signature), ("counts", json.dumps(stats))))
        try:
            os.link(staged, path)
        except FileExistsError:
            if _existing(path, signature) != stats:
                raise
    return dataset, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--surface", choices=("table", "floor"), default="table")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("background_mask_camera") != "front_head" or not manifest["episodes"]:
        raise ValueError("Expected a nonempty front_head manifest with attached masks.")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    groups = defaultdict(list)
    for entry in manifest["episodes"]:
        if "mpds_path" in entry:
            source = ("mpds", entry["mpds_path"], entry.get("mpds_feature_map"))
        else:
            source = ("video", entry["videos"]["front_head"], None)
        groups[entry["dataset"]].append((entry["episode_index"], source,
                                         entry["background_masks_parquet"]))
    # Preserve signatures for pre-existing tabletop caches.
    signature = f"{CACHE_VERSION}:{digest}" if args.surface == "table" else f"{CACHE_VERSION}:{digest}:floor"
    jobs = [(name, entries, root, f"{signature}:{name}", args.surface) for name, entries in groups.items()]
    stats = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(_prepare_group, job) for job in jobs]
        for i, task in enumerate(as_completed(tasks), 1):
            dataset, counts = task.result()
            stats.update(counts)
            print(f"{i}/{len(groups)} {Path(dataset).name}: {dict(counts)}", flush=True)
    if stats["episodes"] != len(manifest["episodes"]) or stats["frames"] != manifest["total_frames"]:
        raise ValueError(f"Appearance cache counts do not match the training manifest: {stats}.")
    index = dict(version=CACHE_VERSION, manifest_sha256=digest, surface=args.surface, datasets=len(groups), **stats)
    if stats[f"empty_{args.surface}_mask"] == stats["frames"]:
        raise ValueError(f"No {args.surface} is available in the selected dataset.")
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        marker = Path(temporary) / "index.json"
        marker.write_text(json.dumps(index, indent=2) + "\n")
        marker.replace(root / marker.name)
    print(json.dumps(index), flush=True)


if __name__ == "__main__":
    main()
