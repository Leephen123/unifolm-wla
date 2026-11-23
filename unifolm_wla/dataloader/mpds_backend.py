"""High-performance MPDS episode reader for UnifoLM-WLA full-pipeline training.

Supports both:
  1. ``packed`` episodes: each frame stores pre-packed long-history state
     (100 or 200 slots of arm_state 35 + hand_state 12), state_valid,
     50-step packed actions (arm_actions 5100 + hand_actions 600), and action_valid.
  2. ``single_frame`` episodes: each frame stores per-frame state (arm 35 + hand 12,
     or combined 47) and per-frame or 50-step packed actions, supporting both
     synchronous and asynchronous multi-rate sensor timestamps.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Default candidate stream names in MPDS containers (supports both WLA and openpi conventions).
MPDS_CAMERA_CANDIDATES: dict[str, tuple[str, ...]] = {
    "front_head": ("camera/front/rgb", "observation/images/front_head", "front_head"),
    "fov_head": ("camera/fov/rgb", "observation/images/fov_head", "fov_head"),
    "left_hand": ("camera/left/rgb", "observation/images/left_hand", "left_hand"),
    "right_hand": ("camera/right/rgb", "observation/images/right_hand", "right_hand"),
}

MPDS_FEATURE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "arm_state": ("robot/g1/arm_state", "robot/dualpiper/dof_positions", "arm_state"),
    "hand_state": ("robot/g1/hand_state", "robot/dualpiper/hand_state", "hand_state"),
    "state": ("robot/g1/state", "state"),
    "arm_actions": ("robot/g1/arm_actions", "robot/g1/arm_action", "robot/dualpiper/actions", "arm_actions", "arm_action"),
    "hand_actions": ("robot/g1/hand_actions", "robot/g1/hand_action", "robot/dualpiper/hand_actions", "hand_actions", "hand_action"),
    "actions": ("robot/g1/actions", "robot/g1/action", "actions", "action"),
    "state_valid": ("robot/g1/state_valid", "state_valid"),
    "action_valid": ("robot/g1/action_valid", "action_valid"),
    "prompt": ("prompt/texts", "prompt", "task"),
}

ACTION_STEPS = 50
HISTORY_FRAMES = 50
HISTORY_STEP_SEC = 0.1  # 10 Hz history (-5.0s .. -0.1s)
PARQUET_HISTORY_SLOTS = {
    100: slice(-51, -1),
    200: slice(-101, -1, 2),
}

# Per-worker bounded LRU caches (spawn-safe).
_mpds_handles: OrderedDict[str, tuple[Any, dict[str, str], dict[str, Fraction]]] = OrderedDict()
_mpds_numeric_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_HANDLE_CACHE_SIZE = 32
_NUMERIC_CACHE_SIZE = 32


def _fraction_from_meta(meta_entry: dict[str, Any], default: Fraction = Fraction(1, 20)) -> Fraction:
    tb = meta_entry.get("time_base")
    if tb is None:
        return default
    return Fraction(tb)


def resolve_mpds_schema(
    metadata: dict[str, Any],
    head_camera: str = "front_head",
    feature_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve logical WLA roles to concrete MPDS feature stream names."""
    features = metadata.get("features", {})
    available = set(features.keys())
    overrides = dict(feature_map or {})
    resolved: dict[str, str] = {}

    def pick(role: str, candidates: tuple[str, ...], required: bool = False) -> None:
        if role in overrides and overrides[role]:
            key = overrides[role]
            if key not in available:
                raise KeyError(f"Configured MPDS feature {role}={key!r} not found in {sorted(available)}")
            resolved[role] = key
            return
        for cand in candidates:
            if cand in available:
                resolved[role] = cand
                return
        if required:
            raise KeyError(
                f"MPDS episode lacks required feature for role {role!r}; "
                f"tried {candidates}, available={sorted(available)}"
            )

    for cam_role in (head_camera, "left_hand", "right_hand"):
        pick(cam_role, MPDS_CAMERA_CANDIDATES.get(cam_role, (cam_role,)), required=True)

    for role, cands in MPDS_FEATURE_CANDIDATES.items():
        pick(role, cands, required=False)

    # Support split openpi-style naming where arm stream is named `state`/`action` alongside `hand_state`/`hand_action`.
    if "state" in resolved and "hand_state" in resolved and "arm_state" not in resolved:
        resolved["arm_state"] = resolved.pop("state")
    if "actions" in resolved and "hand_actions" in resolved and "arm_actions" not in resolved:
        resolved["arm_actions"] = resolved.pop("actions")

    has_state = ("state" in resolved) or ("arm_state" in resolved and "hand_state" in resolved)
    has_actions = ("actions" in resolved) or ("arm_actions" in resolved and "hand_actions" in resolved)
    if not has_state:
        raise KeyError(f"MPDS episode lacks state streams; available={sorted(available)}")
    if not has_actions:
        raise KeyError(f"MPDS episode lacks action streams; available={sorted(available)}")

    return resolved


def _open_mpds(path: str, head_camera: str = "front_head", feature_map: dict[str, str] | None = None):
    """Return a cached (MPDataset, schema, time_bases) tuple with single-threaded FFmpeg decoding."""
    cache_key = f"{path}|{head_camera}|{sorted((feature_map or {}).items())}"
    cached = _mpds_handles.get(cache_key)
    if cached is not None:
        _mpds_handles.move_to_end(cache_key)
        return cached

    if len(_mpds_handles) >= _HANDLE_CACHE_SIZE:
        _, (old_ds, _, _) = _mpds_handles.popitem(last=False)
        try:
            old_ds.close()
        except Exception:
            pass

    from mpds.dataset import MPDataset
    from mpds.device import mp_use_device

    with mp_use_device(num_threads=1):
        ds = MPDataset.open(str(path))
        meta = ds.metadata
        schema = resolve_mpds_schema(meta, head_camera=head_camera, feature_map=feature_map)
        time_bases = {
            k: _fraction_from_meta(meta["features"][k])
            for k in meta["features"]
        }
    entry = (ds, schema, time_bases)
    _mpds_handles[cache_key] = entry
    return entry


def clear_mpds_caches() -> None:
    """Close all cached MPDS handles and clear numeric caches."""
    while _mpds_handles:
        _, (ds, _, _) = _mpds_handles.popitem(last=False)
        try:
            ds.close()
        except Exception:
            pass
    _mpds_numeric_cache.clear()


def _extract_prompt(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)) and val:
        return str(val[0])
    if isinstance(val, dict):
        for k in ("prompt", "text", "task", "lang"):
            if k in val and val[k]:
                return str(val[k])
    return str(val)


def _read_stream_all(ds, key: str) -> tuple[np.ndarray, list[Any]]:
    """Read all (timestamp, value) pairs for a single non-video stream."""
    ts_list: list[float] = []
    vals: list[Any] = []
    for t, item in ds.select([key]).enumerate():
        ts_list.append(float(t))
        vals.append(item[key])
    return np.asarray(ts_list, dtype=np.float64), vals


def _read_single_frame_at_time(ds, key: str, t: Fraction, dt: Fraction):
    """Read the sample in [t, t + dt) for stream key via non-deprecated traverse."""
    from mpds._utils.interval import Interval

    window = Interval(t, t + dt)
    for _, _, val in ds.select([key]).traverse(time_range=window):
        return val
    # Fallback: expand slightly for minor timestamp rounding differences
    eps = dt / 2
    best_dist = None
    best_val = None
    for ts, _, val in ds.select([key]).traverse(time_range=Interval(max(Fraction(0), t - eps), t + dt + eps)):
        dist = abs(float(ts - t))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_val = val
    if best_val is None:
        raise IndexError(f"No sample found for MPDS feature {key!r} near time={t} ({float(t):.4f}s)")
    return best_val


def _parse_state_sample(
    arm_val: Any | None,
    hand_val: Any | None,
    combined_val: Any | None,
    sv_val: Any | None,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Normalize a state sample into (states_2d_or_1d, state_valid_or_None, is_packed)."""
    if combined_val is not None:
        arr = np.asarray(combined_val, dtype=np.float64)
        if arr.ndim == 1 and arr.size == 47:
            return arr, None, False
        if arr.ndim == 2 and arr.shape[1] == 47 and arr.shape[0] in PARQUET_HISTORY_SLOTS:
            sv = np.ones(arr.shape[0], dtype=bool) if sv_val is None else np.asarray(sv_val, dtype=bool).ravel()
            return arr, sv, True
        if arr.ndim == 1 and (arr.size // 47) in PARQUET_HISTORY_SLOTS and arr.size % 47 == 0:
            steps = arr.size // 47
            sv = np.ones(steps, dtype=bool) if sv_val is None else np.asarray(sv_val, dtype=bool).ravel()
            return arr.reshape(steps, 47), sv, True
        raise ValueError(f"Unsupported combined state shape: {arr.shape}")

    arm = np.asarray(arm_val, dtype=np.float64)
    hand = np.asarray(hand_val, dtype=np.float64)
    if arm.ndim == 1 and arm.size == 35 and hand.ndim == 1 and hand.size == 12:
        return np.concatenate((arm, hand), axis=-1), None, False

    # Packed long-history state (100 or 200 slots)
    if arm.ndim == 2 and arm.shape[1] == 35 and hand.ndim == 2 and hand.shape == (arm.shape[0], 12):
        steps = arm.shape[0]
        if steps in PARQUET_HISTORY_SLOTS:
            sv = np.ones(steps, dtype=bool) if sv_val is None else np.asarray(sv_val, dtype=bool).ravel()
            return np.concatenate((arm, hand), axis=-1), sv, True
    if arm.ndim == 1 and arm.size % 35 == 0 and hand.ndim == 1 and hand.size % 12 == 0:
        steps = arm.size // 35
        if steps in PARQUET_HISTORY_SLOTS and hand.size // 12 == steps:
            sv = np.ones(steps, dtype=bool) if sv_val is None else np.asarray(sv_val, dtype=bool).ravel()
            return np.concatenate((arm.reshape(steps, 35), hand.reshape(steps, 12)), axis=-1), sv, True
    raise ValueError(f"Unsupported arm/hand state shapes: arm={arm.shape}, hand={hand.shape}")


def _parse_action_sample(
    arm_val: Any | None,
    hand_val: Any | None,
    combined_val: Any | None,
) -> tuple[np.ndarray, bool]:
    """Normalize an action sample into (action_array, is_packed_50x114)."""
    if combined_val is not None:
        arr = np.asarray(combined_val, dtype=np.float64)
        if arr.shape == (ACTION_STEPS, 114):
            return arr, True
        if arr.ndim == 1 and arr.size == ACTION_STEPS * 114:
            return arr.reshape(ACTION_STEPS, 114), True
        if arr.ndim == 1 and arr.size == 114:
            return arr, False
        raise ValueError(f"Unsupported combined action shape: {arr.shape}")

    arm = np.asarray(arm_val, dtype=np.float64)
    hand = np.asarray(hand_val, dtype=np.float64)
    if arm.shape == (ACTION_STEPS, 102) and hand.shape == (ACTION_STEPS, 12):
        return np.concatenate((arm, hand), axis=-1), True
    if arm.ndim == 1 and arm.size == ACTION_STEPS * 102 and hand.ndim == 1 and hand.size == ACTION_STEPS * 12:
        return np.concatenate((arm.reshape(ACTION_STEPS, 102), hand.reshape(ACTION_STEPS, 12)), axis=-1), True
    if arm.ndim == 1 and arm.size == 102 and hand.ndim == 1 and hand.size == 12:
        return np.concatenate((arm, hand), axis=-1), False
    raise ValueError(f"Unsupported arm/hand action shapes: arm={arm.shape}, hand={hand.shape}")


def load_mpds_numeric_streams(
    path: str,
    head_camera: str = "front_head",
    feature_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load and cache all lightweight numeric streams of an MPDS episode in RAM."""
    cache_key = f"{path}|{head_camera}|{sorted((feature_map or {}).items())}"
    cached = _mpds_numeric_cache.get(cache_key)
    if cached is not None:
        _mpds_numeric_cache.move_to_end(cache_key)
        return cached

    if len(_mpds_numeric_cache) >= _NUMERIC_CACHE_SIZE:
        _mpds_numeric_cache.popitem(last=False)

    ds, schema, time_bases = _open_mpds(path, head_camera=head_camera, feature_map=feature_map)
    head_key = schema[head_camera]
    head_tb = time_bases[head_key]

    # Collect head camera timestamps (fast packet/timeline scan without video decode)
    head_ts = list(ds.select([head_key]).timeline)

    out: dict[str, Any] = {
        "schema": schema,
        "head_time_base": head_tb,
        "head_timestamps": head_ts,
        "head_ts_f64": np.asarray([float(t) for t in head_ts], dtype=np.float64),
    }

    # State stream(s)
    if "state" in schema:
        st_ts, st_vals = _read_stream_all(ds, schema["state"])
        first_arr, _, state_packed = _parse_state_sample(None, None, st_vals[0], None)
        out["state_packed"] = state_packed
        out["state_ts"] = st_ts
        if not state_packed:
            out["state_arr"] = np.stack([np.asarray(v, dtype=np.float64).reshape(47) for v in st_vals], axis=0)
    else:
        arm_ts, arm_vals = _read_stream_all(ds, schema["arm_state"])
        hand_ts, hand_vals = _read_stream_all(ds, schema["hand_state"])
        _, _, state_packed = _parse_state_sample(arm_vals[0], hand_vals[0], None, None)
        out["state_packed"] = state_packed
        out["state_ts"] = arm_ts
        if not state_packed:
            arm_arr = np.stack([np.asarray(v, dtype=np.float64).reshape(35) for v in arm_vals], axis=0)
            hand_arr = np.stack([np.asarray(v, dtype=np.float64).reshape(12) for v in hand_vals], axis=0)
            if len(arm_ts) == len(hand_ts) and np.allclose(arm_ts, hand_ts, atol=1e-6):
                out["state_arr"] = np.concatenate((arm_arr, hand_arr), axis=-1)
            else:
                # Align hand_state onto arm_state timeline via previous-value hold
                idx = np.clip(np.searchsorted(hand_ts, arm_ts + 1e-6, side="right") - 1, 0, len(hand_ts) - 1)
                out["state_arr"] = np.concatenate((arm_arr, hand_arr[idx]), axis=-1)

    # Action stream(s)
    if "actions" in schema:
        act_ts, act_vals = _read_stream_all(ds, schema["actions"])
        _, act_packed = _parse_action_sample(None, None, act_vals[0])
        out["action_packed"] = act_packed
        out["action_ts"] = act_ts
        if not act_packed:
            out["action_arr"] = np.stack([np.asarray(v, dtype=np.float64).reshape(114) for v in act_vals], axis=0)
        else:
            out["action_arr"] = np.stack(
                [np.asarray(v, dtype=np.float64).reshape(ACTION_STEPS, 114) for v in act_vals], axis=0
            )
    else:
        a_ts, a_vals = _read_stream_all(ds, schema["arm_actions"])
        h_ts, h_vals = _read_stream_all(ds, schema["hand_actions"])
        _, act_packed = _parse_action_sample(a_vals[0], h_vals[0], None)
        out["action_packed"] = act_packed
        out["action_ts"] = a_ts
        if not act_packed:
            a_arr = np.stack([np.asarray(v, dtype=np.float64).reshape(102) for v in a_vals], axis=0)
            h_arr = np.stack([np.asarray(v, dtype=np.float64).reshape(12) for v in h_vals], axis=0)
            if len(a_ts) == len(h_ts) and np.allclose(a_ts, h_ts, atol=1e-6):
                out["action_arr"] = np.concatenate((a_arr, h_arr), axis=-1)
            else:
                idx = np.clip(np.searchsorted(h_ts, a_ts + 1e-6, side="right") - 1, 0, len(h_ts) - 1)
                out["action_arr"] = np.concatenate((a_arr, h_arr[idx]), axis=-1)
        else:
            a_arr = np.stack([np.asarray(v, dtype=np.float64).reshape(ACTION_STEPS, 102) for v in a_vals], axis=0)
            h_arr = np.stack([np.asarray(v, dtype=np.float64).reshape(ACTION_STEPS, 12) for v in h_vals], axis=0)
            out["action_arr"] = np.concatenate((a_arr, h_arr), axis=-1)

    # Optional validity & prompt streams
    if "action_valid" in schema:
        av_ts, av_vals = _read_stream_all(ds, schema["action_valid"])
        out["action_valid_ts"] = av_ts
        out["action_valid"] = np.asarray([bool(np.asarray(v).ravel()[0]) for v in av_vals], dtype=bool)
    if "state_valid" in schema and not out["state_packed"]:
        sv_ts, sv_vals = _read_stream_all(ds, schema["state_valid"])
        out["state_valid_ts"] = sv_ts
        out["state_valid"] = np.asarray([bool(np.asarray(v).ravel()[0]) for v in sv_vals], dtype=bool)
    if "prompt" in schema:
        p_ts, p_vals = _read_stream_all(ds, schema["prompt"])
        out["prompt_ts"] = p_ts
        out["prompt_vals"] = [_extract_prompt(v) for v in p_vals]

    _mpds_numeric_cache[cache_key] = out
    return out


def inspect_mpds_episode(
    path: str | Path,
    head_camera: str = "front_head",
    feature_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Inspect an MPDS episode for preparation: detect layout, valid anchors, states, actions."""
    from mpds.dataset import MPDataset
    from mpds.device import mp_use_device

    path_str = str(Path(path).resolve(strict=True))
    with mp_use_device(num_threads=1):
        with MPDataset.open(path_str) as ds:
            meta = ds.metadata
            schema = resolve_mpds_schema(meta, head_camera=head_camera, feature_map=feature_map)
            head_key = schema[head_camera]
            head_tb = _fraction_from_meta(meta["features"][head_key])
            head_ts = list(ds.select([head_key]).timeline)
            head_ts_f64 = np.asarray([float(t) for t in head_ts], dtype=np.float64)
            num_frames = len(head_ts)
            if num_frames == 0:
                raise ValueError(f"Empty MPDS episode: {path_str}")

            # Read first sample of state & action to detect packed vs single_frame
            if "state" in schema:
                st_ts, st_vals = _read_stream_all(ds, schema["state"])
                _, _, state_packed = _parse_state_sample(None, None, st_vals[0], None)
            else:
                arm_ts, arm_vals = _read_stream_all(ds, schema["arm_state"])
                hand_ts, hand_vals = _read_stream_all(ds, schema["hand_state"])
                _, _, state_packed = _parse_state_sample(arm_vals[0], hand_vals[0], None, None)
                st_ts = arm_ts

            if "actions" in schema:
                act_ts, act_vals = _read_stream_all(ds, schema["actions"])
                _, action_packed = _parse_action_sample(None, None, act_vals[0])
            else:
                a_ts, a_vals = _read_stream_all(ds, schema["arm_actions"])
                h_ts, h_vals = _read_stream_all(ds, schema["hand_actions"])
                _, action_packed = _parse_action_sample(a_vals[0], h_vals[0], None)
                act_ts = a_ts

            # Optional validity flags
            action_valid_arr = None
            if "action_valid" in schema:
                _, av_vals = _read_stream_all(ds, schema["action_valid"])
                action_valid_arr = np.asarray([bool(np.asarray(v).ravel()[0]) for v in av_vals], dtype=bool)
            state_valid_raw = None
            if "state_valid" in schema:
                _, sv_vals = _read_stream_all(ds, schema["state_valid"])
                state_valid_raw = sv_vals

            prompt_text = None
            if "prompt" in schema:
                for _, item in ds.select([schema["prompt"]]).enumerate():
                    prompt_text = _extract_prompt(item[schema["prompt"]])
                    if prompt_text:
                        break

            # Build per-anchor state (current 47-dim) & valid_rows & action windows for statistics
            layout = "packed" if (state_packed and action_packed) else "single_frame"

            # Prepare arrays for statistics
            if state_packed:
                cur_states = []
                cur_state_valid = []
                for i in range(len(st_ts)):
                    sv_i = state_valid_raw[i] if state_valid_raw is not None else None
                    if "state" in schema:
                        s_mat, sv_arr, _ = _parse_state_sample(None, None, st_vals[i], sv_i)
                    else:
                        s_mat, sv_arr, _ = _parse_state_sample(arm_vals[i], hand_vals[i], None, sv_i)
                    cur_states.append(s_mat[-1])
                    cur_state_valid.append(bool(sv_arr[-1]))
                cur_states_arr = np.stack(cur_states, axis=0)
                cur_state_valid_arr = np.asarray(cur_state_valid, dtype=bool)
            else:
                if "state" in schema:
                    raw_s = np.stack([np.asarray(v, dtype=np.float64).reshape(47) for v in st_vals], axis=0)
                else:
                    arm_a = np.stack([np.asarray(v, dtype=np.float64).reshape(35) for v in arm_vals], axis=0)
                    hand_a = np.stack([np.asarray(v, dtype=np.float64).reshape(12) for v in hand_vals], axis=0)
                    if len(arm_ts) == len(hand_ts) and np.allclose(arm_ts, hand_ts, atol=1e-6):
                        raw_s = np.concatenate((arm_a, hand_a), axis=-1)
                    else:
                        hi = np.clip(np.searchsorted(hand_ts, arm_ts + 1e-6, side="right") - 1, 0, len(hand_ts) - 1)
                        raw_s = np.concatenate((arm_a, hand_a[hi]), axis=-1)
                # Align to head_ts_f64
                si = np.clip(np.searchsorted(st_ts, head_ts_f64 + 1e-6, side="right") - 1, 0, len(st_ts) - 1)
                cur_states_arr = raw_s[si]
                if state_valid_raw is not None:
                    sv_b = np.asarray([bool(np.asarray(v).ravel()[0]) for v in state_valid_raw], dtype=bool)
                    cur_state_valid_arr = sv_b[si]
                else:
                    cur_state_valid_arr = np.ones(num_frames, dtype=bool)

            if action_packed:
                if "actions" in schema:
                    packed_acts = np.stack([np.asarray(v, dtype=np.float64).reshape(ACTION_STEPS, 114) for v in act_vals], axis=0)
                else:
                    aa = np.stack([np.asarray(v, dtype=np.float64).reshape(ACTION_STEPS, 102) for v in a_vals], axis=0)
                    ha = np.stack([np.asarray(v, dtype=np.float64).reshape(ACTION_STEPS, 12) for v in h_vals], axis=0)
                    packed_acts = np.concatenate((aa, ha), axis=-1)
                if len(packed_acts) != num_frames:
                    ai = np.clip(np.searchsorted(act_ts, head_ts_f64 + 1e-6, side="right") - 1, 0, len(act_ts) - 1)
                    packed_acts = packed_acts[ai]
                    if action_valid_arr is not None:
                        action_valid_arr = action_valid_arr[ai]
                valid_mask = action_valid_arr if action_valid_arr is not None else np.ones(num_frames, dtype=bool)
                valid_rows = np.flatnonzero(valid_mask).tolist()
                valid_actions = packed_acts[valid_rows]
            else:
                if "actions" in schema:
                    raw_a = np.stack([np.asarray(v, dtype=np.float64).reshape(114) for v in act_vals], axis=0)
                else:
                    aa = np.stack([np.asarray(v, dtype=np.float64).reshape(102) for v in a_vals], axis=0)
                    ha = np.stack([np.asarray(v, dtype=np.float64).reshape(12) for v in h_vals], axis=0)
                    if len(a_ts) == len(h_ts) and np.allclose(a_ts, h_ts, atol=1e-6):
                        raw_a = np.concatenate((aa, ha), axis=-1)
                    else:
                        hi = np.clip(np.searchsorted(h_ts, a_ts + 1e-6, side="right") - 1, 0, len(h_ts) - 1)
                        raw_a = np.concatenate((aa, ha[hi]), axis=-1)
                # Anchor row r is valid iff there are ACTION_STEPS future action samples after head_ts[r]
                valid_rows = []
                action_windows = []
                for r in range(num_frames):
                    if action_valid_arr is not None and r < len(action_valid_arr) and not action_valid_arr[r]:
                        continue
                    pos = int(np.searchsorted(act_ts, head_ts_f64[r] + 1e-6, side="right"))
                    if pos + ACTION_STEPS <= len(raw_a):
                        valid_rows.append(r)
                        action_windows.append(raw_a[pos : pos + ACTION_STEPS])
                valid_actions = np.stack(action_windows, axis=0) if action_windows else np.zeros((0, ACTION_STEPS, 114), dtype=np.float64)

            if not np.isfinite(cur_states_arr[cur_state_valid_arr]).all():
                raise ValueError(f"Nonfinite state values in MPDS episode: {path_str}")
            if len(valid_actions) and not np.isfinite(valid_actions).all():
                raise ValueError(f"Nonfinite action values in MPDS episode: {path_str}")

            return {
                "path": path_str,
                "schema": schema,
                "layout": layout,
                "state_packed": state_packed,
                "action_packed": action_packed,
                "head_time_base": str(head_tb),
                "num_frames": num_frames,
                "valid_rows": valid_rows,
                "row_timestamps": [str(t) for t in head_ts],
                "current_states": cur_states_arr,
                "current_state_valid": cur_state_valid_arr,
                "valid_actions": valid_actions,
                "prompt": prompt_text,
            }


def read_mpds_sample(
    path: str,
    row: int,
    *,
    head_camera: str = "front_head",
    history_frames: int = 0,
    feature_map: dict[str, str] | None = None,
    row_timestamps: list[str] | None = None,
    layout: str | None = None,
) -> tuple[list[Image.Image], np.ndarray, bool, np.ndarray, np.ndarray | None, np.ndarray | None, str | None]:
    """Read one training sample from an MPDS episode.

    Returns:
      (images_pil_rgb_list, raw_state47, state_is_valid, action_50x114,
       history_raw_50x47_or_None, history_valid_50_or_None, prompt_or_None)
    """
    ds, schema, time_bases = _open_mpds(path, head_camera=head_camera, feature_map=feature_map)
    head_key = schema[head_camera]
    head_tb = time_bases[head_key]

    if row_timestamps is not None:
        t = Fraction(row_timestamps[row])
    else:
        t = head_tb * int(row)
    t_f64 = float(t)

    # 1. Decode the 3 camera frames at anchor timestamp t
    images: list[Image.Image] = []
    for cam_role in (head_camera, "left_hand", "right_hand"):
        cam_key = schema[cam_role]
        cam_tb = time_bases[cam_key]
        raw_img = _read_single_frame_at_time(ds, cam_key, t, cam_tb)
        arr = np.asarray(raw_img, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image from {cam_key}, got {arr.shape}")
        images.append(Image.fromarray(arr, mode="RGB"))

    # 2. Fast single-row read for pre-packed MPDS episodes
    if layout == "packed":
        st_key = schema.get("state") or schema["arm_state"]
        st_tb = time_bases[st_key]
        if "state" in schema:
            st_val = _read_single_frame_at_time(ds, schema["state"], t, st_tb)
            arm_val = hand_val = None
        else:
            st_val = None
            arm_val = _read_single_frame_at_time(ds, schema["arm_state"], t, time_bases[schema["arm_state"]])
            hand_val = _read_single_frame_at_time(ds, schema["hand_state"], t, time_bases[schema["hand_state"]])
        sv_val = (
            _read_single_frame_at_time(ds, schema["state_valid"], t, time_bases[schema["state_valid"]])
            if "state_valid" in schema else None
        )
        row_states, row_sv, is_packed_s = _parse_state_sample(arm_val, hand_val, st_val, sv_val)
        assert is_packed_s and row_sv is not None

        if "actions" in schema:
            act_val = _read_single_frame_at_time(ds, schema["actions"], t, time_bases[schema["actions"]])
            action, is_packed_a = _parse_action_sample(None, None, act_val)
        else:
            aa_val = _read_single_frame_at_time(ds, schema["arm_actions"], t, time_bases[schema["arm_actions"]])
            ha_val = _read_single_frame_at_time(ds, schema["hand_actions"], t, time_bases[schema["hand_actions"]])
            action, is_packed_a = _parse_action_sample(aa_val, ha_val, None)
        assert is_packed_a

        if "action_valid" in schema:
            av_val = _read_single_frame_at_time(ds, schema["action_valid"], t, time_bases[schema["action_valid"]])
            if not bool(np.asarray(av_val).ravel()[0]):
                raise ValueError(f"Manifest selected invalid action row {row} in {path}")

        prompt = None
        if "prompt" in schema:
            p_val = _read_single_frame_at_time(ds, schema["prompt"], t, time_bases[schema["prompt"]])
            prompt = _extract_prompt(p_val)

        raw_state = row_states[-1]
        state_ok = bool(row_sv[-1])
        hist_raw = hist_valid = None
        if history_frames:
            slots = PARQUET_HISTORY_SLOTS[row_states.shape[0]]
            hist_raw = row_states[slots]
            hist_valid = row_sv[slots]
        return images, raw_state, state_ok, action, hist_raw, hist_valid, prompt

    # 3. Cached numeric streams path for single-frame / async multi-rate MPDS episodes
    num = load_mpds_numeric_streams(path, head_camera=head_camera, feature_map=feature_map)
    if num["state_packed"] and num["action_packed"]:
        # Redirect to packed fast path when layout wasn't pre-specified in manifest
        return read_mpds_sample(
            path, row, head_camera=head_camera, history_frames=history_frames,
            feature_map=feature_map, row_timestamps=row_timestamps, layout="packed",
        )

    st_ts = num["state_ts"]
    st_arr = num["state_arr"]
    cur_idx = int(np.clip(np.searchsorted(st_ts, t_f64 + 1e-6, side="right") - 1, 0, len(st_ts) - 1))
    raw_state = st_arr[cur_idx]
    if "state_valid" in num:
        state_ok = bool(num["state_valid"][cur_idx])
    else:
        state_ok = True

    hist_raw = hist_valid = None
    if history_frames:
        assert history_frames == HISTORY_FRAMES, f"Expected history_frames=50, got {history_frames}"
        # Target 10 Hz timestamps: t - 5.0, t - 4.9, ..., t - 0.1
        offsets = np.arange(-HISTORY_FRAMES, 0, dtype=np.float64) * HISTORY_STEP_SEC
        target_ts = t_f64 + offsets
        first_ts = float(st_ts[0])
        h_idx = np.searchsorted(st_ts, target_ts + 1e-6, side="right") - 1
        h_valid = (h_idx >= 0) & (target_ts >= first_ts - 1e-4)
        safe_idx = np.clip(h_idx, 0, len(st_ts) - 1)
        hist_raw = st_arr[safe_idx].copy()
        if "state_valid" in num:
            h_valid = h_valid & num["state_valid"][safe_idx]
        hist_raw[~h_valid] = 0.0
        hist_valid = h_valid

    act_ts = num["action_ts"]
    act_arr = num["action_arr"]
    if num["action_packed"]:
        a_idx = int(np.clip(np.searchsorted(act_ts, t_f64 + 1e-6, side="right") - 1, 0, len(act_ts) - 1))
        action = act_arr[a_idx]
    else:
        pos = int(np.searchsorted(act_ts, t_f64 + 1e-6, side="right"))
        if pos + ACTION_STEPS > len(act_arr):
            raise IndexError(f"Insufficient future actions ({len(act_arr) - pos} < {ACTION_STEPS}) at row {row} in {path}")
        action = act_arr[pos : pos + ACTION_STEPS]

    prompt = None
    if "prompt_vals" in num and num["prompt_vals"]:
        p_idx = int(np.clip(np.searchsorted(num["prompt_ts"], t_f64 + 1e-6, side="right") - 1, 0, len(num["prompt_vals"]) - 1))
        prompt = num["prompt_vals"][p_idx]

    return images, raw_state, state_ok, action, hist_raw, hist_valid, prompt
