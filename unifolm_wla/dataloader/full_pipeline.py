"""Our current-state47 / incoming-root-delta action114 data contract."""

import json
from collections import OrderedDict
from functools import lru_cache
from fractions import Fraction
from math import gcd
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .recolor_config import palette_colors, recolor_surfaces, resolve_palettes, resolve_training_palettes

CAMERAS = (("front_head", "head_left"), ("left_hand", "cam_wrist_left"),
           ("right_hand", "cam_wrist_right"))
# Source G1_29_BODY_NAMES: wr_rel_1010/data_res/dds.py. Keep every state47 channel.
STATE_SLOTS = np.array(list(range(48, 54)) + list(range(54, 60)) +
                       list(range(32, 35)) + list(range(0, 7)) +
                       list(range(16, 23)) + list(range(41, 47)) +
                       list(range(10, 16)) + list(range(26, 32)))


def cameras_for_head(head_camera):
    recolor_surfaces(head_camera)
    return ((head_camera, CAMERAS[0][1]), *CAMERAS[1:])


def pad_resize(image: Image.Image, size=(448, 320)) -> Image.Image:
    """Center black padding to exact target aspect ratio, then resize (W,H)."""
    w, h = image.size
    tw, th = size
    divisor = gcd(tw, th)
    rw, rh = tw // divisor, th // divisor
    k = max((w + rw - 1) // rw, (h + rh - 1) // rh)
    dx, dy = k * rw - w, k * rh - h
    image = ImageOps.expand(image, (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2), fill=0)
    return image.resize(size, Image.Resampling.BILINEAR)


def pack_state(state47):
    state47 = np.asarray(state47, dtype=np.float32)
    assert state47.shape == (47,) and np.isfinite(state47).all()
    state = np.zeros(60, dtype=np.float32)
    mask = np.zeros(60, dtype=bool)
    state[STATE_SLOTS] = state47
    mask[STATE_SLOTS] = True
    return state, mask


@lru_cache(maxsize=8)
def load_body(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("arm_state", "hand_state", "arm_actions", "hand_actions", "centers")}


# In both layouts the last slot is current. Select -5.0 s through -0.1 s.
PARQUET_HISTORY_SLOTS = {
    100: slice(-51, -1),
    200: slice(-101, -1, 2),
}


_parquet_files = OrderedDict()


def _parquet_file(path):
    import pyarrow.parquet as pq

    cached = _parquet_files.get(path)
    if cached is not None:
        _parquet_files.move_to_end(path)
        return cached
    if len(_parquet_files) == 32:
        _parquet_files.popitem(last=False)
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    rg_ends = np.cumsum([meta.row_group(i).num_rows for i in range(meta.num_row_groups)])
    entry = (pf, rg_ends)
    _parquet_files[path] = entry
    return entry


def read_state_valid_row(path, row):
    """Read and validate a single row from a state-valid Parquet file via its row group."""
    import pyarrow as pa

    pf, rg_ends = _parquet_file(path)
    if len(rg_ends) == 0 or row < 0 or row >= int(rg_ends[-1]):
        raise ValueError(f"Row {row} out of range for state-valid episode: {path}")
    rg_idx = int(np.searchsorted(rg_ends, row, side="right"))
    rg_start = 0 if rg_idx == 0 else int(rg_ends[rg_idx - 1])
    local = row - rg_start
    columns = ["arm_state", "hand_state", "arm_actions", "hand_actions", "state_valid", "action_valid"]
    table = pf.read_row_group(rg_idx, columns=columns)
    row_table = table.slice(local, 1)

    av_col = row_table["action_valid"].combine_chunks()
    if pa.types.is_boolean(av_col.type):
        if av_col.null_count:
            raise ValueError(f"Null in action_valid: {path}")
        action_valid = bool(av_col[0].as_py())
    else:
        if av_col.null_count or av_col.values.null_count or len(av_col[0].values) != 1:
            raise ValueError(f"Invalid action_valid in {path}")
        action_valid = bool(av_col[0].values[0].as_py())

    sv_scalar = row_table["state_valid"].combine_chunks()[0]
    if not sv_scalar.is_valid or not hasattr(sv_scalar, "values"):
        raise ValueError(f"Expected a per-state validity list in {path}")
    sv_vals = sv_scalar.values
    if sv_vals.null_count or not pa.types.is_boolean(sv_vals.type):
        raise ValueError(f"Expected non-null boolean state_valid in {path}")
    state_valid = sv_vals.to_numpy(zero_copy_only=False).astype(bool, copy=False)
    state_steps = len(state_valid)
    if state_steps not in PARQUET_HISTORY_SLOTS:
        raise ValueError(f"Expected 100 or 200 state slots, got {state_steps}: {path}")

    def extract(key, width):
        col = row_table[key].combine_chunks()
        scalar = col[0]
        if not scalar.is_valid or not hasattr(scalar, "values"):
            raise ValueError(f"Expected a list column {key} in {path}")
        vals = scalar.values
        if len(vals) != width or vals.null_count:
            raise ValueError(f"Invalid width or null in {key}: {path}")
        return vals.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)

    states = np.concatenate((
        extract("arm_state", state_steps * 35).reshape(state_steps, 35),
        extract("hand_state", state_steps * 12).reshape(state_steps, 12),
    ), axis=-1)
    actions = np.concatenate((
        extract("arm_actions", 5100).reshape(50, 102),
        extract("hand_actions", 600).reshape(50, 12),
    ), axis=-1)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Nonfinite state or action in {path}")
    return states, actions, state_valid, action_valid


@lru_cache(maxsize=2)
def load_state_valid_episode(path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = ("arm_state", "hand_state", "arm_actions", "hand_actions", "state_valid", "action_valid")
    table = pq.read_table(path, columns=list(columns))
    if not len(table):
        raise ValueError(f"Empty state-valid episode: {path}")
    first_mask = table["state_valid"][0]
    if not first_mask.is_valid or not hasattr(first_mask, "values"):
        raise ValueError(f"Expected a per-state validity list in {path}")
    state_steps = len(first_mask.values)
    if state_steps not in PARQUET_HISTORY_SLOTS:
        raise ValueError(f"Expected 100 or 200 state slots, got {state_steps}: {path}")

    def values(key, width, dtype):
        column = table[key].combine_chunks()
        if key == "action_valid" and pa.types.is_boolean(column.type):
            if column.null_count:
                raise ValueError(f"Null in {key}: {path}")
            return column.to_numpy(zero_copy_only=False).reshape(len(table), 1)
        if pa.types.is_fixed_size_list(column.type):
            valid_width = column.type.list_size == width
        elif pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
            valid_width = np.all(np.diff(column.offsets.to_numpy()) == width)
        else:
            raise ValueError(f"Expected a list column {key} in {path}")
        if not valid_width or column.null_count or column.values.null_count:
            raise ValueError(f"Invalid width or null in {key}: {path}")
        if dtype is bool and not pa.types.is_boolean(column.type.value_type):
            raise ValueError(f"Expected boolean {key} in {path}")
        return column.flatten().to_numpy(zero_copy_only=False).astype(dtype, copy=False).reshape(len(table), width)

    states = np.concatenate((
        values("arm_state", state_steps * 35, np.float64).reshape(-1, state_steps, 35),
        values("hand_state", state_steps * 12, np.float64).reshape(-1, state_steps, 12),
    ), axis=-1)
    actions = np.concatenate((
        values("arm_actions", 5100, np.float64).reshape(-1, 50, 102),
        values("hand_actions", 600, np.float64).reshape(-1, 50, 12),
    ), axis=-1)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"Nonfinite state or action in {path}")
    return states, actions, values("state_valid", state_steps, bool), values("action_valid", 1, bool)[:, 0]


_video_readers = OrderedDict()


def video_reader(path):
    import av
    if path in _video_readers:
        _video_readers.move_to_end(path)
        return _video_readers[path]
    if len(_video_readers) == 24:
        _, oldest = _video_readers.popitem(last=False)
        oldest.close()
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_count = 1
    if stream.average_rate != 20:
        container.close()
        raise AssertionError(path)
    _video_readers[path] = container
    return container


def read_frame(path, index):
    container = video_reader(path)
    stream = container.streams.video[0]
    target = Fraction(int(index), 20)
    container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
    for frame in container.decode(stream):
        timestamp = frame.pts * stream.time_base
        if timestamp >= target:
            assert timestamp == target, (path, index, timestamp)
            return frame.to_image()
    raise IndexError((path, index))


DEFAULT_BACKGROUND_RECOLOR_PROBABILITY = 0.5


def _recolor_probability(value):
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("background_recolor_probability must be between 0 and 1.")
    return probability


def resolve_training_data(data):
    probability = _recolor_probability(
        data.get("background_recolor_probability", DEFAULT_BACKGROUND_RECOLOR_PROBABILITY)
    )
    if data.get("table_appearance") is not None:
        if data.get("background_recolor", False) and probability > 0:
            raise ValueError("Choose either background_recolor or table_appearance.")
        return Path(data["background_recolor_manifest_path"]), 0.0
    if data.get("background_recolor", False) and probability > 0:
        return Path(data.get("background_recolor_manifest_path") or data.manifest_path), probability
    return Path(data.manifest_path), 0.0


class FullPipelineDataset(Dataset):
    def __init__(self, manifest_path, statistics_path, history_frames=0, *,
                 background_recolor=False, prompt_mode=None,
                 background_recolor_probability=DEFAULT_BACKGROUND_RECOLOR_PROBABILITY,
                 head_camera="front_head", recolor_palettes=None, table_appearance=None,
                 mpds_feature_map=None):
        manifest = json.loads(Path(manifest_path).read_text())
        self.episodes = manifest["episodes"]
        self.head_camera = head_camera
        self.cameras = cameras_for_head(head_camera)
        self.mpds_feature_map = dict(mpds_feature_map) if mpds_feature_map is not None else None
        if manifest.get("head_camera", head_camera) != head_camera:
            raise ValueError(f"Training manifest does not match head camera {head_camera}.")
        for entry in self.episodes:
            if "mpds_path" in entry:
                continue
            if head_camera not in entry["videos"]:
                raise ValueError(f"Training manifest lacks {head_camera}: {entry['dataset']} episode "
                                 f"{entry['episode_index']}.")
        probability = _recolor_probability(background_recolor_probability)
        self.background_recolor_probability = probability if background_recolor else 0.0
        self.prompt_mode = prompt_mode
        self.table_appearance = table_appearance
        if table_appearance is not None and self.background_recolor_probability:
            raise ValueError("Choose either background recoloring or table appearance.")
        if self.background_recolor_probability > 0 or table_appearance is not None:
            mask_camera = manifest.get("background_mask_camera", "front_head")
            if mask_camera != head_camera:
                raise ValueError(f"Mask camera {mask_camera} does not match training head camera {head_camera}.")
            if any("background_masks_parquet" not in e for e in self.episodes):
                raise ValueError("background_recolor requires background_masks_parquet for every episode.")
        if self.background_recolor_probability > 0:
            palettes = recolor_palettes if recolor_palettes is not None else resolve_palettes(head_camera)
            self.recolor_palettes = palette_colors(head_camera, palettes)
        self.history_frames = int(history_frames)
        assert self.history_frames in (0, 50)
        self.prompt = manifest["prompt"]
        self.ends = np.cumsum([len(e["valid_rows"]) for e in self.episodes])
        self.statistics = json.loads(Path(statistics_path).read_text())["full_pipeline"]
        self.so = np.asarray(self.statistics["state"]["offset"], dtype=np.float32)
        self.ss = np.asarray(self.statistics["state"]["scale"], dtype=np.float32)
        self.ao = np.asarray(self.statistics["action"]["offset"], dtype=np.float32)
        self.asc = np.asarray(self.statistics["action"]["scale"], dtype=np.float32)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        episode_id = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if episode_id == 0 else int(self.ends[episode_id - 1])
        entry = self.episodes[episode_id]
        row = entry["valid_rows"][index - start]
        mpds_path = entry.get("mpds_path")
        mpds_hist_raw = None
        mpds_hist_valid = None
        mpds_prompt = None
        if mpds_path:
            from .mpds_backend import read_mpds_sample
            feature_map = entry.get("mpds_feature_map") or self.mpds_feature_map
            (
                images,
                raw_state,
                state_ok,
                action,
                mpds_hist_raw,
                mpds_hist_valid,
                mpds_prompt,
            ) = read_mpds_sample(
                mpds_path,
                row,
                head_camera=self.head_camera,
                history_frames=self.history_frames,
                feature_map=feature_map,
                row_timestamps=entry.get("row_timestamps"),
                layout=entry.get("mpds_layout"),
            )
            parquet = None
        else:
            parquet = entry.get("state_valid_parquet")
            if parquet:
                row_states, action, row_state_valid, row_action_valid = read_state_valid_row(parquet, row)
                if not row_action_valid:
                    raise ValueError(f"Manifest selected invalid action row {row} in {parquet}")
                raw_state = row_states[-1]
                state_ok = bool(row_state_valid[-1])
            else:
                z = load_body(entry["body"])
                center = int(z["centers"][row])
                raw_state = np.concatenate((z["arm_state"][center], z["hand_state"][center]))
                indices = center + np.arange(1, 51)
                action = np.concatenate((z["arm_actions"][indices], z["hand_actions"][indices]), axis=-1)
                state_ok = True
            images = [read_frame(entry["videos"][key], row) for key, _ in self.cameras]
        if not state_ok:
            state, state_mask = np.zeros(60, np.float32), np.zeros(60, bool)
        else:
            state, state_mask = pack_state((raw_state - self.so) / self.ss)
        action = ((action - self.ao) / self.asc).astype(np.float32)
        assert action.shape == (50, 114) and np.isfinite(action).all()
        probability = self.background_recolor_probability
        if self.table_appearance is not None:
            images[0], appearance = self.table_appearance(images[0], entry, row)
        elif probability == 1.0 or (probability > 0 and np.random.random() < probability):
            from .background_recolor import recolor_head
            images[0] = recolor_head(images[0], entry["background_masks_parquet"], row,
                                     self.head_camera, self.recolor_palettes)
        images = [pad_resize(image) for image in images]
        default_prompt = mpds_prompt or self.prompt
        example = {"image": images, "image_roles": [role for _, role in self.cameras],
                "lang": default_prompt, "state": state, "state_mask": state_mask,
                "action": action, "action_mask": np.ones(114, dtype=bool),
                "arm_type": "dual_with_legs", "robot_type": "unitree"}
        if self.table_appearance is not None:
            example["table_appearance"] = appearance
        if self.history_frames:
            from .state_history import history_rows, pack_history
            if mpds_path:
                raw = mpds_hist_raw
                valid = mpds_hist_valid
            elif parquet:
                history_slots = PARQUET_HISTORY_SLOTS[row_states.shape[0]]
                raw = row_states[history_slots]
                valid = row_state_valid[history_slots]
            else:
                rows, valid = history_rows(row)
                history_centers = z['centers'][rows]
                raw = np.concatenate((z['arm_state'][history_centers], z['hand_state'][history_centers]), axis=-1)
            example['state_history'], example['state_history_mask'] = pack_history(raw, valid, self.so, self.ss)
            draw = 0.0 if self.prompt_mode == 'full_only' else np.random.random()
            if draw < .5:
                source, prompt = 'full', default_prompt
            elif draw < .75:
                source, prompt = 'scene', entry['scene_prompt']
            else:
                from bisect import bisect_right
                captions = entry['captions']
                k = bisect_right([c['start'] for c in captions], row) - 1
                if k >= 0 and row < captions[k]['end']:
                    source, prompt = 'caption', captions[k]['text']
                else:
                    source, prompt = 'caption_missing_scene', entry['scene_prompt']
            example['lang'], example['prompt_source'] = prompt, source
        return example


def collate_examples(batch):
    return batch


def build_full_pipeline_dataloader(cfg, collate_fn=None):
    data = cfg.datasets.vla_data
    assert list(data.image_size_wh) == [448, 320]
    manifest, probability = resolve_training_data(data)
    table_appearance = None
    if data.get("table_appearance") is not None:
        from .table_appearance import TableAppearance
        table_appearance = TableAppearance(data, manifest, cfg.seed)
    mpds_feature_map = data.get("mpds_feature_map")
    if mpds_feature_map is not None:
        from omegaconf import OmegaConf
        mpds_feature_map = OmegaConf.to_container(mpds_feature_map, resolve=True)
    dataset = FullPipelineDataset(
        manifest, data.statistics_path, data.get("history_frames", 0),
        background_recolor=probability > 0,
        background_recolor_probability=probability,
        head_camera=data.get("head_camera", "front_head"),
        recolor_palettes=resolve_training_palettes(data) if probability > 0 else None,
        prompt_mode=data.get("prompt_mode"),
        table_appearance=table_appearance,
        mpds_feature_map=mpds_feature_map,
    )
    # Accelerate shards the ordinary dataloader exactly once across ranks.
    loader_kwargs = dict(
        batch_size=data.per_device_batch_size,
        shuffle=True,
        num_workers=data.num_workers,
        collate_fn=collate_examples if collate_fn is None else collate_fn,
        drop_last=True,
        persistent_workers=data.num_workers > 0,
        multiprocessing_context="spawn" if data.num_workers > 0 else None,
    )
    if data.num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
    return DataLoader(dataset, **loader_kwargs)
