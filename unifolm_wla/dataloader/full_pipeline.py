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

CAMERAS = (("front_head", "head_left"), ("left_hand", "cam_wrist_left"),
           ("right_hand", "cam_wrist_right"))
# Source G1_29_BODY_NAMES: wr_rel_1010/data_res/dds.py. Keep every state47 channel.
STATE_SLOTS = np.array(list(range(48, 54)) + list(range(54, 60)) +
                       list(range(32, 35)) + list(range(0, 7)) +
                       list(range(16, 23)) + list(range(41, 47)) +
                       list(range(10, 16)) + list(range(26, 32)))


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


_video_readers = OrderedDict()


def video_reader(path):
    import av
    if path in _video_readers:
        _video_readers.move_to_end(path)
        return _video_readers[path]
    if len(_video_readers) == 6:
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


class FullPipelineDataset(Dataset):
    def __init__(self, manifest_path, statistics_path, history_frames=0):
        manifest = json.loads(Path(manifest_path).read_text())
        self.episodes = manifest["episodes"]
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
        z = load_body(entry["body"])
        center = int(z["centers"][row])
        raw_state = np.concatenate((z["arm_state"][center], z["hand_state"][center]))
        state, state_mask = pack_state((raw_state - self.so) / self.ss)
        indices = center + np.arange(1, 51)
        action = np.concatenate((z["arm_actions"][indices], z["hand_actions"][indices]), axis=-1)
        action = ((action - self.ao) / self.asc).astype(np.float32)
        assert action.shape == (50, 114) and np.isfinite(action).all()
        images = [pad_resize(read_frame(entry["videos"][key], row))
                  for key, role in CAMERAS]
        example = {"image": images, "image_roles": [role for key, role in CAMERAS],
                "lang": self.prompt, "state": state, "state_mask": state_mask,
                "action": action, "action_mask": np.ones(114, dtype=bool),
                "arm_type": "dual_with_legs", "robot_type": "unitree"}
        if self.history_frames:
            from .state_history import history_rows, pack_history
            rows, valid = history_rows(row)
            history_centers = z['centers'][rows]
            raw = np.concatenate((z['arm_state'][history_centers], z['hand_state'][history_centers]), axis=-1)
            example['state_history'], example['state_history_mask'] = pack_history(raw, valid, self.so, self.ss)
            draw = np.random.random()
            if draw < .5:
                source, prompt = 'full', self.prompt
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


def build_full_pipeline_dataloader(cfg):
    data = cfg.datasets.vla_data
    assert list(data.image_size_wh) == [448, 320]
    dataset = FullPipelineDataset(data.manifest_path, data.statistics_path, data.get("history_frames", 0))
    # Accelerate shards the ordinary dataloader exactly once across ranks.
    return DataLoader(dataset, batch_size=data.per_device_batch_size, shuffle=True,
                      num_workers=data.num_workers, collate_fn=collate_examples,
                      drop_last=True, persistent_workers=data.num_workers > 0,
                      multiprocessing_context="spawn" if data.num_workers > 0 else None)
