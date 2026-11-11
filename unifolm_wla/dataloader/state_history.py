"""50 past observations on the published 20 Hz camera timeline (-5s..-0.1s)."""
import numpy as np
from .full_pipeline import STATE_SLOTS


def history_rows(row, segment_start=0):
    rows = row - np.arange(100, 0, -2)
    valid = rows >= segment_start
    return np.maximum(rows, segment_start), valid


def pack_history(raw, valid, offset, scale):
    assert raw.shape == (50, 47) and valid.shape == (50,)
    values = np.zeros((50, 60), dtype=np.float32)
    masks = np.zeros((50, 60), dtype=bool)
    values[:, STATE_SLOTS] = ((raw-offset)/scale).astype(np.float32) * valid[:, None]
    masks[:, STATE_SLOTS] = valid[:, None]
    return values, masks
