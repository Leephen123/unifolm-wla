"""Apply training-only front camera recoloring using lossless Parquet masks."""

from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image

from .recolor_config import mask_fields

LUMINANCE = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)


@lru_cache(maxsize=4)
def _mask_rows(parquet_path, fields):
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path, columns=["frame_index", *fields])


def _decode_mask(value, size):
    with Image.open(BytesIO(value["bytes"])) as image:
        if image.mode != "L" or image.size != size:
            raise ValueError(f"Expected a grayscale mask of size {size}, got {image.mode} {image.size}")
        pixels = np.asarray(image)
        if not np.all((pixels == 0) | (pixels == 255)):
            raise ValueError("A background mask contains values other than 0 and 255.")
        return pixels == 255


def _feather_weights(mask):
    """Linear 2 px inward ramp, using exact distances on the pixel grid."""
    padded = np.pad(mask, 1)
    cardinal = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    corners = padded[:-2, :-2] & padded[:-2, 2:] & padded[2:, :-2] & padded[2:, 2:]
    weights = np.where(cardinal, np.where(corners, 1.0, np.sqrt(0.5)), 0.5)
    return weights[mask].astype(np.float32)[:, None]


def recolor_head(image, parquet_path, row, camera, palettes):
    """Change the two masked surfaces at native resolution before pad/resize."""
    fields = mask_fields(camera)
    table = _mask_rows(parquet_path, fields)
    if table["frame_index"][row].as_py() != row:
        raise ValueError(f"Background mask row {row} does not match frame_index in {parquet_path}")
    surface, floor = [_decode_mask(table[key][row].as_py(), image.size) for key in fields]
    if np.any(surface & floor):
        raise ValueError(f"Background masks overlap at row {row} in {parquet_path}")
    image = image.convert("RGB")
    output = np.array(image)
    colors = palettes[0] if len(palettes) == 1 else palettes[np.random.randint(len(palettes))]
    for mask, (dark, light) in zip((surface, floor), colors, strict=True):
        if not mask.any():
            continue
        dark, light = np.asarray(dark, np.float32), np.asarray(light, np.float32)
        selected = output[mask].astype(np.float32)
        luminance = (selected @ LUMINANCE)[:, None] / 255
        desired = dark + luminance * (light - dark)
        alpha = _feather_weights(mask)
        output[mask] = np.rint(selected * (1 - alpha) + desired * alpha)
    return Image.fromarray(output)
