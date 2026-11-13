"""Apply training-only front camera recoloring using lossless Parquet masks."""

from functools import lru_cache
from io import BytesIO

import numpy as np
from PIL import Image


MASK_FIELDS = ("front_head_table_mask", "front_head_floor_mask")
# Each palette contains the dark and light RGB endpoints for desk and floor.
COLOR_PALETTES = (
    ("White / Black", ((150, 150, 150), (255, 255, 255)), ((7, 7, 7), (64, 64, 64))),
    ("Red / White", ((90, 6, 8), (255, 50, 50)), ((150, 150, 150), (255, 255, 255))),
    ("Black / White", ((7, 7, 7), (64, 64, 64)), ((150, 150, 150), (255, 255, 255))),
    ("Light oak / Warm ivory tile", ((102, 72, 42), (232, 200, 151)), ((124, 120, 108), (246, 239, 218))),
    ("Walnut / Light concrete", ((43, 26, 18), (171, 121, 82)), ((107, 110, 110), (220, 223, 221))),
    ("Cream white / Medium oak", ((137, 132, 118), (251, 247, 232)), ((73, 49, 29), (210, 170, 118))),
    ("Greige / Beige tile", ((84, 75, 66), (207, 194, 176)), ((117, 98, 75), (230, 209, 177))),
    ("Charcoal / Cool gray tile", ((25, 28, 30), (108, 114, 120)), ((91, 100, 107), (212, 220, 227))),
    ("Muted sage / Warm gray concrete", ((48, 64, 52), (161, 183, 152)), ((105, 98, 90), (206, 199, 189))),

    ("Honey beech / Sandstone", ((105, 67, 33), (232, 186, 121)), ((112, 100, 78), (217, 205, 178))),
    ("Pale birch / Cool concrete", ((113, 91, 64), (244, 223, 184)), ((103, 110, 115), (216, 224, 226))),
    ("Smoked oak / Ivory porcelain", ((58, 43, 33), (157, 130, 104)), ((143, 138, 126), (246, 241, 225))),
    ("Espresso / Taupe stone", ((27, 18, 15), (105, 78, 64)), ((99, 91, 82), (190, 180, 163))),
    ("Cream white / Dark slate", ((139, 129, 111), (251, 244, 225)), ((42, 47, 51), (140, 150, 158))),
    ("Navy / Mist gray stone", ((26, 41, 59), (108, 134, 160)), ((118, 123, 123), (229, 231, 226))),
    ("Terracotta / Warm ivory tile", ((96, 53, 38), (203, 139, 112)), ((140, 133, 116), (245, 235, 210))),
    ("Olive green / Pale limestone", ((50, 59, 40), (144, 156, 115)), ((136, 130, 112), (241, 231, 207))),
    ("Dusty blue / Honey oak", ((57, 74, 86), (161, 181, 191)), ((98, 62, 34), (220, 179, 119))),
)
LUMINANCE = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)


@lru_cache(maxsize=4)
def _mask_rows(parquet_path):
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path, columns=["frame_index", *MASK_FIELDS])


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


def recolor_front_head(image, parquet_path, row):
    """Change the two masked surfaces at native resolution before pad/resize."""
    table = _mask_rows(parquet_path)
    if table["frame_index"][row].as_py() != row:
        raise ValueError(f"Background mask row {row} does not match frame_index in {parquet_path}")
    desk, floor = [_decode_mask(table[key][row].as_py(), image.size) for key in MASK_FIELDS]
    if np.any(desk & floor):
        raise ValueError(f"Background masks overlap at row {row} in {parquet_path}")
    image = image.convert("RGB")
    output = np.array(image)
    _, desk_colors, floor_colors = COLOR_PALETTES[np.random.randint(len(COLOR_PALETTES))]
    for mask, (dark, light) in zip((desk, floor), (desk_colors, floor_colors), strict=True):
        if not mask.any():
            continue
        dark, light = np.asarray(dark, np.float32), np.asarray(light, np.float32)
        selected = output[mask].astype(np.float32)
        luminance = (selected @ LUMINANCE)[:, None] / 255
        desired = dark + luminance * (light - dark)
        alpha = _feather_weights(mask)
        output[mask] = np.rint(selected * (1 - alpha) + desired * alpha)
    return Image.fromarray(output)
