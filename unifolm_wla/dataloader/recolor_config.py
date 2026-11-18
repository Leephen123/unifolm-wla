"""Resolve surface color and combination names for training."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import re


RECOLOR_CONFIG = Path(__file__).resolve().parents[1] / "config" / "recolor"
CAMERA_SURFACES = {
    "front_head": ("table", "floor"),
    "fov_head": ("sofa", "floor"),
}
DEFAULT_TABLE_PALETTES = (
    "white_table_black_floor",
    "red_table_white_floor",
    "black_table_white_floor",
    "light_oak_table_warm_ivory_tile_floor",
    "walnut_table_light_gray_floor",
    "cream_white_table_medium_oak_floor",
    "greige_table_beige_tile_floor",
    "charcoal_table_cool_gray_tile_floor",
    "muted_sage_table_warm_gray_concrete_floor",
    "honey_beech_table_sandstone_floor",
    "pale_birch_table_cool_concrete_floor",
    "smoked_oak_table_ivory_porcelain_floor",
    "espresso_table_taupe_stone_floor",
    "warm_cream_white_table_dark_slate_floor",
    "navy_table_mist_gray_stone_floor",
    "terracotta_table_golden_ivory_tile_floor",
    "olive_green_table_pale_limestone_floor",
    "dusty_blue_table_honey_oak_floor",
)
RGB_KEYS = ("dark_rgb", "light_rgb")


def recolor_surfaces(camera):
    if not isinstance(camera, str) or camera not in CAMERA_SURFACES:
        raise ValueError(f"Unknown head camera {camera!r}. Choose from {', '.join(CAMERA_SURFACES)}.")
    return CAMERA_SURFACES[camera]


def mask_fields(camera):
    return tuple(f"{camera}_{surface}_mask" for surface in recolor_surfaces(camera))


def _named_config(folder, name):
    from omegaconf import OmegaConf

    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError(f"Invalid {folder} config name: {name!r}")
    path = RECOLOR_CONFIG / folder / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown {folder} config {name!r}. Expected {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _require_keys(value, keys, label):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{label} must contain exactly {', '.join(keys)}.")


def _validate_color(color, label):
    _require_keys(color, RGB_KEYS, label)
    result = {}
    for key in RGB_KEYS:
        rgb = color[key]
        if (not isinstance(rgb, Sequence) or len(rgb) != 3
                or any(type(channel) is not int or not 0 <= channel <= 255 for channel in rgb)):
            raise ValueError(f"{label}.{key} must contain three integers from 0 to 255.")
        result[key] = list(rgb)
    return result


def resolve_palettes(camera, names=None, *, saved=None):
    surfaces = recolor_surfaces(camera)
    if names is None:
        if camera != "front_head":
            raise ValueError(f"Set background_recolor_palettes for {camera}.")
        names = DEFAULT_TABLE_PALETTES
    if (not isinstance(names, Sequence) or isinstance(names, str) or not names
            or any(not isinstance(name, str) for name in names) or len(set(names)) != len(names)):
        raise ValueError("background_recolor_palettes must be a nonempty list of distinct combination names.")
    if saved is not None:
        _require_keys(saved, names, "background_recolor_resolved_palettes")
    result = {}
    for name in names:
        label = f"combination {name!r}"
        if saved is None:
            references = _named_config("combinations", name)
            _require_keys(references, surfaces, label)
            palette = {surface: _named_config(surface, references[surface]) for surface in surfaces}
        else:
            palette = saved[name]
            _require_keys(palette, surfaces, label)
        result[name] = {surface: _validate_color(palette[surface], f"{label}.{surface}") for surface in surfaces}
    return result


def resolve_training_palettes(data):
    return resolve_palettes(
        data.get("head_camera", "front_head"),
        data.get("background_recolor_palettes"),
        saved=data.get("background_recolor_resolved_palettes"),
    )


def palette_colors(camera, palettes):
    surfaces = recolor_surfaces(camera)
    return tuple(
        tuple((tuple(palette[surface]["dark_rgb"]), tuple(palette[surface]["light_rgb"])) for surface in surfaces)
        for palette in palettes.values()
    )
