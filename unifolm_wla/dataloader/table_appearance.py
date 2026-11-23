"""Sample tabletop and floor appearance at the original camera resolution."""
import hashlib
import json
import sqlite3
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .background_recolor import LUMINANCE, _feather_weights
from .recolor_config import (
    _validate_color, palette_colors, resolve_floor_textures, resolve_table_textures, resolve_training_palettes,
)

TABLE_APPEARANCES = ("original", "color", "texture", "empty_table_mask", "floor_texture", "table_floor_texture", "table_color_floor_texture")
CACHE_VERSION = 1


def table_cache_path(directory, dataset):
    name = hashlib.sha256(str(dataset).encode()).hexdigest()[:24]
    return Path(directory) / f"{name}.sqlite3"


@lru_cache(maxsize=16)
def _database(path):
    return sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)


@lru_cache(maxsize=16)
def _texture(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


@lru_cache(maxsize=16)
def _pixel_grid(width, height):
    y, x = np.indices((height, width), dtype=np.float32)
    return x, y


def _texture_coordinates(matrix, size, placement):
    width, height = size
    x, y = _pixel_grid(width, height)
    denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    near_zero = np.abs(denominator) < 1e-5
    denominator[near_zero] = np.copysign(1e-5, denominator[near_zero])
    u = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator
    v = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator
    angle, scale, offset_u, offset_v = placement
    sine, cosine = np.sin(angle), np.cos(angle)
    return ((cosine * u - sine * v) * scale + offset_u,
            (sine * u + cosine * v) * scale + offset_v)


def _sample_texture(texture, u, v, *, offset=0):
    import cv2

    map_x = ((u % 1) * texture.shape[1] + offset).astype(np.float32)
    map_y = ((v % 1) * texture.shape[0] + offset).astype(np.float32)
    return cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _repeating_texture(texture, matrix, size, placement):
    import cv2

    # Each rank already uses persistent worker processes.
    cv2.setNumThreads(1)
    return _sample_texture(texture, *_texture_coordinates(matrix, size, placement))


@lru_cache(maxsize=16)
def _texture_pyramid(path):
    import cv2

    levels = [_texture(path)]
    while max(levels[-1].shape[:2]) > 1:
        height, width = levels[-1].shape[:2]
        size = max(1, width // 2), max(1, height // 2)
        levels.append(cv2.resize(levels[-1], size, interpolation=cv2.INTER_AREA))
    return tuple(levels)


def _filtered_floor_texture(path, matrix, size, placement, mask):
    import cv2

    cv2.setNumThreads(1)
    levels = _texture_pyramid(path)
    height, width = levels[0].shape[:2]
    u, v = _texture_coordinates(matrix, size, placement)

    # Measure before wrapping so repeat seams do not inflate the footprint.
    y, x = np.nonzero(mask)
    y, x = y[::16], x[::16]
    h, w = mask.shape
    dx = np.where(x == w - 1, x - 1, x + 1)
    dy = np.where(y == h - 1, y - 1, y + 1)
    along_x = np.hypot((u[y, dx] - u[y, x]) * width, (v[y, dx] - v[y, x]) * height)
    along_y = np.hypot((u[dy, x] - u[y, x]) * width, (v[dy, x] - v[y, x]) * height)
    # A robust floor-wide footprint avoids abrupt changes from a few edge pixels.
    footprint = float(np.percentile(np.maximum(along_x, along_y), 80))
    lod = float(np.clip(np.log2(max(footprint, 1.0)) + 0.35, 0, len(levels) - 1))
    lower = int(lod)
    primary = _sample_texture(levels[lower], u, v, offset=-0.5)
    fraction = lod - lower
    if fraction == 0:
        return primary
    secondary = _sample_texture(levels[lower + 1], u, v, offset=-0.5)
    return cv2.addWeighted(primary, 1 - fraction, secondary, fraction, 0)


def _surface_cache(directory, manifest_hash, surface):
    path = Path(directory).resolve(strict=True)
    index = json.loads((path / "index.json").read_text())
    # The original tabletop cache predates the surface field.
    actual = (index["version"], index["manifest_sha256"], index.get("surface", "table"))
    if actual != (CACHE_VERSION, manifest_hash, surface):
        raise ValueError(f"The {surface} appearance cache does not match the training manifest or surface.")
    return path


def _surface_frame(cache, entry, row, size, surface):
    path = table_cache_path(cache, entry["dataset"])
    record = _database(path).execute(
        "SELECT mask, projection FROM frames WHERE episode = ? AND row = ?",
        (entry["episode_index"], row),
    ).fetchone()
    if record is None:
        raise ValueError(f"{surface.capitalize()} appearance cache lacks frame {row} for {entry['dataset']}.")
    packed, projection = record
    if packed is None:
        return None
    width, height = size
    decoded = np.frombuffer(zlib.decompress(packed), dtype=np.uint8)
    if len(decoded) != (width * height + 7) // 8:
        raise ValueError(f"{surface.capitalize()} mask dimensions do not match the video for {entry['dataset']}.")
    mask = np.unpackbits(decoded, count=width * height).reshape(height, width).astype(bool)
    return mask, projection


def _composite(output, mask, *, palette=None, texture=None):
    selected = output[mask].astype(np.float32)
    luma = (selected @ LUMINANCE)[:, None] / 255
    if palette is not None:
        dark, light = np.asarray(palette, dtype=np.float32)
        replacement = dark + luma * (light - dark)
    else:
        replacement = texture[mask] * np.clip(0.64 + 0.83 * luma, 0.72, 1.07)
    alpha = _feather_weights(mask)
    output[mask] = np.clip(np.rint(selected + alpha * (replacement - selected)), 0, 255).astype(np.uint8)


class TableAppearance:
    """An exclusive augmentation with legacy or unified table-color + floor color/texture modes."""

    def __init__(self, data, manifest, seed):
        config = data.table_appearance
        self.camera = data.get("head_camera", "front_head")
        if self.camera != "front_head":
            raise ValueError("Table appearance requires front_head.")
        self.floor_texture_prefilter = config.get("floor_texture_prefilter", True)
        if type(self.floor_texture_prefilter) is not bool:
            raise ValueError("floor_texture_prefilter must be a boolean.")
        self.random_placement = config.get("random_placement", False)
        if type(self.random_placement) is not bool:
            raise ValueError("random_placement must be a boolean.")
        self.min_table_area_fraction = float(config.get("min_table_area_fraction", 0.0))
        if not (0.0 <= self.min_table_area_fraction < 1.0):
            raise ValueError("min_table_area_fraction must be in [0, 1).")

        unified = config.get("unified_probability")
        self.unified_probability = None if unified is None else float(unified)
        self.seed = int(seed)
        manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        self.cache = _surface_cache(config.cache_dir, manifest_hash, "table")
        self.floor_cache = None
        self.colors = ()
        self.floor_colors = ()
        self.textures = ()
        self.floor_textures = ()

        if self.unified_probability is not None:
            if not (0 < self.unified_probability <= 1):
                raise ValueError("unified_probability must be in (0, 1].")
            if float(config.get("color_probability", 0)) or float(config.get("texture_probability", 0)):
                raise ValueError("Do not combine unified_probability with color_probability/texture_probability.")
            if config.get("textures"):
                raise ValueError("Unified table-color + floor mode requires empty table_appearance.textures.")
            self.color_probability = 0.0
            self.texture_probability = 0.0
            self.total_probability = self.unified_probability

            palettes = resolve_training_palettes(data)
            paired = palette_colors(self.camera, palettes)
            extra = tuple(
                _validate_color(color, f"table_appearance.extra_colors.{name}")
                for name, color in config.get("extra_colors", {}).items()
            )
            self.colors = tuple(c[0] for c in paired) + tuple((c["dark_rgb"], c["light_rgb"]) for c in extra)
            self.floor_colors = tuple(c[1] for c in paired)
            floor_names = config.get("floor_textures")
            self.floor_textures = resolve_floor_textures(floor_names) if floor_names else ()
            if not config.get("floor_cache_dir"):
                raise ValueError("Unified mode requires table_appearance.floor_cache_dir.")
            self.floor_cache = _surface_cache(config.floor_cache_dir, manifest_hash, "floor")
        else:
            self.color_probability = float(config.color_probability)
            self.texture_probability = float(config.texture_probability)
            self.total_probability = self.color_probability + self.texture_probability
            if not (self.color_probability >= 0 and self.texture_probability >= 0
                    and 0 < self.total_probability <= 1):
                raise ValueError("Table appearance probabilities must be nonnegative and sum to (0, 1].")
            if self.color_probability:
                palettes = resolve_training_palettes(data)
                extra = tuple(
                    _validate_color(color, f"table_appearance.extra_colors.{name}")
                    for name, color in config.get("extra_colors", {}).items()
                )
                base = tuple(colors[0] for colors in palette_colors(self.camera, palettes))
                self.colors = base + tuple((color["dark_rgb"], color["light_rgb"]) for color in extra)
            table_names = config.get("textures") or ()
            if self.texture_probability and table_names:
                self.textures = resolve_table_textures(table_names)
            floor_names = config.get("floor_textures")
            self.floor_textures = resolve_floor_textures(floor_names) if floor_names else ()
            if self.texture_probability and not self.textures and not self.floor_textures:
                raise ValueError("table_appearance requires textures or floor_textures when texture_probability > 0.")
            if self.floor_textures:
                if not self.texture_probability or not config.get("floor_cache_dir"):
                    raise ValueError("Floor textures require texture_probability and table_appearance.floor_cache_dir.")
                self.floor_cache = _surface_cache(config.floor_cache_dir, manifest_hash, "floor")

    def _random_pose(self):
        return (
            float(np.random.uniform(-0.22, 0.22)),
            float(np.random.uniform(1.45, 2.35)),
            float(np.random.random()),
            float(np.random.random()),
        )

    def _placement(self, entry, surface="table"):
        textures = self.floor_textures if surface == "floor" else self.textures
        if self.random_placement:
            return textures[int(np.random.randint(len(textures)))], self._random_pose()
        # The frame order, DataLoader worker, and rank do not change the placement.
        key = f'{self.seed}\0{entry["dataset"]}\0{entry["episode_index"]}'.encode()
        if surface == "floor":
            key = b"floor\0" + key
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        return textures[int(rng.integers(len(textures)))], (rng.uniform(-0.22, 0.22), rng.uniform(1.45, 2.35),
                                      rng.random(), rng.random())

    def _valid_table(self, table):
        if table is None:
            return False
        if self.min_table_area_fraction > 0 and float(table[0].mean()) < self.min_table_area_fraction:
            return False
        return True

    def __call__(self, image, entry, row):
        draw = np.random.random()
        if draw >= self.total_probability:
            return image, "original"

        table = _surface_frame(self.cache, entry, row, image.size, "table")
        if self.unified_probability is not None:
            if not self._valid_table(table):
                return image, "empty_table_mask"
            output = np.array(image.convert("RGB"))
            table_palette = self.colors[int(np.random.randint(len(self.colors)))]
            _composite(output, table[0], palette=table_palette)

            floor = _surface_frame(self.floor_cache, entry, row, image.size, "floor")
            if floor is None:
                return Image.fromarray(output), "color"

            n_floor_colors = len(self.floor_colors)
            n_floor_textures = len(self.floor_textures)
            choice = int(np.random.randint(n_floor_colors + n_floor_textures))
            if choice < n_floor_colors:
                _composite(output, floor[0], palette=self.floor_colors[choice])
                return Image.fromarray(output), "color"

            asset = self.floor_textures[choice - n_floor_colors]
            if self.random_placement:
                placement = self._random_pose()
            else:
                _, placement = self._placement(entry, "floor")
            mask, projection = floor
            matrix = np.frombuffer(projection, dtype="<f4").reshape(3, 3)
            if self.floor_texture_prefilter:
                texture = _filtered_floor_texture(asset, matrix, image.size, placement, mask)
            else:
                texture = _repeating_texture(_texture(asset), matrix, image.size, placement)
            _composite(output, mask, texture=texture)
            return Image.fromarray(output), "table_color_floor_texture"

        if draw < self.color_probability:
            if not self._valid_table(table):
                return image, "empty_table_mask"
            output = np.array(image.convert("RGB"))
            _composite(output, table[0], palette=self.colors[np.random.randint(len(self.colors))])
            return Image.fromarray(output), "color"

        floor = None
        if self.floor_cache is not None:
            floor = _surface_frame(self.floor_cache, entry, row, image.size, "floor")
        if not self._valid_table(table) and (self.min_table_area_fraction > 0 or floor is None):
            return image, "empty_table_mask"
        output = np.array(image.convert("RGB"))
        surfaces = []
        if self.textures:
            surfaces.append(("table", table if self._valid_table(table) else None))
        if self.floor_textures:
            surfaces.append(("floor", floor))
        applied = set()
        for surface, frame in surfaces:
            if frame is None:
                continue
            mask, projection = frame
            asset, placement = self._placement(entry, surface)
            matrix = np.frombuffer(projection, dtype="<f4").reshape(3, 3)
            if surface == "floor" and self.floor_texture_prefilter:
                texture = _filtered_floor_texture(asset, matrix, image.size, placement, mask)
            else:
                texture = _repeating_texture(_texture(asset), matrix, image.size, placement)
            _composite(output, mask, texture=texture)
            applied.add(surface)

        if not applied:
            return image, "empty_table_mask"
        if applied == {"table", "floor"}:
            appearance = "table_floor_texture"
        elif applied == {"floor"}:
            appearance = "floor_texture"
        else:
            appearance = "texture"
        return Image.fromarray(output), appearance
