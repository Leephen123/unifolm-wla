#!/usr/bin/env python3
"""Attach a standalone SAM mask index to a selected WLA training manifest."""

import argparse
import json
from pathlib import Path
import sys
import tempfile


def existing_file(value):
    path = Path(value).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")
    return path


def episode_key(entry):
    dataset = Path(entry["dataset"]).resolve(strict=True)
    episode = entry["episode_index"]
    if not dataset.is_dir() or type(episode) is not int or episode < 0:
        raise ValueError(f"Invalid dataset or episode index: {dataset}, {episode}")
    return dataset, episode


def merge(training, mask_index):
    import av
    import pyarrow.parquet as pq

    from unifolm_wla.dataloader.recolor_config import mask_fields

    camera = mask_index.get("camera", "front_head")
    columns = {"frame_index", *mask_fields(camera)}
    if training.get("head_camera", camera) != camera:
        raise ValueError(f"Training manifest head camera does not match mask index camera {camera}.")
    masks = {}
    for entry in mask_index["episodes"]:
        key = episode_key(entry)
        if key in masks:
            raise ValueError(f"Duplicate episode in mask index: {key}")
        masks[key] = entry

    seen = set()
    metadata = {}
    if not training["episodes"]:
        raise ValueError("Training manifest has no episodes.")
    for entry in training["episodes"]:
        key = episode_key(entry)
        dataset, episode = key
        if key in seen:
            raise ValueError(f"Duplicate episode in training manifest: {key}")
        seen.add(key)
        if key not in masks:
            raise ValueError(f"No masks for training episode: {key}")
        mask = masks[key]
        if dataset not in metadata:
            metadata[dataset] = json.loads((dataset / "meta/info.json").read_text())
        info = metadata[dataset]
        parquet = existing_file(entry.get("state_valid_parquet") or dataset / info["data_path"].format(
            episode_chunk=episode // info["chunks_size"], episode_index=episode,
        ))
        if camera not in entry["videos"] or camera not in mask:
            raise ValueError(f"Expected {camera} in both the training manifest and mask index: {key}")
        video = existing_file(entry["videos"][camera])
        if parquet != existing_file(mask["source_parquet"]) or video != existing_file(mask[camera]):
            raise ValueError(f"Source Parquet or {camera} video does not match mask index: {key}")
        masks_file = existing_file(mask["background_masks_parquet"])
        frames = pq.ParquetFile(parquet).metadata.num_rows
        mask_parquet = pq.ParquetFile(masks_file)
        if (frames < 1 or mask["frames"] != frames or mask_parquet.metadata.num_rows != frames
                or not columns.issubset(mask_parquet.schema_arrow.names)):
            raise ValueError(f"Mask rows or columns do not match the source episode: {key}")
        with av.open(str(video)) as container:
            if container.streams.video[0].frames != frames:
                raise ValueError(f"{camera} video frame count does not match the source episode: {key}")
        if any(type(row) is not int or not 0 <= row < frames for row in entry["valid_rows"]):
            raise ValueError(f"Invalid training row for episode: {key}")
        entry["background_masks_parquet"] = str(masks_file)
    training["background_mask_camera"] = camera
    return training


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = existing_file(args.manifest)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    training = json.loads(source.read_text())
    index = json.loads(existing_file(args.mask_index).read_text())
    result = merge(training, index)
    if result.get("selection") and output.parent.resolve() != source.parent:
        result["selection"] = str((source.parent / result["selection"]).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / output.name
        staged.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
        output.hardlink_to(staged)
    print(f"Merged masks for {len(result['episodes'])} episodes: {output}")


if __name__ == "__main__":
    main()
