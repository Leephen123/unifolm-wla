from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch


_PREPACK_ARRAY_KEYS = (
    "state",
    "state_mask",
    "state_history",
    "state_history_mask",
    "action",
    "action_mask",
)


def _patchify_images_gpu(
    raw_u8: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """Exact GPU equivalent of Qwen2VLImageProcessor for fixed-size uint8 RGB images."""
    x = raw_u8.to(dtype=torch.float32).mul_(1.0 / 255.0)
    x = (x - mean) / std
    n, h, w, c = x.shape
    x = x.permute(0, 3, 1, 2).unsqueeze(1).expand(n, temporal_patch_size, c, h, w)
    gh, gw = h // patch_size, w // patch_size
    x = x.reshape(
        n,
        1,
        temporal_patch_size,
        c,
        gh // merge_size,
        merge_size,
        patch_size,
        gw // merge_size,
        merge_size,
        patch_size,
    )
    x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return x.reshape(n * gh * gw, c * temporal_patch_size * patch_size * patch_size)


class CPUInputsCollator:
    """Fast worker-side collator: caches chat templates, ships uint8 images for GPU patchify,
    pre-stacks numeric tensors, and avoids pickling PIL images over IPC."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.interface = None
        self._template_cache = OrderedDict()
        self._trim_full_only = cfg.datasets.vla_data.get("prompt_mode") == "full_only"

    def _ensure_interface(self):
        if self.interface is None:
            from omegaconf import OmegaConf
            from transformers import AutoProcessor
            from examples.unifolm_wla.pytree_cycle_fix import install_cycle_free_pytree

            install_cycle_free_pytree()
            processor = AutoProcessor.from_pretrained(self.cfg.framework.qwenvl.base_vlm)
            processor.tokenizer.padding_side = "right"
            # Under prompt_mode="full_only" all samples share one identical prompt; skip
            # fixed right-padding to text_max_length so attention runs unmasked FlashAttention.
            cfg_for_text = self.cfg
            if self._trim_full_only and self.cfg.datasets.vla_data.get("text_max_length", 0):
                cfg_for_text = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
                cfg_for_text.datasets.vla_data.text_max_length = 0
            self.interface = SimpleNamespace(
                processor=processor,
                config=cfg_for_text,
                model=SimpleNamespace(device=torch.device("cpu")),
            )

    def _get_text_template(self, example):
        from unifolm_wla.model.modules.vlm.QWen3 import _QWen3_VL_Interface

        key = (tuple(example["image_roles"]), example["lang"], example["arm_type"])
        cached = self._template_cache.get(key)
        if cached is not None:
            self._template_cache.move_to_end(key)
            return cached
        single = _QWen3_VL_Interface.build_qwenvl_inputs(
            self.interface,
            images=[example["image"]],
            instructions=[example["lang"]],
            image_roles=[example["image_roles"]],
            arm_types=[example["arm_type"]],
        )
        entry = {k: v.clone() for k, v in single.items() if k != "pixel_values"}
        entry["_all_valid"] = bool(entry["attention_mask"].all().item())
        if len(self._template_cache) >= 4096:
            self._template_cache.popitem(last=False)
        self._template_cache[key] = entry
        return entry

    def __call__(self, examples):
        self._ensure_interface()
        batch_size = len(examples)
        templates = [self._get_text_template(e) for e in examples]
        first = templates[0]
        if all(t is first for t in templates):
            prepared_inputs = {
                "input_ids": first["input_ids"].expand(batch_size, -1).contiguous(),
                "image_grid_thw": first["image_grid_thw"].repeat(batch_size, 1),
            }
            if "mm_token_type_ids" in first:
                prepared_inputs["mm_token_type_ids"] = (
                    first["mm_token_type_ids"].expand(batch_size, -1).contiguous()
                )
            if not first["_all_valid"]:
                prepared_inputs["attention_mask"] = (
                    first["attention_mask"].expand(batch_size, -1).contiguous()
                )
        else:
            prepared_inputs = {
                "input_ids": torch.cat([t["input_ids"] for t in templates], dim=0),
                "image_grid_thw": torch.cat([t["image_grid_thw"] for t in templates], dim=0),
            }
            if "mm_token_type_ids" in first:
                prepared_inputs["mm_token_type_ids"] = torch.cat(
                    [t["mm_token_type_ids"] for t in templates], dim=0
                )
            if not all(t["_all_valid"] for t in templates):
                prepared_inputs["attention_mask"] = torch.cat(
                    [t["attention_mask"] for t in templates], dim=0
                )

        # Pack uint8 RGB images (8x smaller IPC than float32 pixel_values); patchify on GPU.
        raw_images = np.stack(
            [np.asarray(im, dtype=np.uint8) for e in examples for im in e["image"]],
            axis=0,
        )
        prepared_inputs["raw_images_u8"] = torch.from_numpy(raw_images)

        # Pre-stack numeric arrays into contiguous CPU tensors for pinned non_blocking H2D.
        prepared_tensors = {}
        sample0 = examples[0]
        for key in _PREPACK_ARRAY_KEYS:
            if key in sample0:
                prepared_tensors[key] = torch.from_numpy(
                    np.stack([e[key] for e in examples], axis=0)
                )

        # Strip PIL images and already-stacked arrays from a shallow copy (never mutate caller's list).
        drop_keys = set(prepared_tensors.keys()) | {"image"}
        stripped_examples = [
            {"image": [], **{k: v for k, v in e.items() if k not in drop_keys}}
            for e in examples
        ]
        return {
            "examples": stripped_examples,
            "prepared_inputs": prepared_inputs,
            "prepared_tensors": prepared_tensors,
        }


def use_worker_inputs(model):
    current = {}
    ip = model.qwen_vl_interface.processor.image_processor
    patch_size = int(ip.patch_size)
    temporal_patch_size = int(ip.temporal_patch_size)
    merge_size = int(ip.merge_size)
    mean_cpu = torch.tensor(ip.image_mean, dtype=torch.float32)
    std_cpu = torch.tensor(ip.image_std, dtype=torch.float32)
    device_stats = {}

    def prebuilt(*a, **kw):
        return current["inputs"]

    model.qwen_vl_interface.build_qwenvl_inputs = prebuilt
    original_forward, original_predict = model.forward, model.predict_action

    def select(batch):
        inputs = dict(batch["prepared_inputs"])
        size = len(batch["examples"])
        assert inputs["input_ids"].shape[0] == size
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            assert inputs["attention_mask"].shape == inputs["input_ids"].shape
        grid = inputs["image_grid_thw"]
        assert grid.shape == (size * 3, 3) and bool((grid == grid.new_tensor([1, 20, 28])).all())
        assert "assistant_mask" not in inputs

        raw_u8 = inputs.pop("raw_images_u8", None)
        if raw_u8 is not None:
            dev = raw_u8.device
            stats = device_stats.get(dev)
            if stats is None:
                stats = (mean_cpu.to(dev), std_cpu.to(dev))
                device_stats[dev] = stats
            inputs["pixel_values"] = _patchify_images_gpu(
                raw_u8, stats[0], stats[1], patch_size, temporal_patch_size, merge_size
            )
        current["inputs"] = inputs
        return batch["examples"]

    def forward(batch):
        examples = select(batch)
        model._prepacked_tensors = batch.get("prepared_tensors")
        try:
            return original_forward(examples)
        finally:
            model._prepacked_tensors = None

    def predict_action(batch):
        examples = select(batch)
        model._prepacked_tensors = batch.get("prepared_tensors")
        try:
            return original_predict(examples)
        finally:
            model._prepacked_tensors = None

    model.forward = forward
    model.predict_action = predict_action
