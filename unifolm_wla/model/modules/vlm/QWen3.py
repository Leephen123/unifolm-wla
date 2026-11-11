from typing import Optional

import os

import torch
from unifolm_wla.training.trainer_utils import initialize_overwatch
from unifolm_wla.model.modules.projector.robot_state import (
    RobotStateProjector,
    load_robot_state_projector_state_dict,
)
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Prompt-format constants. Full-body post-training uses the MOCAP control mode.
# The two state tokens are emitted as plain text — the pretrained VLM is
# expected to register them and carry the state projection itself.
ROBOT_STATE_TOKEN = "<|robot_state|>"
ROBOT_STATE_IMPLICIT_STATS_TOKEN = "<|robot_state_implicit_stats|>"
CONTROL_MODE_ARMS_ONLY = "Control Mode: Arms: EE"
CONTROL_MODE_WITH_LOW = "Control Mode: Arms: MOCAP, LOW BODY: MOCAP"


import torch.nn as nn


def _is_weightless_base_vlm(model_id: str) -> bool:
    """True if `model_id` is a local dir with config/tokenizer but no weight files.

    Released VLA checkpoints ship their own full VLM weights in `model.safetensors`
    (loaded on top afterwards), so `base_vlm` only needs to provide the architecture
    config + tokenizer (including any custom tokens like `<|robot_state|>`) — actual
    base weights would be downloaded/loaded and then immediately overwritten.
    """
    if not os.path.isdir(model_id):
        return False
    weight_files = (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    )
    return not any(os.path.exists(os.path.join(model_id, f)) for f in weight_files)


class _QWen3_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        Following https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "Qwen/Qwen3-VL-4B-Instruct")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        if _is_weightless_base_vlm(model_id):
            # `base_vlm` only ships config/tokenizer (no weight files) — this happens
            # for released VLA checkpoints, where the full VLM weights already live in
            # the VLA `model.safetensors` and would just be overwritten on load. Build
            # the architecture from config (random init) instead of downloading/loading
            # base weights that are immediately discarded.
            hf_config = AutoConfig.from_pretrained(model_id)
            model = Qwen3VLForConditionalGeneration._from_config(
                hf_config, attn_implementation=attn_implementation, dtype=torch.bfloat16,
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=attn_implementation,
                dtype=torch.bfloat16,
                ignore_mismatched_sizes=True, # resize image no longer needed? @TODO check bug
            )
        processor = AutoProcessor.from_pretrained(model_id)
        # Right padding: training packs the full chat (system + user + assistant
        # + im_end), so pads sit at the tail and never intrude into the
        # assistant span. Inference (`predict_action`) uses this module as an
        # encoder for the DiT head, not for autoregressive generation, so the
        # generation-side preference for left padding does not apply.
        processor.tokenizer.padding_side = "right"

        self.model = model
        self.processor = processor
        self.config = config

        # alin qwen3 with qwen2.5
        self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # Cache the token length of the assistant role header
        # (`<|im_start|>assistant\n`) so labels can skip it — standard SFT
        # convention (TRL / Qwen / LLaMA-Factory): the deterministic chat
        # template prefix carries no learnable signal and is masked from LM-CE.
        # DiT-side still hides the full span including the header.
        self._assistant_header_len = len(
            self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        )
        self._im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def attach_robot_state_projector(
        self,
        input_dim: int,
        projector_hidden_size: Optional[int] = None,
        load_from: Optional[str] = None,
    ) -> None:
        """Attach a `RobotStateProjector` that injects robot state into the
        `<|robot_state|>` token's input embedding (mirrors ms-swift pretraining).

        Weights are loaded from `load_from` (the pretrained VLM checkpoint dir)
        under the `robot_state_projector.*` keys and co-trained afterwards.
        """
        embeddings = self.model.get_input_embeddings()
        hidden = self.model.config.text_config.hidden_size
        projector = RobotStateProjector(input_dim, hidden, projector_hidden_size)
        projector.to(device=embeddings.weight.device, dtype=embeddings.weight.dtype)
        self.model.robot_state_projector = projector

        token_id = self.processor.tokenizer.convert_tokens_to_ids(ROBOT_STATE_TOKEN)
        unk_id = self.processor.tokenizer.unk_token_id
        if token_id is None or token_id == unk_id:
            logger.warning(
                f"[QWen3] {ROBOT_STATE_TOKEN!r} is NOT a registered token in this checkpoint "
                f"(id={token_id}); robot-state injection will target the wrong positions. "
                f"Point base_vlm at the pretrained VLM that added this token."
            )
        self.model.config.robot_state_token_id = token_id

        state_dict = load_robot_state_projector_state_dict(load_from) if load_from else {}
        if state_dict:
            projector.load_state_dict(state_dict, strict=False)
            logger.info(
                f"[QWen3] loaded robot_state_projector weights ({len(state_dict)} tensors) "
                f"from {load_from}"
            )
        else:
            logger.warning(
                f"[QWen3] no robot_state_projector.* weights found under {load_from!r}; "
                f"projector is randomly initialized (training from scratch)."
            )

    def forward(
        self,
        robot_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen3-VL backbone.

        When `robot_states` (B, input_dim) is provided and a robot_state_projector
        is attached, the projected state is scattered onto the `<|robot_state|>`
        token embedding via a temporary forward hook on the input-embedding layer.

        Crucially `input_ids` is kept intact (NOT replaced by `inputs_embeds`).
        Qwen3VL's `get_placeholder_mask` then locates image/video tokens by TOKEN
        ID (`input_ids == image_token_id`). Dropping `input_ids` would force its
        embedding-comparison branch (`inputs_embeds == embed(image_token_id)`),
        which over-counts whenever another special token's embedding collides with
        the image-pad embedding — raising "Image features and image tokens do not
        match". Keeping `input_ids` also lets M-RoPE be computed normally.
        """
        if robot_states is not None and getattr(self.model, "robot_state_projector", None) is not None:
            handle = self.model.get_input_embeddings().register_forward_hook(
                self._make_robot_state_hook(robot_states)
            )
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return self.model(**kwargs)
            finally:
                handle.remove()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def _make_robot_state_hook(self, robot_states: torch.Tensor):
        """Forward hook that scatters the projected robot state onto the
        `<|robot_state|>` positions of the input-embedding output. Hooking the
        embedding layer (instead of precomputing `inputs_embeds` and nulling
        `input_ids`) keeps token-id placeholder detection and M-RoPE intact while
        still training the projector end-to-end (masked_scatter is differentiable).
        """
        projector = self.model.robot_state_projector
        token_id = self.model.config.robot_state_token_id

        def hook(_module, args, output):
            input_ids = args[0] if args else None
            if input_ids is None:
                return output
            state_mask = input_ids == token_id
            num_state_tokens = int(state_mask.sum().item())
            if num_state_tokens != robot_states.shape[0]:
                raise ValueError(
                    f"robot state token count ({num_state_tokens}) does not match "
                    f"robot_states batch ({robot_states.shape[0]}); expected matching flattened state tokens: "
                    f"{ROBOT_STATE_TOKEN!r} per sample."
                )
            proj_param = next(projector.parameters())
            state_embeds = projector(robot_states.to(device=proj_param.device, dtype=proj_param.dtype))
            state_embeds = state_embeds.to(device=output.device, dtype=output.dtype)
            scatter_mask = state_mask.unsqueeze(-1).expand_as(output)
            return output.masked_scatter(scatter_mask, state_embeds)

        return hook

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None,
                            image_roles=None, arm_types=None,
                            add_generation_prompt=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow Oficial Qwen3-VL Instruct format: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

        When ``image_roles`` is provided, the user turn reproduces the pretraining
        prompt format emitted by ``build_vlm_dataset.py`` — per-camera role labels
        interleaved with their images, then ``Task:`` / ``State:`` / ``Control Mode:``
        lines, e.g.::

            head_left: <image>
            cam_wrist_left: <image>
            Task: <instruction>
            State: <|robot_state_implicit_stats|><|robot_state|>
            Control Mode: Arms: MOCAP, LOW BODY: MOCAP

        ``Control Mode`` includes the LOW BODY line only when the per-sample
        ``arm_type`` is ``dual_with_legs``. When ``image_roles`` is None the legacy
        plain prompt (all images followed by the bare instruction) is used.

        Args:
            image_roles: optional list (per sample) of role-name lists aligned with
                ``images``. Order MUST match the pixel order fed to the processor.
            arm_types: optional list (per sample) of arm_type strings.
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for idx, (imgs, instruction) in enumerate(zip(images, instructions)):
            roles = image_roles[idx] if image_roles is not None else None

            if roles is not None:
                # Pretraining-format prompt: role label immediately precedes its
                # image, then the trailing Task / State / Control Mode block.
                content = []
                for j, (role, img) in enumerate(zip(roles, imgs)):
                    prefix = ("" if j == 0 else "\n") + f"{role}: "
                    content.append({"type": "text", "text": prefix})
                    content.append({"type": "image", "image": img})

                arm_type = arm_types[idx] if arm_types is not None else None
                control_mode = (
                    CONTROL_MODE_WITH_LOW if arm_type == "dual_with_legs" else CONTROL_MODE_ARMS_ONLY
                )
                state_block = ROBOT_STATE_IMPLICIT_STATS_TOKEN + ROBOT_STATE_TOKEN * (1 + int(self.config.datasets.vla_data.get("history_frames", 0)))
                lead = "\n" if content else ""
                trailing = (
                    f"{lead}Task: {instruction}\n"
                    f"State: {state_block}\n"
                    f"{control_mode}"
                )
                content.append({"type": "text", "text": trailing})
            else:
                content = [{"type": "image", "image": img} for img in imgs]
                if "CoT_prompt" in self.config.datasets.vla_data:  # grounding prompt
                    CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                    prompt = CoT_prompt.replace("{instruction}", instruction)
                else:
                    prompt = instruction
                content.append({"type": "text", "text": prompt})

            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # When training (solutions provided) the assistant turn is the supervision
        # target; we must NOT append a generation prompt after it. When inferring
        # (no solutions) the default is to keep the trailing generation prompt so
        # an autoregressive decoder knows to start emitting.
        #
        # QwenMMDiT continuous-action training uses solutions=None. Its inference
        # path also keeps this fixed assistant header so both prefixes match.
        if add_generation_prompt is None:
            add_generation_prompt = solutions is None

        processor_kwargs = {"padding": True, "return_tensors": "pt"}
        if self.config.datasets.vla_data.get('text_max_length', 0):
            processor_kwargs.update(padding='max_length', max_length=int(self.config.datasets.vla_data.text_max_length), truncation=False)
        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=True,
            processor_kwargs=processor_kwargs,
        )

        # if solutions, build (a) the LM-CE labels and (b) the DiT keys-mask.
        # Both derive from the same assistant span: the entire last assistant
        # turn, located via the last `<|im_start|>` token (chat template emits
        # one `<|im_start|>{role}\n` per message and the assistant is always
        # last in a training sample). Padding (right-padded) is excluded via
        # attention_mask. The two diverge by exactly the role header:
        #   - DiT hides `<|im_start|>assistant\n ... <|im_end|>` — full span.
        #   - LM-CE supervises only `<content> ... <|im_end|>` — header skipped.
        if solutions is not None:
            input_ids = batch_inputs["input_ids"]                            # (B, L)
            attention_mask = batch_inputs.get("attention_mask")
            B, L = input_ids.shape

            # Last `<|im_start|>` per row, vectorised. Rows without any match
            # (shouldn't happen with the chat template) get start = L so both
            # masks become all-False.
            is_start = input_ids == self._im_start_id                        # (B, L) bool
            has_start = is_start.any(dim=1)                                  # (B,)
            last_start = (L - 1) - is_start.flip(dims=[1]).int().argmax(dim=1)  # (B,)
            last_start = torch.where(has_start, last_start, torch.full_like(last_start, L))
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0)   # (1, L)
            assistant_mask = positions >= last_start.unsqueeze(1)            # (B, L)
            label_mask = positions >= (last_start + self._assistant_header_len).unsqueeze(1)
            if attention_mask is not None:
                attn_bool = attention_mask.to(dtype=torch.bool)
                assistant_mask = assistant_mask & attn_bool
                label_mask = label_mask & attn_bool

            labels = input_ids.clone()
            labels[~label_mask] = IGNORE_INDEX
            batch_inputs["labels"] = labels
            batch_inputs["assistant_mask"] = assistant_mask

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/unifolm_wla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    qwen_vl = _QWen3_VL_Interface(cfg)
    pass
