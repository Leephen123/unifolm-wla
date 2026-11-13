from typing import Optional, List, Dict, Any, Tuple, Union
import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_dispatch import AttentionBackendName
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from torch import nn

_FLASH_ATTN_AVAILABLE = False


def dispatch_attention_fn(q, k, v, dropout_p=0.0, is_causal=False, backend=None, attn_mask=None):
    y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                                      attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
    return y.transpose(1,2)


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        """Embed timesteps.

        Supports both the standard flow-matching schedule (a single scalar per
        batch item, shape ``[B]``) and the RTC schedule where every action step
        carries its own diffusion time (shape ``[B, H]``). For the 2D case the
        ``H`` axis is flattened, embedded, then restored so the output is
        ``[B, H, D]``; the 1D case returns ``[B, D]`` unchanged.
        """
        dtype = next(self.parameters()).dtype
        if timesteps.dim() == 2:
            b, h = timesteps.shape
            flat = timesteps.reshape(b * h)
            timesteps_proj = self.time_proj(flat).to(dtype)
            timesteps_emb = self.timestep_embedder(timesteps_proj)  # (B*H, D)
            return timesteps_emb.reshape(b, h, -1)  # (B, H, D)
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class QwenActionTextRotaryEmbed(nn.Module):
    """Produce 1D RoPE freqs for the joint (text + action) attention stream.

    Positions are assigned on a single continuous axis with **action first**:
    action tokens occupy ``0 .. S_act-1`` and text tokens continue at
    ``S_act .. S_act+S_txt-1``.

    Action-first is deliberate: the action block length ``S_act`` is fixed (the
    action horizon), while the text length ``S_txt`` varies between training
    (context + assistant turn + right-padding) and inference (context only).
    Anchoring action at ``0`` makes the action stream's rotary positions
    identical train/inference, removing the position shift. The assistant turn
    and padding sit at the tail of the text block and are masked out of the
    keys, so the masked positions they consume are never attended to.

    The returned (cos, sin) real tensors each have shape ``[S, head_dim]`` and
    are consumed by :func:`apply_rotary_emb_qwen` with ``use_real=True`` — the
    real-valued formulation is used (instead of the mathematically equivalent
    complex ``view_as_complex``/``view_as_real`` path) because complex tensors
    have no ONNX/TensorRT representation.
    """

    def __init__(self, head_dim: int, theta: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, seq_txt: int, seq_action: int, device: torch.device):
        total = seq_txt + seq_action
        # float dtype (not the arange default int64): the legacy ONNX exporter
        # mis-traces torch.outer's int64->float32 type promotion inside
        # get_1d_rotary_pos_embed, producing an int64 Sin/Cos node.
        pos = torch.arange(total, device=device, dtype=torch.float32)
        freqs_cos, freqs_sin = get_1d_rotary_pos_embed(
            self.head_dim, pos, theta=self.theta, use_real=True
        )  # [total, D] real
        action_freqs = (freqs_cos[:seq_action], freqs_sin[:seq_action])  # action first: positions 0 .. S_act-1
        txt_freqs = (freqs_cos[seq_action:], freqs_sin[seq_action:])     # text continues: S_act .. S_act+S_txt-1
        return action_freqs, txt_freqs


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x



def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        # x here is [B, S, H, D] (seq before heads — see docstring), not the
        # [B, H, S, D] layout this branch was originally written for, so
        # broadcast over dims 0 (B) and 2 (H) instead of 0 and 1.
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            # reshape(*x.shape[:-1], -1) instead of .flatten(3): the dynamo ONNX
            # exporter was lowering .flatten(3) to a Reshape with the batch dim
            # baked in from the trace-time example input instead of kept dynamic.
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).reshape(*x.shape[:-1], -1)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]
        seq_action = hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        action_query = attn.to_q(hidden_states)
        action_key = attn.to_k(hidden_states)
        action_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        action_query = action_query.unflatten(-1, (attn.heads, -1))
        action_key = action_key.unflatten(-1, (attn.heads, -1))
        action_value = action_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            action_query = attn.norm_q(action_query)
        if attn.norm_k is not None:
            action_key = attn.norm_k(action_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            action_freqs, txt_freqs = image_rotary_emb
            action_query = apply_rotary_emb_qwen(action_query, action_freqs, use_real=True)
            action_key = apply_rotary_emb_qwen(action_key, action_freqs, use_real=True)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=True)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=True)

        # Concatenate for joint attention.
        # Order: [action, text] — action-first. The joint key-padding mask
        # (built in DiT.forward) is `[action(always-True), txt(contiguous
        # True-prefix then padding)]`, which is itself a single contiguous
        # True run starting at 0. FLASH_VARLEN's key packer only ever takes
        # `key[b, :valid_len]` (see diffusers `_flash_varlen_attention`), so
        # it silently assumes the True positions are a 0-anchored prefix; a
        # [text, action] order would carve a gap into the middle of that
        # prefix (masked assistant-turn / padding tokens sit between the
        # valid text prefix and the always-valid action tokens) and corrupt
        # which keys are actually attended to. Also matches
        # `QwenActionTextRotaryEmbed`'s existing action-first RoPE layout.
        joint_query = torch.cat([action_query, txt_query], dim=1)
        joint_key = torch.cat([action_key, txt_key], dim=1)
        joint_value = torch.cat([action_value, txt_value], dim=1)

        # Joint attention via diffusers' backend dispatch. Tensors are
        # [B, S, H, D] here. `self._attention_backend` defaults to None, in
        # which case we pick FLASH/FLASH_VARLEN when flash-attn is importable
        # and fall back to NATIVE (plain SDPA) otherwise — diffusers'
        # FLASH/FLASH_VARLEN backends raise RuntimeError outright if
        # flash-attn isn't installed, unlike QWen3.py's VLM path which has
        # its own fallback. Explicitly set `self._attention_backend` to
        # `AttentionBackendName.NATIVE` to export to ONNX/TensorRT, since
        # flash-attn kernels have no ONNX equivalent. flash-attn requires
        # bf16/fp16, so only cast Q/K/V to bf16 for the flash backends —
        # NATIVE runs in whatever dtype the module is already in.
        orig_dtype = joint_query.dtype
        flash_backends = (AttentionBackendName.FLASH, AttentionBackendName.FLASH_VARLEN)
        if _FLASH_ATTN_AVAILABLE:
            default_backend = AttentionBackendName.FLASH if attention_mask is None else AttentionBackendName.FLASH_VARLEN
        else:
            default_backend = AttentionBackendName.NATIVE
        backend = self._attention_backend or default_backend

        if backend in flash_backends:
            q = joint_query.to(torch.bfloat16)
            k = joint_key.to(torch.bfloat16)
            v = joint_value.to(torch.bfloat16)
        else:
            q, k, v = joint_query, joint_key, joint_value

        if attention_mask is None:
            joint_hidden_states = dispatch_attention_fn(
                q, k, v,
                dropout_p=0.0,
                is_causal=False,
                backend=backend,
            )
        else:
            joint_hidden_states = dispatch_attention_fn(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=backend,
            )

        joint_hidden_states = joint_hidden_states.to(orig_dtype)

        # Reshape back. reshape(*shape[:2], -1) instead of .flatten(2, 3): same
        # dynamo-ONNX-export batch-baking issue as apply_rotary_emb_qwen above.
        joint_hidden_states = joint_hidden_states.reshape(*joint_hidden_states.shape[:2], -1)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back — action-first order (see concat above).
        action_attn_output = joint_hidden_states[:, :seq_action, :]
        txt_attn_output = joint_hidden_states[:, seq_action:, :]

        # Apply output projections
        action_attn_output = attn.to_out[0](action_attn_output.contiguous())
        if len(attn.to_out) > 1:
            action_attn_output = attn.to_out[1](action_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return action_attn_output, txt_attn_output

class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.action_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.action_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.action_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.action_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by action_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.zero_cond_t = zero_cond_t

    def _modulate(self, x, mod_params):
        """Apply AdaLN modulation to ``x`` (shape ``[B, L, dim]``).

        ``mod_params`` is the concatenation of ``shift, scale, gate``:
          - ``[B, 3*dim]``        -> shared across the whole sequence (broadcast over L)
          - ``[B, L, 3*dim]``     -> RTC per-step modulation, one set per position
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if shift.dim() == 2:  # [B, dim] -> broadcast over sequence
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
            gate = gate.unsqueeze(1)
        # else already [B, L, dim]: per-step modulation, used as-is
        return x * (1 + scale) + shift, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        txt_temb: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Action stream conditioning may be per-step under RTC (temb: [B, H, dim]);
        # the text stream always uses the standard (last-step) conditioning.
        action_mod_params = self.action_mod(temb)  # [B, 6*dim] or [B, H, 6*dim]

        if txt_temb is None:
            txt_temb = temb
        txt_mod_params = self.txt_mod(txt_temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        action_mod1, action_mod2 = action_mod_params.chunk(2, dim=-1)  # Each [B, (H,) 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        action_normed = self.action_norm1(hidden_states)
        action_modulated, action_gate1 = self._modulate(action_normed, action_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=action_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (action_output, txt_output) when encoder_hidden_states is provided
        action_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + action_gate1 * action_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        action_normed2 = self.action_norm2(hidden_states)
        action_modulated2, action_gate2 = self._modulate(action_normed2, action_mod2)
        action_mlp_output = self.action_mlp(action_modulated2)
        hidden_states = hidden_states + action_gate2 * action_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
        use_vlm_intermediate_layers: bool = False,
        num_vlm_intermediate_layers: int = 3,
        **kwargs
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Project the VLM (text-stream) features into the DiT latent space.
        # `cross_attention_dim` is the VLM hidden size, set by the framework
        # before construction (mirrors QwenGR00T cross_attention_dim alignment).
        txt_in_dim = cross_attention_dim if cross_attention_dim is not None else self.inner_dim
        self.txt_norm = RMSNorm(txt_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(txt_in_dim, self.inner_dim)
        compute_dtype = getattr(self.config, 'compute_dtype', torch.float32)
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=self.inner_dim, compute_dtype=compute_dtype
        )

        # Shared 1D RoPE for the joint (text + action) stream.
        self.rope = QwenActionTextRotaryEmbed(head_dim=self.config.attention_head_dim, theta=10000)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    zero_cond_t=False,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        vlm_intermediate_hidden_states: Optional[list] = None,
    ):
        # Encode timesteps. Standard schedule -> temb [B, D]; RTC schedule
        # (timestep [B, H]) -> temb [B, H, D] with one conditioning per action step.
        temb = self.timestep_encoder(timestep)
        # The text stream always uses the standard (last-step) conditioning; under
        # RTC reduce the per-step temb to its last step (matches passing
        # t_discretized_action[:, -1] to txt_mod). The output head uses the same.
        if temb.dim() == 3:
            txt_temb = temb[:, -1]  # [B, D]
        else:
            txt_temb = temb

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        seq_txt = encoder_hidden_states.shape[1]
        seq_action = hidden_states.shape[1]
        image_rotary_emb = self.rope(seq_txt, seq_action, hidden_states.device)

        # Build the joint key mask over [action, text] (the attn processor
        # concatenates keys action-first; see QwenDoubleStreamAttnProcessor2_0).
        # Action keys are always visible; text keys follow encoder_attention_mask
        # (True = attend). Placing the always-True action block first keeps the
        # combined True run anchored at 0 — required by FLASH_VARLEN's key
        # packer, which slices `key[b, :valid_len]` off the front of the raw
        # (unfiltered) per-sample key tensor rather than gathering at the
        # actual True positions; a [text, action] order would leave a
        # masked-out gap (assistant turn + right-padding) in the middle of an
        # otherwise-valid run and get the wrong keys attended to.
        # Shape [B, 1, 1, S_act+S_txt] broadcasts over heads and query
        # positions in SDPA.
        attention_mask = None
        if encoder_attention_mask is not None:
            txt_mask = encoder_attention_mask.to(dtype=torch.bool)  # [B, S_txt]
            action_mask = torch.ones(
                txt_mask.shape[0], seq_action, dtype=torch.bool, device=txt_mask.device
            )
            joint_mask = torch.cat([action_mask, txt_mask], dim=1)  # [B, S_act+S_txt]
            attention_mask = joint_mask[:, None, None, :]

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            block_kwargs = dict(encoder_hidden_states=encoder_hidden_states,
                                temb=temb, txt_temb=txt_temb,
                                image_rotary_emb=image_rotary_emb,
                                attention_mask=attention_mask)
            if self.training and self.gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                encoder_hidden_states, hidden_states = checkpoint(
                    block, hidden_states, use_reentrant=False, **block_kwargs)
            else:
                encoder_hidden_states, hidden_states = block(hidden_states, **block_kwargs)
            all_hidden_states.append(hidden_states)

        # Output processing. The final AdaLN modulates the ACTION stream, so it is
        # conditioned on the action diffusion time `temb` (NOT the reduced txt_temb):
        #   - standard schedule -> temb [B, D], broadcast over the action sequence,
        #   - RTC               -> temb [B, S_act, D], one conditioning per action step
        #                          (matches the per-step action_mod inside the blocks).
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=-1)
        if scale.dim() == 2:  # [B, D] -> broadcast over the action sequence
            shift, scale = shift[:, None], scale[:, None]
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)
