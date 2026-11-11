from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from unifolm_wla.model.modules.action_model.DiT_modules.mmdit import DiT
from unifolm_wla.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """Encode a (noised) action trajectory together with its diffusion time.

    `timesteps`: one diffusion time per batch item, shape `[B]`, replicated
    across the `T` action steps. Returns `[B, T, hidden_size]`.
    """

    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size, bias=False)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        B, T, _ = actions.shape
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            assert timesteps.shape == (B, T)

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)  # [B, T, w]
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x


@dataclass
class MMDiTFlowmatchingActionHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(default=True)
    diffusion_model_cfg: dict = field(default=None)
    input_embedding_dim: int = field(default=1536)
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    num_timestep_buckets: int = field(default=1000)
    num_inference_timesteps: int = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# Head dims for the MMDiT backbone (inner_dim = num_heads * head_dim).
DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}

# Embodiment-type conditioning. Single-arm sources (bridge/fractal/libero/...) all
# map into the SHARED left action slot, so without an embodiment signal the shared
# decoder averages their conflicting distributions and the left arm collapses to ~0.
# A per-robot_type embedding (injected after action_encoder, before the DiT) lets the
# shared encoder/decoder stay embodiment-agnostic while the DiT disambiguates robots.
# Order is the serialized identity contract: keep append-only so existing checkpoints
# map the same ids. Index 0 is reserved for "unknown".
DEFAULT_EMBODIMENT_TYPES = [
    "unitree",
    "aloha",
    "fractal",
    "franka",
    "bridge",
    "robochallenge_single",
    "robochallenge_dual",
]


def build_embodiment_map(embodiment_types) -> dict:
    """Map robot_type string -> embedding index. Index 0 is reserved for unknown."""
    return {name: i + 1 for i, name in enumerate(embodiment_types)}


class MMDiTFlowmatchingActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        # --- DiT backbone selection (MMDiT double-stream) ---
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        config.update(action_model_cfg)  # merge into main config for convenience
        config.diffusion_model_cfg.update(action_model_cfg)  # merge into main config for convenience
        self.full_config.framework.action_model.update(action_model_cfg)  # also update the full config for downstream access (e.g. by the trainer)
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)

        self.action_horizon = int(config.action_horizon)

        # --- Action / state dims ---
        self.action_output_dim = config.action_dim  # raw robot DoF
        encoder_action_dim = config.action_dim * 2   # action + mask concatenated

        self.action_encoder = ActionEncoder(
            action_dim=encoder_action_dim,
            hidden_size=self.input_embedding_dim,
        )

        self.num_inference_timesteps = config.num_inference_timesteps
        self.hidden_size = config.hidden_size

        # Optional state token (auto-detect: state_dim == 0 disables it, e.g.
        # when the VLM robot_state_projector injects state into the text stream).
        self.state_encoder = (
            MLP(
                input_dim=config.state_dim * 2,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )

        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_output_dim,
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # --- Flow-matching noise schedule ---
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

        # --- Embodiment-type conditioning (FiLM / AdaLN gain) ---
        # One shared per-robot_type embedding, projected to (scale, shift) and applied
        # affinely (x*(1+scale)+shift) at two points so the shared encoder/decoder can
        # disambiguate embodiments (single-arm sources share the left action slot):
        #   - pre-DiT  (input_embedding_dim): modulates the action tokens into the DiT,
        #   - pre-decoder (DiT output_dim): modulates the output right where the shared
        #     decoder maps features -> action and the left slot collapses.
        # AdaLN-Zero: the FiLM projections are zero-init so scale=shift=0 => no-op at
        # step 0, matching the no-conditioning baseline. Index 0 = unknown.
        self.embodiment_types = list(config.get("embodiment_types", DEFAULT_EMBODIMENT_TYPES))
        self.embodiment_map = build_embodiment_map(self.embodiment_types)
        out_dim = self.model.config.output_dim
        self.embodiment_embedding = nn.Embedding(
            len(self.embodiment_types) + 1, self.input_embedding_dim
        )
        self.embodiment_film_pre = nn.Linear(self.input_embedding_dim, 2 * self.input_embedding_dim)
        self.embodiment_film_dec = nn.Linear(self.input_embedding_dim, 2 * out_dim)
        for film in (self.embodiment_film_pre, self.embodiment_film_dec):
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

        self.config = config

    @staticmethod
    def _apply_film(x, film_out):
        """Affine FiLM modulation broadcast over the token axis.

        x: (B, L, C); film_out: (B, 2C) -> split into (scale, shift). Returns x*(1+scale)+shift.
        """
        scale, shift = film_out.chunk(2, dim=-1)
        return x * (1 + scale[:, None, :].to(x.dtype)) + shift[:, None, :].to(x.dtype)

    def _add_embodiment_embed(self, action_features, body_type_ids):
        """Pre-DiT FiLM on the action tokens (broadcast over H).

        action_features: (B, H, embed_dim); body_type_ids: (B,) long, or None.
        """
        if body_type_ids is None:
            return action_features
        emb = self.embodiment_embedding(body_type_ids)  # (B, embed_dim)
        return self._apply_film(action_features, self.embodiment_film_pre(emb))

    def _add_embodiment_embed_dec(self, model_output, body_type_ids):
        """Pre-decoder FiLM on the DiT output (broadcast over S+H).

        model_output: (B, S+H, out_dim); body_type_ids: (B,) long, or None. Modulates all
        tokens; only the trailing H action rows are kept after the decoder.
        """
        if body_type_ids is None:
            return model_output
        emb = self.embodiment_embedding(body_type_ids)  # (B, embed_dim) — SAME table
        return self._apply_film(model_output, self.embodiment_film_dec(emb))

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return self.config.noise_s * (1 - sample)

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _add_action_pos_embed(self, action_features, device):
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        return action_features

    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None,
        encoder_attention_mask=None, body_type_ids: torch.Tensor = None,
    ):
        """
        vl_embs: (B, seq_length, feature_dim) — joint text stream.
        actions: (B, action_horizon, action_dim[*2]) — action, optionally with mask concatenated.
        body_type_ids: (B,) long embodiment ids for the per-robot embedding, or None.
        """
        device = vl_embs.device
        B, H = actions.shape[0], actions.shape[1]

        actions_only = actions[:, :, : self.action_output_dim]
        mask_part = actions[:, :, self.action_output_dim :]  # (B, H, action_output_dim) or empty
        noise = torch.randn(actions_only.shape, device=actions.device, dtype=actions.dtype)

        t = self.sample_time(B, device=actions.device, dtype=actions.dtype)  # (B,)
        t_b = t[:, None, None]  # (B, 1, 1)
        t_discretized_action = (t * self.num_timestep_buckets).long()  # (B,)

        noisy_actions = (1 - t_b) * noise + t_b * actions_only
        rtc_loss_mask = None
        rtc = self.full_config.framework.action_model.get('training_rtc', None)
        if self.training and rtc is not None and rtc.enabled:
            from .rtc_training import apply_rtc_prefix
            assert 0 <= rtc.min_prefix <= rtc.max_prefix < H
            lengths = torch.randint(rtc.min_prefix, rtc.max_prefix+1, (B,), device=device)
            enabled = torch.rand(B, device=device) < rtc.probability
            lengths = torch.where(enabled, lengths, torch.zeros_like(lengths))
            noisy_actions, step_times, rtc_loss_mask = apply_rtc_prefix(actions_only, noise, t, lengths)
            t_discretized_action = (step_times * self.num_timestep_buckets).long()
        # Zero the noised input on invalid DOFs so the encoder sees the same thing it
        # sees at inference (predict_action masks `actions` every step). Loss is already
        # mask-weighted, so this only fixes the train/inference input mismatch.
        if mask_part.shape[2] > 0:
            noisy_actions = noisy_actions * mask_part
        velocity = actions_only - noise
        noisy_trajectory = (
            torch.cat([noisy_actions, mask_part], dim=2) if mask_part.shape[2] > 0 else noisy_actions
        )

        action_features = self.action_encoder(noisy_trajectory, t_discretized_action)
        action_features = self._add_action_pos_embed(action_features, device)
        action_features = self._add_embodiment_embed(action_features, body_type_ids)

        # Pure action stream, with an optional leading state token.
        state_features = self.state_encoder(state) if (self.state_encoder is not None and state is not None) else None
        if state_features is not None:
            hidden_states = torch.cat((state_features, action_features), dim=1)
        else:
            hidden_states = action_features

        model_output = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized_action,
        )
        model_output = self._add_embodiment_embed_dec(model_output, body_type_ids)
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -H:]

        sq_err = (pred_actions - velocity) ** 2

        # Loss weight = action-DoF mask.
        weight = mask_part if mask_part.shape[2] > 0 else None

        if rtc_loss_mask is not None:
            weight = rtc_loss_mask.expand_as(sq_err).to(sq_err.dtype) if weight is None else weight * rtc_loss_mask
        if weight is not None:
            loss = (sq_err * weight).sum() / weight.sum().clamp(min=1)
        else:
            loss = sq_err.mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        encoder_attention_mask=None,
        body_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_output_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        mask_part = None
        if action_mask is not None:
            mask_part = action_mask.float().unsqueeze(1).expand(-1, self.action_horizon, -1).contiguous()
            actions = actions * mask_part

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if (self.state_encoder is not None and state is not None) else None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

            encoder_input = torch.cat([actions, mask_part], dim=2) if mask_part is not None else actions
            action_features = self.action_encoder(encoder_input, timesteps_tensor)
            action_features = self._add_action_pos_embed(action_features, device)
            action_features = self._add_embodiment_embed(action_features, body_type_ids)

            if state_features is not None:
                hidden_states = torch.cat((state_features, action_features), dim=1)
            else:
                hidden_states = action_features

            model_output = self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            model_output = self._add_embodiment_embed_dec(model_output, body_type_ids)
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            actions = (
                (actions + dt * pred_velocity) * mask_part
                if mask_part is not None
                else actions + dt * pred_velocity
            )
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """Factory: build MMDiTFlowmatchingActionHead from global framework config."""
    return MMDiTFlowmatchingActionHead(full_config=config)


if __name__ == "__main__":
    pass
