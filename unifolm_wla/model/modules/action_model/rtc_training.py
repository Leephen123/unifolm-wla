"""Clean executed prefixes for flow matching; unknown suffix alone contributes loss."""
import torch


def apply_rtc_prefix(actions, noise, times, prefix_lengths):
    horizon = actions.shape[1]
    prefix = torch.arange(horizon, device=actions.device)[None, :] < prefix_lengths[:, None]
    step_times = torch.where(prefix, torch.ones_like(times[:, None]), times[:, None])
    noisy = (1-step_times[..., None])*noise + step_times[..., None]*actions
    return noisy, step_times, (~prefix)[..., None]
