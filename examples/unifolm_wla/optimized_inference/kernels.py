"""Fixed 3-camera Qwen3-VL patch packing, fused with normalization."""
import torch
import triton
import triton.language as tl


@triton.jit
def _image_patches(images, output, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    patch = i // 1536
    element = i % 1536
    camera = patch // 560
    patch = patch % 560
    channel = element // 512
    pixel = element % 256
    # Layout: grid_h/2, grid_w/2, merge_h, merge_w, C, time, p_h, p_w.
    row = (patch // 56 * 2 + patch % 4 // 2) * 16 + pixel // 16
    col = (patch % 56 // 4 * 2 + patch % 2) * 16 + pixel % 16
    address = ((camera * 320 + row) * 448 + col) * 3 + channel
    value = tl.load(images + address, i < N, other=0).to(tl.float32)
    value = (value * (1.0 / 255.0) - 0.5) / 0.5
    tl.store(output + i, value, i < N)


def image_patches(images):
    output = torch.empty((images.shape[0]*560, 1536), dtype=torch.float32, device=images.device)
    _image_patches[(triton.cdiv(output.numel(), 512),)](images, output, output.numel(), 512,
                                                     enable_fp_fusion=False)
    return output
