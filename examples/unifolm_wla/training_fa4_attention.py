"""Compiled-training bridge for flash-attn-4 4.0.0b37 on padded WLA inputs."""
from pathlib import Path
import site
import torch
import flash_attn

site.addsitedir(str(Path("/storage/wangdonglinLab/weizhenyu/kuimouyu/research/unifoLM-WLA-fp8/.attention-deps")))
flash_attn.__path__.insert(0,str(Path("/storage/wangdonglinLab/weizhenyu/kuimouyu/research/unifoLM-WLA-fp8/.attention-deps/flash_attn")))
from flash_attn.cute.interface import flash_attn_varlen_func, _flash_attn_bwd
from examples.unifolm_wla.training_optimizations import DETERMINISTIC


@torch.library.custom_op('wla_fa4::forward', mutates_args=())
def fa4_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor,
                causal: bool, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    return flash_attn_varlen_func(q, k, v, seqused_q=lengths, seqused_k=lengths,
        max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1], causal=causal,
        softmax_scale=scale, deterministic=DETERMINISTIC, return_lse=True)


@fa4_forward.register_fake
def fa4_forward_fake(q, k, v, lengths, causal, scale):
    return torch.empty(q.shape,device=q.device,dtype=q.dtype), torch.empty(
        (q.shape[0],q.shape[2],q.shape[1]),device=q.device,dtype=torch.float32)


@torch.library.custom_op('wla_fa4::backward', mutates_args=())
def fa4_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                 dout: torch.Tensor, lse: torch.Tensor, lengths: torch.Tensor,
                 causal: bool, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # seqused kernels only write valid rows; padding has zero derivative.
    dq, dk, dv = [torch.zeros(x.shape,device=x.device,dtype=x.dtype) for x in (q,k,v)]
    return _flash_attn_bwd(q,k,v,out,dout,lse,softmax_scale=scale,causal=causal,
        seqused_q=lengths,seqused_k=lengths,max_seqlen_q=q.shape[1],max_seqlen_k=k.shape[1],
        deterministic=DETERMINISTIC,dq=dq,dk=dk,dv=dv)


@fa4_backward.register_fake
def fa4_backward_fake(q,k,v,out,dout,lse,lengths,causal,scale):
    return tuple(torch.empty(x.shape,device=x.device,dtype=x.dtype) for x in (q,k,v))


def setup_context(ctx,inputs,output):
    q,k,v,lengths,ctx.causal,ctx.scale=inputs
    out,lse=output
    ctx.save_for_backward(q,k,v,out,lse,lengths)
    ctx.mark_non_differentiable(lse)


def backward(ctx,dout,dlse):
    q,k,v,out,lse,lengths=ctx.saved_tensors
    dq,dk,dv=fa4_backward(q,k,v,out,dout,lse,lengths,ctx.causal,ctx.scale)
    return dq,dk,dv,None,None,None


fa4_forward.register_autograd(backward,setup_context=setup_context)


def padded_attention(q,k,v,mask,*,causal=False,softmax_scale=None):
    lengths=mask.sum(-1,dtype=torch.int32)
    scale=q.shape[-1]**-0.5 if softmax_scale is None else softmax_scale
    head_dim=q.shape[-1]
    # FA4 b37 backward requires a 64-wide layout for WLA's 48-wide heads.
    if head_dim == 48:
        q,k,v=(torch.nn.functional.pad(x,(0,16)) for x in (q,k,v))
    out,_=fa4_forward(q,k,v,lengths,causal,scale)
    out=out[...,:head_dim]
    return torch.where(mask[:,:,None,None],out,0)


def dense_attention(q,k,v,dropout_p=0.0,softmax_scale=None,causal=False,deterministic=False):
    assert dropout_p == 0.0 and deterministic == DETERMINISTIC
    lengths=torch.full((q.shape[0],),q.shape[1],device=q.device,dtype=torch.int32)
    scale=q.shape[-1]**-0.5 if softmax_scale is None else softmax_scale
    out,_=fa4_forward(q,k,v,lengths,causal,scale)
    return out
