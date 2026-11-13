import os
import torch
from flash_attn import flash_attn_varlen_func
from diffusers.models import attention_dispatch as dispatch
from unifolm_wla.model.modules.action_model.DiT_modules import mmdit
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

ORIGINAL_DISPATCH=mmdit.dispatch_attention_fn
DETERMINISTIC=os.environ.get('FLASH_ATTENTION_DETERMINISTIC')=='1'

def vector_attention(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,backend=None,**kwargs):
    batch,seqlen,_,_=q.shape
    normalized=dispatch._normalize_attn_mask(attn_mask,batch,k.shape[1])
    (_,lengths),(cuq,cuk),(maxq,maxk)=dispatch._prepare_for_flash_attn_or_sage_varlen(
        batch,seqlen,k.shape[1],attn_mask=normalized,device=q.device)
    indices=(torch.arange(k.shape[1],device=k.device)[None,:]<lengths[:,None]).flatten().nonzero().squeeze(-1)
    kp=k.flatten(0,1).index_select(0,indices)
    vp=v.flatten(0,1).index_select(0,indices)
    out=flash_attn_varlen_func(q.flatten(0,1),kp,vp,cuq,cuk,maxq,maxk,
        dropout_p=dropout_p,causal=is_causal,deterministic=DETERMINISTIC)
    return out.unflatten(0,(batch,-1))

def apply_candidate(model, variant):
    assert variant in ('baseline','easy')
    undo=[]
    def patch(obj,key,value):
        old=getattr(obj,key);undo.append((obj,key,old));setattr(obj,key,value)
    if DETERMINISTIC:
        def deterministic(*a,**kw):
            kw['attention_kwargs']={'deterministic':True}
            return ORIGINAL_DISPATCH(*a,**kw)
        patch(mmdit,'dispatch_attention_fn',deterministic)
    if variant!='baseline':
        patch(mmdit,'dispatch_attention_fn',vector_attention)
        original=model.qwen_vl_interface.model.forward
        def no_full_logits(*a,**kw):
            kw['logits_to_keep']=1
            return original(*a,**kw)
        patch(model.qwen_vl_interface.model,'forward',no_full_logits)
        processor=model.qwen_vl_interface.processor
        original_template=processor.apply_chat_template
        def gpu_processor(*a,**kw):
            kw['processor_kwargs']['images_kwargs']={'device':'cuda'}
            return original_template(*a,**kw)
        patch(processor,'apply_chat_template',gpu_processor)
    def restore():
        for obj,key,old in reversed(undo):setattr(obj,key,old)
    return restore
