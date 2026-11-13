"""Fixed-shape full-pipeline block compilation; input contract checked each batch."""
import types
import torch
from torch.nn import functional as F
from flash_attn import flash_attn_func
from examples.unifolm_wla.training_optimizations import apply_candidate as apply_easy, DETERMINISTIC
from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
from unifolm_wla.model.modules.action_model.DiT_modules import mmdit


def text_attention(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
    assert past_key_values is None
    shape=(*hidden_states.shape[:-1],-1,self.head_dim)
    q=self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1,2)
    k=self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1,2)
    v=self.v_proj(hidden_states).view(shape)
    q,k=qwen.apply_rotary_pos_emb(q,k,*position_embeddings)
    if attention_mask is None:
        y=flash_attn_func(q.transpose(1,2),k.transpose(1,2),v,
            dropout_p=0.0,softmax_scale=self.scaling,causal=True,deterministic=DETERMINISTIC)
    else:
        assert attention_mask.ndim == 2
        length=hidden_states.shape[1]
        causal=torch.ones((length,length),device=q.device,dtype=torch.bool).tril()
        mask=causal[None,None] & attention_mask[:,None,None,:].bool()
        groups=q.shape[1]//k.shape[1]
        k=k.repeat_interleave(groups,dim=1)
        v=v.transpose(1,2).repeat_interleave(groups,dim=1)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
            dropout_p=0.0,scale=self.scaling).transpose(1,2)
    return self.o_proj(y.reshape(*hidden_states.shape[:-1],-1)),None


def vision_attention(self,hidden_states,cu_seqlens,rotary_pos_emb=None,position_embeddings=None,**kwargs):
    length=hidden_states.shape[0]
    q,k,v=self.qkv(hidden_states).reshape(length,3,self.num_heads,-1).permute(1,0,2,3).unbind(0)
    q,k=qwen.apply_rotary_pos_emb_vision(q,k,*position_embeddings)
    shape=(-1,560,self.num_heads,self.head_dim)
    y=flash_attn_func(q.reshape(shape),k.reshape(shape),v.reshape(shape),dropout_p=0.0,
        softmax_scale=self.scaling,causal=False,deterministic=DETERMINISTIC)
    return self.proj(y.reshape(length,-1))


def dense_action_attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,backend=None,**kwargs):
    if attn_mask is None:
        return flash_attn_func(q,k,v,dropout_p=dropout_p,causal=is_causal,deterministic=DETERMINISTIC)
    assert attn_mask.ndim == 4
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        attn_mask=attn_mask.bool(),dropout_p=dropout_p,is_causal=is_causal).transpose(1,2)


def apply_candidate(model,variant):
    if variant=='baseline':return apply_easy(model,'baseline')
    assert variant in ('dense','blocks')
    undo=[];restore_easy=apply_easy(model,'easy')
    def patch(obj,key,value):
        undo.append((obj,key,getattr(obj,key)));setattr(obj,key,value)
    builder=model.qwen_vl_interface.build_qwenvl_inputs
    def checked_inputs(*a,**kw):
        batch=builder(*a,**kw)
        size = batch['input_ids'].shape[0]
        assert batch['input_ids'].ndim == 2
        assert batch['attention_mask'].shape == batch['input_ids'].shape
        grid=batch['image_grid_thw']
        assert grid.shape==(size*3,3) and bool((grid==grid.new_tensor([1,20,28])).all())
        assert 'assistant_mask' not in batch
        return batch
    patch(model.qwen_vl_interface,'build_qwenvl_inputs',checked_inputs)
    patch(mmdit,'dispatch_attention_fn',dense_action_attention)
    for m in model.modules():
        if type(m).__name__=='Qwen3VLTextAttention':patch(m,'forward',types.MethodType(text_attention,m))
        if type(m).__name__=='Qwen3VLVisionAttention':patch(m,'forward',types.MethodType(vision_attention,m))
    if variant=='blocks':
        from transformers.utils.output_capturing import maybe_install_capturing_hooks
        maybe_install_capturing_hooks(model.qwen_vl_interface.model.model.language_model)
        maybe_install_capturing_hooks(model.qwen_vl_interface.model.model.visual)
        for module in model.modules():
            if type(module).__name__ in ('Qwen3VLTextAttention','Qwen3VLVisionAttention'):
                hooks=module._forward_hooks
                patch(module,'_forward_hooks',type(hooks)((k,v) for k,v in hooks.items() if v.__name__!='output_capturing_hook'))
    options={'emulate_precision_casts':True,'fallback_random':True}
    blocks=list(model.qwen_vl_interface.model.model.language_model.layers)+list(model.qwen_vl_interface.model.model.visual.blocks)+list(model.action_model.model.transformer_blocks)
    for m in (blocks if variant=='blocks' else []):
        patch(m,'forward',torch.compile(m.forward,fullgraph=True,dynamic=False,options=options))
    def restore():
        for obj,key,old in reversed(undo):setattr(obj,key,old)
        restore_easy()
    return restore
