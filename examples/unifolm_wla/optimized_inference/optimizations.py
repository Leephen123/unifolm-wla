from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
from transformers.integrations.sdpa_attention import sdpa_attention_forward

def attention(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
 shape=(*hidden_states.shape[:-1],-1,self.head_dim)
 q,k,v=self.qkv(hidden_states).split(self.qkv_sizes,dim=-1)
 q=self.q_norm(q.reshape(shape)).transpose(1,2)
 k=self.k_norm(k.reshape(shape)).transpose(1,2)
 v=v.reshape(shape).transpose(1,2)
 q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
 y,_=sdpa_attention_forward(self,q,k,v,attention_mask,dropout=0.,scaling=self.scaling,**kwargs)
 return self.o_proj(y.reshape(*hidden_states.shape[:-1],-1)),None

def mlp(self,x):
 gate,up=self.gate_up(x).chunk(2,dim=-1)
 return self.down_proj(self.act_fn(gate)*up)

def combine(parts):
 has_bias=parts[0].bias is not None
 assert all((p.bias is not None)==has_bias for p in parts)
 layer=nn.Linear(parts[0].in_features,sum(p.out_features for p in parts),bias=has_bias,device=parts[0].weight.device,dtype=parts[0].weight.dtype)
 layer.weight=nn.Parameter(torch.cat([p.weight for p in parts],dim=0),requires_grad=False)
 if has_bias:layer.bias=nn.Parameter(torch.cat([p.bias for p in parts]),requires_grad=False)
 return layer

def fuse_text(core):
 for layer in core.backbone.language_model.layers:
  a=layer.self_attn
  a.qkv_sizes=[a.q_proj.out_features,a.k_proj.out_features,a.v_proj.out_features]
  a.qkv=combine([a.q_proj,a.k_proj,a.v_proj])
  del a.q_proj,a.k_proj,a.v_proj
  a.forward=MethodType(attention,a)
  m=layer.mlp
  m.gate_up=combine([m.gate_proj,m.up_proj])
  del m.gate_proj,m.up_proj
  m.forward=MethodType(mlp,m)

from types import MethodType
import torch
from torch.nn import functional as F

def block_forward(self,hidden_states,encoder_hidden_states,temb,txt_temb=None,encoder_hidden_states_mask=None,image_rotary_emb=None,attention_mask=None,joint_attention_kwargs=None):
 am1,am2=self.action_mod_cache[temb].chunk(2,-1)
 tm1,tm2=self.txt_mod_cache[temb].chunk(2,-1)
 a,ag=self._modulate(self.action_norm1(hidden_states),am1)
 t,tg=self._modulate(self.txt_norm1(encoder_hidden_states),tm1)
 ao,to=self.attn(hidden_states=a,encoder_hidden_states=t,encoder_hidden_states_mask=None,attention_mask=None,image_rotary_emb=image_rotary_emb)
 hidden_states=hidden_states+ag*ao
 encoder_hidden_states=encoder_hidden_states+tg*to
 a,ag=self._modulate(self.action_norm2(hidden_states),am2)
 t,tg=self._modulate(self.txt_norm2(encoder_hidden_states),tm2)
 return encoder_hidden_states+tg*self.txt_mlp(t),hidden_states+ag*self.action_mlp(a)

def dit_forward(self,hidden_states,encoder_hidden_states,timestep):
 index=timestep//250
 context=self.txt_in(self.txt_norm(encoder_hidden_states.contiguous()))
 for block in self.transformer_blocks:
  context,hidden_states=block(hidden_states,context,temb=index,image_rotary_emb=self.cached_rope)
 shift,scale=self.output_mod_cache[index].chunk(2,-1)
 hidden_states=self.norm_out(hidden_states)*(1+scale[:,None])+shift[:,None]
 return self.proj_out_2(hidden_states)

@torch.inference_mode()
def cache_conditions(core):
 m=core.head.model
 with torch.autocast('cuda',dtype=torch.bfloat16):
  # Preserve batch-one arithmetic while evaluating the four fixed time conditions.
  times=[torch.tensor([i*250],device='cuda') for i in range(4)]
  embs=[m.timestep_encoder(t) for t in times]
  for block in m.transformer_blocks:
   block.register_buffer('action_mod_cache',torch.cat([block.action_mod(e) for e in embs]))
   block.register_buffer('txt_mod_cache',torch.cat([block.txt_mod(e) for e in embs]))
   block.forward=MethodType(block_forward,block)
  m.register_buffer('output_mod_cache',torch.cat([m.proj_out_1(F.silu(e)) for e in embs]))
  m.cached_rope=m.rope(core.ids.shape[1],50,core.ids.device)
  m.forward=MethodType(dit_forward,m)
