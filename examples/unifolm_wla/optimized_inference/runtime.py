"""Compiled current-observation WLA inference; no training or RTC dependencies."""
import json
from pathlib import Path
from types import MethodType
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import AutoConfig, AutoProcessor, Qwen3VLModel
from transformers.initialization import no_init_weights
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

from .kernels import image_patches
from .optimizations import fuse_text, cache_conditions
from .architecture.action_head import MMDiTFlowmatchingActionHead

STATE_SLOTS = list(range(48,54)) + list(range(54,60)) + list(range(32,35)) + list(range(0,7)) + list(range(16,23)) + list(range(41,47)) + list(range(10,16)) + list(range(26,32))
ROLES = ['head_left', 'cam_wrist_left', 'cam_wrist_right']


def fixed_vision_attention(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
    q, k, v = self.qkv(hidden_states).reshape(self.batch_size*1680, 3, self.num_heads, self.head_dim).permute(1,0,2,3).unbind(0)
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    q = q.reshape(self.batch_size*3,560,self.num_heads,self.head_dim).transpose(1,2)
    k = k.reshape(self.batch_size*3,560,self.num_heads,self.head_dim).transpose(1,2)
    v = v.reshape(self.batch_size*3,560,self.num_heads,self.head_dim).transpose(1,2)
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=False, scale=self.scaling)
    return self.proj(y.transpose(1,2).reshape(self.batch_size*1680, self.dim))


class Core(nn.Module):
    def __init__(self, source_model, cfg, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.state_frames = 1 + int(cfg.datasets.vla_data.get("history_frames", 0))
        config = AutoConfig.from_pretrained(cfg.framework.qwenvl.base_vlm, local_files_only=True)
        config._attn_implementation = 'sdpa'
        config.text_config._attn_implementation = 'sdpa'
        config.vision_config._attn_implementation = 'sdpa'
        with no_init_weights():
            self.backbone = Qwen3VLModel(config).to(dtype=torch.bfloat16)
        self.projector = nn.Module()
        self.projector.net = nn.Sequential(nn.Linear(120,2560),nn.SiLU(),nn.Linear(2560,2560))
        self.head = MMDiTFlowmatchingActionHead(cfg)
        self.to(device='cuda', dtype=torch.bfloat16)
        weights = source_model.state_dict()
        mapped = {}
        for name, value in weights.items():
            if name.startswith('qwen_vl_interface.model.model.'):
                mapped['backbone.' + name.removeprefix('qwen_vl_interface.model.model.')] = value
            elif name.startswith('qwen_vl_interface.model.robot_state_projector.'):
                mapped['projector.' + name.removeprefix('qwen_vl_interface.model.robot_state_projector.')] = value
            elif name.startswith('action_model.'):
                mapped['head.' + name.removeprefix('action_model.')] = value
            else:
                assert name == 'qwen_vl_interface.model.lm_head.weight', name
        self.load_state_dict(mapped, strict=True)
        del mapped, weights
        self.to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
        assert self.head.action_horizon == 50 and self.head.num_inference_timesteps == 4
        for block in self.backbone.visual.blocks:
            block.attn.batch_size = batch_size
            block.attn.forward = MethodType(fixed_vision_attention, block.attn)
        statistics = json.loads(Path(cfg.datasets.vla_data.statistics_path).read_text())['full_pipeline']
        for name, section, key in [('state_offset','state','offset'),('state_scale','state','scale'),
                                    ('action_offset','action','offset'),('action_scale','action','scale')]:
            self.register_buffer(name, torch.tensor(statistics[section][key], device='cuda', dtype=torch.float32))
        self.register_buffer('state_slots', torch.tensor(STATE_SLOTS, device='cuda'))
        mask = torch.zeros(self.batch_size,self.state_frames,60,device='cuda');mask[:,:,self.state_slots] = 1
        self.register_buffer('state_mask', mask)
        self.register_buffer('body_ids', torch.ones(batch_size,device='cuda',dtype=torch.long))
        self.register_buffer('action_mask', torch.ones(batch_size,50,114,device='cuda'))
        grid = torch.tensor([[1,20,28]]*(3*batch_size),device='cuda')
        visual = self.backbone.visual
        self.register_buffer('vision_positions', visual.fast_pos_embed_interpolate(grid))
        rope = visual.rot_pos_emb(grid)
        rope = torch.cat((rope,rope),-1)
        self.register_buffer('vision_cos',rope.cos())
        self.register_buffer('vision_sin',rope.sin())
        self.register_buffer('vision_lengths',torch.arange(0,(3*batch_size+1)*560,560,device='cuda',dtype=torch.int32))
        fuse_text(self)

    def set_prompt(self, packed):
        ids = packed['input_ids'].to('cuda')
        self.ids = ids
        self.image_indices = (ids[0] == self.backbone.config.image_token_id).nonzero().flatten()
        self.state_indices = (ids[0] == packed['state_token_id']).nonzero().flatten()
        assert len(self.image_indices) == 420 and len(self.state_indices) == self.state_frames
        self.text_embeddings = self.backbone.get_input_embeddings()(ids).expand(self.batch_size,-1,-1).contiguous()
        position_ids, _ = self.backbone.get_rope_index(
            input_ids=ids, mm_token_type_ids=packed['mm_token_type_ids'].to('cuda'),
            image_grid_thw=packed['image_grid_thw'].to('cuda'),
            attention_mask=packed['attention_mask'].to('cuda'))
        self.text_cos, self.text_sin = self.backbone.language_model.rotary_emb(self.text_embeddings, position_ids)
        cache_conditions(self)

    def vision(self, pixels):
        visual = self.backbone.visual
        hidden = visual.patch_embed(pixels.to(torch.bfloat16)) + self.vision_positions
        features = []
        for i, block in enumerate(visual.blocks):
            hidden = block(hidden, cu_seqlens=self.vision_lengths,
                           position_embeddings=(self.vision_cos,self.vision_sin))
            if i in visual.deepstack_visual_indexes:
                features.append(visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(i)](hidden))
        return visual.merger(hidden), features

    def text(self, images, deepstack, state, history_valid):
        state47 = ((state - self.state_offset) / self.state_scale) * history_valid[:,:,None]
        state60 = torch.zeros(self.batch_size,self.state_frames,60,device=state.device).index_copy(2,self.state_slots,state47)
        state_embedding = self.projector.net(torch.cat((state60,self.state_mask * history_valid[:,:,None]),-1).to(torch.bfloat16))
        hidden = self.text_embeddings.index_copy(1,self.image_indices,images.reshape(self.batch_size,420,-1))
        hidden = hidden.index_copy(1,self.state_indices,state_embedding)
        text = self.backbone.language_model
        for i, layer in enumerate(text.layers):
            hidden = layer(hidden, attention_mask=None, past_key_values=None,
                           position_embeddings=(self.text_cos,self.text_sin), use_cache=False)
            if i < len(deepstack):
                hidden = hidden.index_add(1,self.image_indices,deepstack[i].reshape(self.batch_size,420,-1).to(hidden.dtype))
        return text.norm(hidden)

    def flow_step(self, actions, context, timestep):
        head = self.head
        encoded = head.action_encoder(torch.cat((actions,self.action_mask),-1),timestep)
        encoded = head._add_action_pos_embed(encoded,actions.device)
        encoded = head._add_embodiment_embed(encoded,self.body_ids)
        result = head.model(encoded,encoder_hidden_states=context,timestep=timestep)
        result = head._add_embodiment_embed_dec(result,self.body_ids)
        velocity = head.action_decoder(result)
        return actions + .25 * velocity.float()


class Runtime:
    def __init__(self, source_model, cfg, batch_size=1):
        torch.set_num_threads(8)
        self.processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm,local_files_only=True)
        self.processor.tokenizer.padding_side = 'right'
        self.batch_size = batch_size
        self.state_frames = 1 + int(cfg.datasets.vla_data.get("history_frames", 0))
        self.core = Core(source_model,cfg,batch_size)
        self.prompt = None
        self.images = torch.zeros((batch_size*3,320,448,3),device='cuda',dtype=torch.uint8)
        self.state = torch.zeros((batch_size,self.state_frames,47),device='cuda')
        self.history_valid = torch.ones((batch_size,self.state_frames),device='cuda')
        self.noise = torch.zeros((batch_size,50,114),device='cuda')
        self.timesteps = [torch.tensor([i*250]*batch_size,device='cuda') for i in range(4)]
        self.graph = None

    def pack_prompt(self, prompt):
        from PIL import Image
        content = []
        for i, role in enumerate(ROLES):
            content.extend([{'type':'text','text':('' if i == 0 else '\n')+role+': '},
                            {'type':'image','image':Image.new('RGB',(448,320))}])
        state_tokens = '<|robot_state|>' * self.state_frames
        content.append({'type':'text','text':f'\nTask: {prompt}\nState: <|robot_state_implicit_stats|>{state_tokens}\nControl Mode: Arms: MOCAP, LOW BODY: MOCAP'})
        packed = self.processor.apply_chat_template([[{'role':'user','content':content}]],tokenize=True,
                    add_generation_prompt=True,return_dict=True,processor_kwargs={'padding':True,'return_tensors':'pt'})
        packed['state_token_id'] = self.processor.tokenizer.convert_tokens_to_ids('<|robot_state|>')
        return packed

    def forward(self):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            image_features, deepstack = self.vision(image_patches(self.images))
            context = self.text(image_features,deepstack,self.state,self.history_valid)
            actions = self.noise
            for timestep in self.timesteps:
                actions = self.flow_step(actions,context,timestep)
            return actions

    @torch.inference_mode()
    def prepare(self, prompt, compile_model=True):
        self.graph = None
        self.core.set_prompt(self.pack_prompt(prompt))
        self.prompt = prompt
        if compile_model:
            self.vision = torch.compile(self.core.vision,fullgraph=True,mode='max-autotune-no-cudagraphs')
            self.text = torch.compile(self.core.text,fullgraph=True,mode='max-autotune-no-cudagraphs')
            self.flow_step = torch.compile(self.core.flow_step,fullgraph=True,mode='max-autotune-no-cudagraphs')
        else:
            self.vision,self.text,self.flow_step = self.core.vision,self.core.text,self.core.flow_step
        print('PREPARED PROMPT',self.core.ids.shape[-1], 'compile',compile_model,flush=True)

    @torch.inference_mode()
    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):self.output = self.forward()
        self.graph = graph

    @torch.inference_mode()
    def infer(self, images, state, history_valid, seed=None):
        self.images.copy_(torch.as_tensor(images,device='cpu').reshape(self.batch_size*3,320,448,3))
        self.state.copy_(torch.as_tensor(state,device='cpu').reshape(self.batch_size,self.state_frames,47))
        self.history_valid.copy_(torch.as_tensor(history_valid,device='cpu'))
        if seed is not None:torch.manual_seed(seed)
        self.noise.copy_(torch.randn((self.batch_size,50,114),device='cuda',dtype=torch.bfloat16))
        if self.graph is None:result = self.forward()
        else:
            self.graph.replay()
            result = self.output
        return result.float().cpu().numpy()

    @torch.inference_mode()
    def refresh(self, source_model):
        """Copy live ZeRO-2 replicated weights without replacing captured storage."""
        source = source_model.state_dict()
        prefixes = {'backbone.': 'qwen_vl_interface.model.model.',
                    'projector.': 'qwen_vl_interface.model.robot_state_projector.',
                    'head.': 'action_model.'}
        for name, parameter in self.core.named_parameters():
            prefix = next(p for p in prefixes if name.startswith(p))
            key = prefixes[prefix] + name[len(prefix):]
            if '.self_attn.qkv.' in key:
                value = torch.cat([source[key.replace('.qkv.', '.'+part+'.')]
                                   for part in ('q_proj','k_proj','v_proj')], dim=0)
            elif '.mlp.gate_up.' in key:
                value = torch.cat([source[key.replace('.gate_up.', '.'+part+'.')]
                                   for part in ('gate_proj','up_proj')], dim=0)
            else:
                value = source[key]
            parameter.copy_(value)
        core = self.core
        grid = torch.tensor([[1,20,28]] * (3*self.batch_size), device='cuda')
        core.vision_positions.copy_(core.backbone.visual.fast_pos_embed_interpolate(grid))
        core.text_embeddings.copy_(core.backbone.get_input_embeddings()(core.ids).expand(self.batch_size,-1,-1))
        head = core.head.model
        with torch.autocast('cuda', dtype=torch.bfloat16):
            embs = [head.timestep_encoder(torch.tensor([i*250],device='cuda')) for i in range(4)]
            for block in head.transformer_blocks:
                block.action_mod_cache.copy_(torch.cat([block.action_mod(e) for e in embs]))
                block.txt_mod_cache.copy_(torch.cat([block.txt_mod(e) for e in embs]))
            head.output_mod_cache.copy_(torch.cat([head.proj_out_1(F.silu(e)) for e in embs]))
