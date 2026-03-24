import torch
import torch.nn as nn
from types import MethodType
from .echo_partial_mlp_think import EchoPartialMLP  # 导入新的 MLP
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from typing import Callable, Optional, Union
import os
import transformers

from flash_attn import flash_attn_with_kvcache

from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from typing import Tuple
from .think_utils import repeat_kv, key_pruner_query_driven, recover_cache

def print_gpu_memory(tag=""):
    if not torch.cuda.is_available():
        return
    
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"[{tag}] GPU Mem: Used={allocated:.2f}GB | Cached={reserved:.2f}GB | Peak={max_allocated:.2f}GB")

def calculate_storage_memory(past_key_values, gap_storage):

    kv_bytes = 0
    gap_bytes = 0
    
    if past_key_values is not None:
        if hasattr(past_key_values, 'layers'): 
            for layer in past_key_values.layers:
                if layer.keys is not None:
                    kv_bytes += layer.keys.numel() * layer.keys.element_size()
                if layer.values is not None:
                    kv_bytes += layer.values.numel() * layer.values.element_size()
        elif isinstance(past_key_values, (tuple, list)): # Tuple of tuples
            for k, v in past_key_values:
                kv_bytes += k.numel() * k.element_size()
                kv_bytes += v.numel() * v.element_size()

    if gap_storage:
        for layer_idx, content in gap_storage.items():
            if 'k' in content and isinstance(content['k'], torch.Tensor):
                gap_bytes += content['k'].numel() * content['k'].element_size()
            if 'v' in content and isinstance(content['v'], torch.Tensor):
                gap_bytes += content['v'].numel() * content['v'].element_size()

    return kv_bytes / 1024**3, gap_bytes / 1024**3  # 转换为 GB


class EchoMistralPartialManager(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.block_size = args.block_size
        self.is_echo_mode = False 
        self.sink_size = getattr(args, 'sink_size', 4)
        self.local_window_size = getattr(args, 'local_window_size', 128)
        
        self.echo_input_mode = getattr(args, 'echo_input_mode', 'joint')
        
        self.total_save_dim = getattr(args, 'total_save_dim', 1024) 
        print(f"EchoPartial Mode: block_size={self.block_size}, total_save_dim={self.total_save_dim}")

        self.training_buffer = {} 


        self.think_ratio = 1 - getattr(args, 'compress_ratio', 0.5)          
        self.think_masks = {}    # {layer_idx: mask_tensor}
        self.think_k_storage = {} # {layer_idx: flat_pruned_k_tensor}
        
        # Initial EchoPartialMLP
        self.echo_mlps = nn.ModuleList([
            EchoPartialMLP(
                model.config, 
                total_save_dim=self.total_save_dim, 
                input_mode=self.echo_input_mode
            ) 
            for _ in range(model.config.num_hidden_layers)
        ]).to(model.dtype)

        self.gap_storage = {}
        
        self.is_training = False
        self.original_forwards = {}
        self._apply_patch()

    def save_mlps(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        print(f"Saving Echo MLPs to {path} ...")
        torch.save(self.echo_mlps.state_dict(), path)

    def load_mlps(self, path):
        print(f"Loading Echo MLPs from {path} ...")
        state_dict = torch.load(path, map_location=self.model.device)
        self.echo_mlps.load_state_dict(state_dict)

    def set_mode(self, echo_active: bool):
        self.is_echo_mode = echo_active
        # clear Buffer
        self.training_buffer = {}
        self.gap_storage = {}

    def _apply_patch(self):
        for i, layer in enumerate(self.model.model.layers):
            original_attn = layer.self_attn
            self.original_forwards[i] = original_attn.forward
            
            layer.self_attn.forward = MethodType(
                self._create_echo_forward(i, self.echo_mlps[i]), 
                original_attn
            )
            
        transformers.models.mistral.modeling_mistral.MistralModel.forward = echo_MistralModel_forward

    def get_anchor_info(self, layer_idx):
    
        num_layers = self.model.config.num_hidden_layers
        remainder = num_layers % self.block_size
        if layer_idx < remainder:
            return 0, layer_idx 

        adjusted_idx = layer_idx - remainder
        
        block_id_relative = adjusted_idx // self.block_size
        
        anchor_relative = block_id_relative * self.block_size
        
        anchor_idx = anchor_relative + remainder
        
        
        return block_id_relative, anchor_idx


    def _create_echo_forward(self, layer_idx, mlp_module):
        manager = self
        
        def echo_forward(
            self_attn_module,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        ):

            bsz, q_len, _ = hidden_states.size()
            block_id, anchor_idx = manager.get_anchor_info(layer_idx)
            is_anchor = (layer_idx == anchor_idx)
            if manager.block_size == 1:
                is_anchor = False 

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self_attn_module.head_dim)

            query_states = self_attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self_attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self_attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states = repeat_kv(key_states, self_attn_module.num_key_value_groups) # for think
            # value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            final_key_states = key_states
            final_value_states = value_states
            k_recovered = key_states

            S = manager.sink_size
            W = manager.local_window_size

            n_heads_keep = manager.total_save_dim // self_attn_module.head_dim

            # print(manager.total_save_dim, self_attn_module.head_dim)
            # input()

            # prefilling 阶段
            if q_len > 1:
                if manager.is_echo_mode or manager.is_training:
                    k_pruned, _, mask = key_pruner_query_driven(key_states[:,:,S:-W,:], query_states, 0, manager.think_ratio)
                    k_recovered = recover_cache(k_pruned, mask, manager.think_ratio)
                    k_recovered = torch.cat([key_states[:, :, :S, :], k_recovered, key_states[:, :, -W:, :]], dim=2)

                
                if past_key_values is not None:

                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    
                    if is_anchor or not manager.is_echo_mode:
                        k_recovered, value_states = past_key_values.update(k_recovered, value_states, self_attn_module.layer_idx, cache_kwargs)
                        final_value_states = value_states
                        final_key_states = key_states

                    else :
                        if q_len > (S + W):
                            if W == 0:
                                v_to_save = value_states[:, :, :S, :]
                                gap_end = q_len
                            else:    
                                v_to_save = torch.cat([value_states[:, :, :S, :], value_states[:, :, -W:, :]], dim=2)
                                gap_end = q_len - W

                            # k_gap_compressed = key_states[:, :n_heads_keep, S:gap_end, :]
                            # v_gap_compressed = value_states[:, :n_heads_keep, S:gap_end, :]
                            v_gap_compressed = value_states[:, :, S:gap_end, :].transpose(1,2).reshape(bsz, gap_end - S, -1)[:, :, :manager.total_save_dim].clone()
                            
                            if layer_idx not in manager.gap_storage:
                                manager.gap_storage[layer_idx] = {}
                            manager.gap_storage[layer_idx]['v'] = v_gap_compressed
                            # del k_gap_compressed, v_gap_compressed
                            # del key_states, value_states
                            # torch.cuda.empty_cache()
                            
                        else:
                            v_to_save = value_states
                        
                        if not past_key_values.layers[layer_idx].is_initialized:
                            past_key_values.layers[layer_idx].lazy_initialization(v_to_save)
                        past_key_values.layers[layer_idx].values = v_to_save
                        past_key_values.layers[layer_idx].keys = torch.cat([past_key_values.layers[layer_idx].keys, k_recovered])
                        k_recovered = past_key_values.layers[layer_idx].keys
                        
                if manager.is_training and not manager.is_echo_mode:
                    if layer_idx not in manager.training_buffer:
                        manager.training_buffer[layer_idx] = {}
                    

                    manager.training_buffer[layer_idx]['k'] = final_key_states.detach() # Un-rotated
                    manager.training_buffer[layer_idx]['think_k'] = k_recovered.detach() # Un-rotated
                    manager.training_buffer[layer_idx]['v'] = final_value_states.detach()
                    manager.training_buffer[layer_idx]['q'] = query_states.detach() # Un-rotated
                    

                    manager.training_buffer[layer_idx]['cos'] = cos.detach()
                    manager.training_buffer[layer_idx]['sin'] = sin.detach()
                    
                    manager.training_buffer[layer_idx]['mask'] = attention_mask
                    


            elif q_len == 1:
                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    


                    key_states, value_states = past_key_values.update(key_states, value_states, self_attn_module.layer_idx, cache_kwargs)

                    
                    if is_anchor or not manager.is_echo_mode:
                        final_key_states, final_value_states = key_states, value_states
                    else:



                        has_gap = (layer_idx in manager.gap_storage) and \
                              ('v' in manager.gap_storage[layer_idx])
                        
                        if not has_gap:
                            final_key_states = key_states
                            final_value_states = value_states

                        else:

                            v_sink = value_states[:, :, :S, :]
                            
                            v_win_rest = value_states[:, :, S:, :]
                            

                            v_gap_partial = manager.gap_storage[layer_idx]['v']
                            
                            bsz, gap_len, total_dim = v_gap_partial.shape
                            
                            if manager.block_size > 1:
                                anchor_v_full = past_key_values.layers[anchor_idx].values
 
                                anchor_k_mid = None
                                anchor_v_mid = anchor_v_full[:, :, S : S+gap_len, :]
                            elif manager.block_size == 1:
                                anchor_k_mid = None
                                anchor_v_mid = None
                            
                            _, rec_v_mid = mlp_module.get_projected_kv(
                                None, anchor_v_mid, 
                                None, v_gap_partial,
                                self_attn_module
                            )

                            
                            final_key_states = key_states
                            final_value_states = torch.cat([v_sink, rec_v_mid, v_win_rest], dim=2)

                            del v_sink
                            del v_win_rest
                            del rec_v_mid
                            del anchor_k_mid, anchor_v_mid



            # rope
            if query_states.shape[2] == final_key_states.shape[2]:
                query_states, final_key_states = apply_rotary_pos_emb(query_states, final_key_states, cos, sin)
            elif query_states.shape[2] == 1:
                query_states = apply_rotary_pos_emb_single(query_states, cos[:,-1:,:], sin[:,-1:,:])
                final_key_states = apply_rotary_pos_emb_single(final_key_states, cos, sin)

            
            final_value_states = repeat_kv(final_value_states, self_attn_module.num_key_value_groups)

            attention_interface: Callable = eager_attention_forward
            if self_attn_module.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self_attn_module.config._attn_implementation]


            attn_output, attn_weights = attention_interface(
                self_attn_module,
                query_states,
                final_key_states,
                final_value_states,
                attention_mask,
                dropout=0.0 if not self_attn_module.training else self_attn_module.attention_dropout,
                scaling=self_attn_module.scaling,
                sliding_window=getattr(self_attn_module.config, "sliding_window", None),  # main diff with Llama
                **kwargs,
            )

            # if self_attn_module.config._attn_implementation == "flash_attention_2":
                
            #     attn_output = flash_attn_with_kvcache(
            #         q = query_states.transpose(1,2), 
            #         k = final_key_states.transpose(1,2),
            #         v = final_value_states.transpose(1,2),
            #         rotary_cos=cos,
            #         rotary_sin=sin,
            #         cache_seqlens=torch.full((bsz,), 0, dtype=torch.int32, device=device)
            #         softmax_scale=self_attn_module.scaling,
            #         **kwargs
            #     )


            
            del key_states, value_states, final_key_states, final_value_states
            torch.cuda.empty_cache()

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self_attn_module.o_proj(attn_output)


            if manager.is_training and not manager.is_echo_mode:
                manager.training_buffer[layer_idx]['o'] = attn_output.detach()

            return attn_output, attn_weights

        return echo_forward
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The input tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed




from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask

def echo_MistralModel_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
    causal_mask = mask_function(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    # position_embeddings = self.rotary_emb(hidden_states, position_ids)
    
    # print(position_ids)
    # print(cache_position)
    # print(cache_position[-1])
    # print(torch.arange(0, cache_position[-1]+1, device=hidden_states.device).unsqueeze(0))
    # input()

    ## in echo, we use the full position_embeddings
    full_position_embeddings = self.rotary_emb(hidden_states, torch.arange(0, cache_position[-1]+1, device=hidden_states.device).unsqueeze(0))

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            # position_embeddings=position_embeddings,
            position_embeddings=full_position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )

