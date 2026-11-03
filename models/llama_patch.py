import torch
import torch.nn as nn
from types import MethodType
from .echo_mlp import EchoPartialMLP
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from typing import Callable, Optional, Union
import os
import transformers

from flash_attn import flash_attn_with_kvcache

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from typing import Tuple

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
        elif isinstance(past_key_values, (tuple, list)):
            for k, v in past_key_values:
                kv_bytes += k.numel() * k.element_size()
                kv_bytes += v.numel() * v.element_size()

    if gap_storage:
        for layer_idx, content in gap_storage.items():
            if 'k' in content and isinstance(content['k'], torch.Tensor):
                gap_bytes += content['k'].numel() * content['k'].element_size()
            if 'v' in content and isinstance(content['v'], torch.Tensor):
                gap_bytes += content['v'].numel() * content['v'].element_size()

    return kv_bytes / 1024**3, gap_bytes / 1024**3

class EchoLlamaPartialManager(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.block_size = args.block_size
        self.is_echo_mode = False
        self.sink_size = getattr(args, 'sink_size', 4)
        self.local_window_size = getattr(args, 'local_window_size', 128)
        self.empty_cache_each_layer = getattr(args, 'empty_cache_each_layer', True)
        self.profile_components = getattr(args, 'profile_components', False)
        self.profile_steps = max(0, int(getattr(args, 'profile_steps', 8)))
        self._profile_events = {"reconstruction": [], "assembly": [], "rope": [], "attention": []}
        self._profile_decode_steps = 0
        self._profile_step_active = False

        self.total_save_dim = getattr(args, 'total_save_dim', 1024)
        print(
            f"EchoPartial Mode: block_size={self.block_size}, "
            f"total_save_dim={self.total_save_dim}, "
            f"empty_cache_each_layer={self.empty_cache_each_layer}"
        )

        self.training_buffer = {}

        self.echo_mlps = nn.ModuleList([
            EchoPartialMLP(
                model.config,
                total_save_dim=self.total_save_dim,
                block_size=self.block_size,
                input_mode=getattr(args, 'echo_input_mode', 'joint'),
            )
            if self._layer_needs_predictor(layer_idx)
            else nn.Identity()
            for layer_idx in range(model.config.num_hidden_layers)
        ]).to(model.dtype)

        self.gap_storage = {}

        self.is_training = False
        self.original_attn_forwards = {}
        self._patch_active = False
        self._prefill_monitor_active = False

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
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace(".linear_k.", ".project_k.")
            new_key = new_key.replace(".linear_v.", ".project_v.")

            layer_prefix = new_key.split(".", 1)[0]
            if layer_prefix.isdigit() and not self._layer_needs_predictor(int(layer_prefix)):
                continue

            new_state_dict[new_key] = value

        self.echo_mlps.load_state_dict(new_state_dict)

    def _layer_needs_predictor(self, layer_idx):
        if self.block_size == 1:
            return True

        num_layers = self.model.config.num_hidden_layers
        remainder = num_layers % self.block_size
        if layer_idx < remainder:
            return False
        return (layer_idx - remainder) % self.block_size != 0

    def set_mode(self, echo_active: bool):
        self.is_echo_mode = echo_active
        self.training_buffer = {}
        self.gap_storage = {}

        if self.is_echo_mode:
            self._apply_patch()

    def _apply_patch(self):
        if self._patch_active and not self._prefill_monitor_active:
            return
        for i, layer in enumerate(self.model.model.layers):
            original_attn = layer.self_attn
            if i not in self.original_attn_forwards:
                self.original_attn_forwards[i] = original_attn.forward

            layer.self_attn.forward = MethodType(
                self._create_echo_forward(i, self.echo_mlps[i]),
                original_attn
            )

        if not hasattr(self, 'original_llama_forward'):
            self.original_llama_forward = transformers.models.llama.modeling_llama.LlamaModel.forward

        transformers.models.llama.modeling_llama.LlamaModel.forward = echo_LlamaModel_forward
        self._patch_active = True
        self._prefill_monitor_active = False

    def install_prefill_monitor(self):
        self.remove_patch()
        for i, layer in enumerate(self.model.model.layers):
            original_forward = self.original_attn_forwards[i]
            echo_forward = self._create_echo_forward(i, self.echo_mlps[i])

            def monitored_forward(
                self_attn_module,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                _layer_idx=i,
                _original_forward=original_forward,
                _echo_forward=echo_forward,
                **kwargs,
            ):
                if hidden_states.shape[1] > 1:
                    callback = getattr(self, "prefill_switch_callback", None)
                    if callback is not None:
                        callback(_layer_idx, past_key_values)
                if self.is_echo_mode:
                    return _echo_forward(
                        self_attn_module,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values,
                        cache_position,
                        **kwargs,
                    )
                return _original_forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            layer.self_attn.forward = MethodType(monitored_forward, layer.self_attn)

        self._patch_active = True
        self._prefill_monitor_active = True

    def remove_patch(self):
        if not self._patch_active:
            return
        for i, layer in enumerate(self.model.model.layers):
            if i in self.original_attn_forwards:
                layer.self_attn.forward = self.original_attn_forwards[i]
        if hasattr(self, "original_llama_forward"):
            transformers.models.llama.modeling_llama.LlamaModel.forward = self.original_llama_forward
        self._patch_active = False
        self._prefill_monitor_active = False

    def reset_component_profile(self):
        self._profile_events = {"reconstruction": [], "assembly": [], "rope": [], "attention": []}
        self._profile_decode_steps = 0
        self._profile_step_active = False

    def _profile_start_step(self, layer_idx):
        if not self.profile_components or not torch.cuda.is_available():
            return False
        if layer_idx == 0:
            self._profile_step_active = self._profile_decode_steps < self.profile_steps
            if self._profile_step_active:
                self._profile_decode_steps += 1
        return self._profile_step_active

    def _profile_start(self, name, enabled):
        if not enabled:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _profile_end(self, name, start):
        if start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._profile_events[name].append((start, end))

    def component_profile_summary(self):
        if not self.profile_components or not torch.cuda.is_available():
            return {}
        torch.cuda.synchronize()
        summary = {"profiled_decode_steps": self._profile_decode_steps}
        for name, events in self._profile_events.items():
            values = [float(start.elapsed_time(end)) for start, end in events]
            total_ms = sum(values)
            summary[name] = {
                "event_count": len(values),
                "total_ms": total_ms,
                "mean_ms_per_event": total_ms / len(values) if values else 0.0,
                "mean_ms_per_profiled_token": total_ms / self._profile_decode_steps
                if self._profile_decode_steps else 0.0,
            }
        return summary

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
            prefill_switch_callback = getattr(manager, "prefill_switch_callback", None)
            if q_len > 1 and prefill_switch_callback is not None:
                prefill_switch_callback(layer_idx, past_key_values)
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

            final_key_states = key_states
            final_value_states = value_states

            S = manager.sink_size
            W = manager.local_window_size

            n_heads_keep = manager.total_save_dim // self_attn_module.head_dim

            if q_len > 1:

                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

                    if is_anchor or not manager.is_echo_mode:
                        key_states, value_states = past_key_values.update(key_states, value_states, self_attn_module.layer_idx, cache_kwargs)
                        final_key_states, final_value_states = key_states, value_states

                    else :
                        if q_len > (S + W):
                            if W == 0:
                                k_to_save = key_states[:, :, :S, :]
                                v_to_save = value_states[:, :, :S, :]
                                gap_end = q_len
                            else:
                                k_to_save = torch.cat([key_states[:, :, :S, :], key_states[:, :, -W:, :]], dim=2)
                                v_to_save = torch.cat([value_states[:, :, :S, :], value_states[:, :, -W:, :]], dim=2)
                                gap_end = q_len - W

                            k_gap_compressed = mlp_module.select_local_dims(key_states[:, :, S:gap_end, :].transpose(1,2).reshape(bsz, gap_end - S, -1)).clone()
                            v_gap_compressed = mlp_module.select_local_dims(value_states[:, :, S:gap_end, :].transpose(1,2).reshape(bsz, gap_end - S, -1)).clone()

                            if layer_idx not in manager.gap_storage:
                                manager.gap_storage[layer_idx] = {}
                            manager.gap_storage[layer_idx]['k'] = k_gap_compressed
                            manager.gap_storage[layer_idx]['v'] = v_gap_compressed
                            del k_gap_compressed, v_gap_compressed
                            if manager.empty_cache_each_layer:
                                torch.cuda.empty_cache()

                        else:
                            k_to_save, v_to_save = key_states, value_states

                        if not past_key_values.layers[layer_idx].is_initialized:
                            past_key_values.layers[layer_idx].lazy_initialization(k_to_save)
                        past_key_values.layers[layer_idx].keys = k_to_save
                        past_key_values.layers[layer_idx].values = v_to_save

                if manager.is_training and not manager.is_echo_mode:
                    if layer_idx not in manager.training_buffer:
                        manager.training_buffer[layer_idx] = {}

                    manager.training_buffer[layer_idx]['k'] = final_key_states.detach()
                    manager.training_buffer[layer_idx]['v'] = final_value_states.detach()
                    manager.training_buffer[layer_idx]['q'] = query_states.detach()

                    manager.training_buffer[layer_idx]['cos'] = cos.detach()
                    manager.training_buffer[layer_idx]['sin'] = sin.detach()

                    manager.training_buffer[layer_idx]['mask'] = attention_mask

            elif q_len == 1:
                profile_this_step = manager._profile_start_step(layer_idx)
                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

                    key_states, value_states = past_key_values.update(key_states, value_states, self_attn_module.layer_idx, cache_kwargs)

                    if is_anchor or not manager.is_echo_mode:
                        final_key_states, final_value_states = key_states, value_states
                    else:

                        has_gap = (layer_idx in manager.gap_storage) and \
                              ('k' in manager.gap_storage[layer_idx]) and \
                              (manager.gap_storage[layer_idx]['k'].shape[1] > 0)

                        if not has_gap:
                            final_key_states = key_states
                            final_value_states = value_states

                        else:

                            k_sink = key_states[:, :, :S, :]
                            v_sink = value_states[:, :, :S, :]

                            k_win_rest = key_states[:, :, S:, :]
                            v_win_rest = value_states[:, :, S:, :]

                            k_gap_partial = manager.gap_storage[layer_idx]['k']
                            v_gap_partial = manager.gap_storage[layer_idx]['v']

                            bsz, gap_len, total_dim = k_gap_partial.shape

                            if manager.block_size > 1:
                                anchor_k_full = past_key_values.layers[anchor_idx].keys
                                anchor_v_full = past_key_values.layers[anchor_idx].values

                                anchor_k_mid = anchor_k_full[:, :, S : S+gap_len, :]
                                anchor_v_mid = anchor_v_full[:, :, S : S+gap_len, :]
                            elif manager.block_size == 1:
                                anchor_k_mid = None
                                anchor_v_mid = None

                            reconstruction_start = manager._profile_start(
                                "reconstruction", profile_this_step
                            )
                            rec_k_mid, rec_v_mid = mlp_module.get_projected_kv(
                                anchor_k_mid, anchor_v_mid,
                                k_gap_partial, v_gap_partial,
                                self_attn_module
                            )
                            manager._profile_end("reconstruction", reconstruction_start)

                            assembly_start = manager._profile_start("assembly", profile_this_step)
                            final_key_states = torch.cat([k_sink, rec_k_mid, k_win_rest], dim=2)
                            final_value_states = torch.cat([v_sink, rec_v_mid, v_win_rest], dim=2)
                            manager._profile_end("assembly", assembly_start)

                            del k_sink, v_sink
                            del k_win_rest, v_win_rest
                            del rec_k_mid, rec_v_mid
                            del anchor_k_mid, anchor_v_mid

            rope_start = manager._profile_start("rope", profile_this_step if q_len == 1 else False)
            if query_states.shape[2] == final_key_states.shape[2]:
                query_states, final_key_states = apply_rotary_pos_emb(query_states, final_key_states, cos, sin)
            elif query_states.shape[2] == 1:
                query_states = apply_rotary_pos_emb_single(query_states, cos[:,-1:,:], sin[:,-1:,:])
                final_key_states = apply_rotary_pos_emb_single(final_key_states, cos, sin)
            manager._profile_end("rope", rope_start)

            attention_interface: Callable = eager_attention_forward
            if self_attn_module.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self_attn_module.config._attn_implementation]

            attention_start = manager._profile_start("attention", profile_this_step if q_len == 1 else False)
            attn_output, attn_weights = attention_interface(
                self_attn_module,
                query_states,
                final_key_states,
                final_value_states,
                attention_mask,
                dropout=0.0 if not self_attn_module.training else self_attn_module.attention_dropout,
                scaling=self_attn_module.scaling,
                **kwargs,
            )
            manager._profile_end("attention", attention_start)

            del key_states, value_states, final_key_states, final_value_states, cos, sin
            if manager.empty_cache_each_layer:
                torch.cuda.empty_cache()

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self_attn_module.o_proj(attn_output)

            if manager.is_training and not manager.is_echo_mode:
                manager.training_buffer[layer_idx]['o'] = attn_output.detach()

            return attn_output, attn_weights

        return echo_forward

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask

def echo_LlamaModel_forward(
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

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds

    full_position_embeddings = self.rotary_emb(hidden_states, torch.arange(0, cache_position[-1]+1, device=hidden_states.device).unsqueeze(0))

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=full_position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )
