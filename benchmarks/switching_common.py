from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_snapshot(label: str = "") -> Dict[str, Any]:
    sync_cuda()
    if not torch.cuda.is_available():
        return {"label": label, "allocated_gib": 0.0, "reserved_gib": 0.0, "peak_gib": 0.0}
    gib = float(1024**3)
    return {
        "label": label,
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "peak_gib": torch.cuda.max_memory_allocated() / gib,
    }


def cache_bytes(past_key_values: Any, gap_storage: Optional[Dict[int, Dict[str, torch.Tensor]]] = None) -> Dict[str, int]:
    regular = 0
    if past_key_values is not None and hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if layer.keys is not None:
                regular += layer.keys.numel() * layer.keys.element_size()
            if layer.values is not None:
                regular += layer.values.numel() * layer.values.element_size()

    gap = 0
    for content in (gap_storage or {}).values():
        for value in content.values():
            if isinstance(value, torch.Tensor):
                gap += value.numel() * value.element_size()
    return {"regular_bytes": regular, "gap_bytes": gap, "total_bytes": regular + gap}


@torch.no_grad()
def inverse_rope_cache(model: torch.nn.Module, past_key_values: Any) -> Dict[str, Any]:
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        raise TypeError("inverse_rope_cache requires a transformers DynamicCache")
    if not hasattr(model, "model") or not hasattr(model.model, "rotary_emb"):
        raise TypeError("inverse_rope_cache currently supports Llama models only")

    converted_layers = 0
    total_tokens = 0
    rotary_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in past_key_values.layers:
        keys = layer.keys
        if keys is None or keys.shape[2] == 0:
            continue
        _, _, seq_len, head_dim = keys.shape
        total_tokens = max(total_tokens, seq_len)
        rotary_key = (keys.device, keys.shape[0], seq_len)
        if rotary_key not in rotary_cache:
            position_ids = torch.arange(
                seq_len, device=keys.device, dtype=torch.long
            ).unsqueeze(0).expand(keys.shape[0], -1)


            rotary_input = keys[:, 0, :1, :].float()
            cos, sin = model.model.rotary_emb(rotary_input, position_ids)
            rotary_cache[rotary_key] = (cos[:, None, :, :], sin[:, None, :, :])
        cos, sin = rotary_cache[rotary_key]


        half = head_dim // 2
        first = keys[..., :half].float().clone()
        second = keys[..., half:].float().clone()
        keys[..., :half].copy_(first * cos[..., :half] + second * sin[..., :half])
        keys[..., half:].copy_(second * cos[..., half:] - first * sin[..., half:])
        converted_layers += 1

    return {"converted_layers": converted_layers, "tokens": int(total_tokens)}


def parameter_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _is_anchor(manager: Any, layer_idx: int) -> bool:
    _, anchor_idx = manager.get_anchor_info(layer_idx)
    return manager.block_size != 1 and layer_idx == anchor_idx


@torch.no_grad()
def compact_existing_cache(manager: Any, past_key_values: Any) -> Dict[str, Any]:
    if manager.is_echo_mode:
        raise RuntimeError("compact_existing_cache expects FullKV mode")
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        raise TypeError("EchoKV switching requires a transformers DynamicCache")

    manager.gap_storage = {}
    manager.training_buffer = {}
    total_layers = len(past_key_values.layers)
    converted_layers = 0

    for layer_idx in range(total_layers):
        if _is_anchor(manager, layer_idx):
            continue
        layer = past_key_values.layers[layer_idx]
        keys, values = layer.keys, layer.values
        if keys is None or values is None:
            continue

        batch, _, seq_len, _ = keys.shape
        sink = min(int(manager.sink_size), seq_len)
        window = min(int(manager.local_window_size), max(0, seq_len - sink))
        gap_end = seq_len - window if window else seq_len
        gap_len = max(0, gap_end - sink)
        feature_dim = keys.shape[1] * keys.shape[3]
        save_dim = int(manager.total_save_dim)
        if save_dim < 0 or save_dim > feature_dim:
            raise ValueError(f"total_save_dim={save_dim} is invalid for feature_dim={feature_dim}")

        middle = None
        middle_v = None
        if gap_len:
            if save_dim:
                head_dim = keys.shape[3]
                stored_heads = (save_dim + head_dim - 1) // head_dim
                stored_feature_dim = stored_heads * head_dim
                middle = (
                    keys[:, :stored_heads, sink:gap_end, :]
                    .transpose(1, 2)
                    .reshape(batch, gap_len, stored_feature_dim)
                )
                middle_v = (
                    values[:, :stored_heads, sink:gap_end, :]
                    .transpose(1, 2)
                    .reshape(batch, gap_len, stored_feature_dim)
                )
                stored_k = middle[..., :save_dim].clone()
                stored_v = middle_v[..., :save_dim].clone()
            else:
                stored_k = keys.new_empty((batch, gap_len, 0))
                stored_v = values.new_empty((batch, gap_len, 0))
        else:
            stored_k = keys.new_empty((batch, 0, save_dim))
            stored_v = values.new_empty((batch, 0, save_dim))

        if window:
            kept_k = torch.cat((keys[:, :, :sink, :], keys[:, :, -window:, :]), dim=2).clone()
            kept_v = torch.cat((values[:, :, :sink, :], values[:, :, -window:, :]), dim=2).clone()
        else:
            kept_k = keys[:, :, :sink, :].clone()
            kept_v = values[:, :, :sink, :].clone()

        manager.gap_storage[layer_idx] = {"k": stored_k, "v": stored_v}
        layer.keys = kept_k
        layer.values = kept_v
        converted_layers += 1
        del keys, values, middle, middle_v


    manager.is_echo_mode = True
    return {
        "converted_layers": converted_layers,
        "cache_bytes": cache_bytes(past_key_values, manager.gap_storage),
    }


def reset_manager_full(manager: Any) -> None:
    if manager is None:
        return
    manager.is_echo_mode = False
    manager.gap_storage = {}
    manager.training_buffer = {}
    if hasattr(manager, "prefill_switch_callback"):
        manager.prefill_switch_callback = None
    if hasattr(manager, "prefill_switch_state"):
        manager.prefill_switch_state = None
    if hasattr(manager, "reset_component_profile"):
        manager.reset_component_profile()


def load_runtime(args: Any, mode: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if mode == "palu":


        import importlib
        importlib.import_module(getattr(args, 'palu_model_module', 'palu.model'))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "use_cache": True,
        "attn_implementation": args.attn_implementation,
    }
    if mode == "palu":

        model_kwargs["torch_dtype"] = "auto"
    else:
        model_kwargs["dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    manager = None

    if mode in {"switch", "palu"}:
        if mode == "switch" and model.config.model_type != "llama":
            raise ValueError("The adaptive workload runner currently targets Llama models")
    if mode == "switch":
        from models.llama_patch import EchoLlamaPartialManager

        manager = EchoLlamaPartialManager(model, args)
        manager.load_mlps(args.mlp_path)

        manager.is_echo_mode = False
        manager.to(model.device)

    model.eval()
    return tokenizer, model, manager


def model_forward(model: torch.nn.Module, **kwargs: Any):
    try:
        return model(logits_to_keep=1, return_dict=True, **kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(return_dict=True, **kwargs)


def synthetic_prompt(model: torch.nn.Module, length: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vocab_size = int(model.config.vocab_size)
    ids = torch.randint(2, max(3, vocab_size - 1), (1, length), generator=generator, dtype=torch.long)
    return ids.to(device)


@torch.inference_mode()
def run_greedy_session(
    model: torch.nn.Module,
    manager: Any,
    input_ids: torch.Tensor,
    total_new_tokens: int,
    switch_after: Optional[int],
    stop_on_eos: bool = False,
    progress: Optional[Dict[str, Any]] = None,
    native_fullkv_prefix: bool = False,
    inverse_rope_on_switch: Optional[bool] = None,
    switch_cache_threshold_gib: Optional[float] = None,
    switch_allocated_threshold_gib: Optional[float] = None,
    switch_peak_allocated_threshold_gib: Optional[float] = None,
    prefill_switch_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if total_new_tokens < 1:
        raise ValueError("total_new_tokens must be positive")
    if switch_cache_threshold_gib is not None and switch_cache_threshold_gib <= 0:
        raise ValueError("switch_cache_threshold_gib must be positive")
    if switch_allocated_threshold_gib is not None and switch_allocated_threshold_gib <= 0:
        raise ValueError("switch_allocated_threshold_gib must be positive")
    if (
        switch_peak_allocated_threshold_gib is not None
        and switch_peak_allocated_threshold_gib <= 0
    ):
        raise ValueError("switch_peak_allocated_threshold_gib must be positive")
    if progress is None:
        progress = {}
    if inverse_rope_on_switch is None:
        inverse_rope_on_switch = native_fullkv_prefix
    if native_fullkv_prefix and manager is not None and hasattr(manager, "remove_patch"):
        manager.remove_patch()
    device = input_ids.device
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    sync_cuda()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    progress["phase"] = "prefill"
    prefill_start = time.perf_counter()
    outputs = model_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        cache_position=torch.arange(input_ids.shape[1], device=device),
    )
    sync_cuda()
    prefill_s = time.perf_counter() - prefill_start
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    prefill_memory = memory_snapshot("after_prefill")
    prefill_cache = cache_bytes(past_key_values, getattr(manager, "gap_storage", None))

    generated_by_sample = [[] for _ in range(input_ids.shape[0])]
    first_token_ids = [int(token_id) for token_id in next_token[:, 0].tolist()]
    for sample_idx, token_id in enumerate(first_token_ids):
        generated_by_sample[sample_idx].append(token_id)
    step_times = []
    switched = False
    switch_result: Dict[str, Any] = {}
    switch_latency_s = None
    first_post_switch_s = None
    switch_memory_before = None
    switch_memory_after = None
    switch_peak_memory = None
    switch_cache_before = None
    switch_cache_after = None
    inverse_rope_result: Dict[str, Any] = {}
    inverse_rope_latency_s = None
    patch_install_latency_s = None
    cache_compaction_latency_s = None
    first_post_switch_memory = None
    capture_post_switch_peak = False
    switch_generated_tokens = None
    switch_trigger_reason = None
    switch_trigger_observation: Dict[str, Any] = {}
    prefill_switch_latency_s = None
    prefill_cache_compaction_latency_s = None
    prefill_allocator_cleanup_latency_s = None
    switched_during_prefill = False


    if prefill_switch_state and prefill_switch_state.get("triggered"):
        switched = True
        switched_during_prefill = True
        switch_generated_tokens = 0
        switch_trigger_reason = prefill_switch_state.get("reason")
        switch_trigger_observation = dict(
            prefill_switch_state.get("observation") or {}
        )
        switch_memory_before = prefill_switch_state.get("memory_before")
        switch_memory_after = prefill_switch_state.get("memory_after")
        switch_cache_before = prefill_switch_state.get("cache_before")
        switch_cache_after = prefill_switch_state.get("cache_after")
        switch_result = {
            "cache_bytes": switch_cache_after or prefill_cache,
            "converted_layers": prefill_switch_state.get("converted_layers", 0),
        }
        switch_latency_s = prefill_switch_state.get("latency_s")
        inverse_rope_result = dict(
            prefill_switch_state.get("inverse_rope_result") or {}
        )
        inverse_rope_latency_s = prefill_switch_state.get("inverse_rope_latency_s")
        patch_install_latency_s = prefill_switch_state.get(
            "patch_install_latency_s"
        )
        prefill_switch_latency_s = switch_latency_s
        prefill_cache_compaction_latency_s = prefill_switch_state.get(
            "compaction_latency_s"
        )
        prefill_allocator_cleanup_latency_s = prefill_switch_state.get(
            "allocator_cleanup_latency_s"
        )
        capture_post_switch_peak = True

    eos_ids = model.config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    stopped = bool(
        stop_on_eos and eos_ids and all(token_id in eos_ids for token_id in first_token_ids)
    )
    progress["phase"] = (
        "echo_decode"
        if manager is not None and getattr(manager, "is_echo_mode", False)
        else "fullkv_decode"
    )


    while len(generated_by_sample[0]) < total_new_tokens and not stopped:
        trigger_reason = None
        trigger_observation: Dict[str, Any] = {}
        if manager is not None and not switched and not manager.is_echo_mode:
            generated_count = len(generated_by_sample[0])
            if switch_after is not None and generated_count >= switch_after:
                trigger_reason = "generated_tokens"
                trigger_observation = {
                    "generated_tokens": generated_count,
                    "threshold": switch_after,
                }
            elif switch_cache_threshold_gib is not None:
                current_cache = cache_bytes(past_key_values, None)
                current_cache_gib = current_cache["total_bytes"] / float(1024**3)
                if current_cache_gib >= switch_cache_threshold_gib:
                    trigger_reason = "fullkv_cache_gib"
                    trigger_observation = {
                        "fullkv_cache_gib": current_cache_gib,
                        "threshold_gib": switch_cache_threshold_gib,
                    }
            if trigger_reason is None and switch_allocated_threshold_gib is not None:
                allocated_gib = (
                    torch.cuda.memory_allocated() / float(1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                if allocated_gib >= switch_allocated_threshold_gib:
                    trigger_reason = "allocated_gib"
                    trigger_observation = {
                        "allocated_gib": allocated_gib,
                        "threshold_gib": switch_allocated_threshold_gib,
                    }
            if (
                trigger_reason is None
                and switch_peak_allocated_threshold_gib is not None
            ):
                peak_allocated_gib = (
                    torch.cuda.max_memory_allocated() / float(1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                if peak_allocated_gib >= switch_peak_allocated_threshold_gib:
                    trigger_reason = "peak_allocated_gib"
                    trigger_observation = {
                        "peak_allocated_gib": peak_allocated_gib,
                        "threshold_gib": switch_peak_allocated_threshold_gib,
                    }

        if trigger_reason is not None:
            progress["phase"] = "switch"
            switch_generated_tokens = len(generated_by_sample[0])
            switch_trigger_reason = trigger_reason
            switch_trigger_observation = trigger_observation
            switch_cache_before = cache_bytes(past_key_values, None)
            switch_memory_before = memory_snapshot("before_switch")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            sync_cuda()
            switch_start = time.perf_counter()
            if native_fullkv_prefix:
                if inverse_rope_on_switch:
                    inverse_start = time.perf_counter()
                    inverse_rope_result = inverse_rope_cache(model, past_key_values)
                    sync_cuda()
                    inverse_rope_latency_s = time.perf_counter() - inverse_start
                else:
                    inverse_rope_result = {
                        "skipped": True,
                        "reason": "cache representation is already post-RoPE",
                    }
                if hasattr(manager, "_apply_patch"):
                    patch_start = time.perf_counter()
                    manager._apply_patch()
                    sync_cuda()
                    patch_install_latency_s = time.perf_counter() - patch_start
            compaction_start = time.perf_counter()
            switch_result = compact_existing_cache(manager, past_key_values)
            sync_cuda()
            cache_compaction_latency_s = time.perf_counter() - compaction_start
            switch_latency_s = time.perf_counter() - switch_start
            switch_memory_after = memory_snapshot("after_switch")
            switch_peak_memory = memory_snapshot("switch_peak")
            switched = True
            capture_post_switch_peak = True
            progress["phase"] = "echo_decode"

        if capture_post_switch_peak and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        sync_cuda()
        start = time.perf_counter()
        step_attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)),
            dim=1,
        )
        logical_position = input_ids.shape[1] + len(generated_by_sample[0]) - 1
        outputs = model_forward(
            model,
            input_ids=next_token,
            attention_mask=step_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=torch.tensor([logical_position], dtype=torch.long, device=device),
        )
        sync_cuda()
        elapsed = time.perf_counter() - start
        step_times.append(elapsed)
        past_key_values = outputs.past_key_values
        attention_mask = step_attention_mask
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        token_ids = [int(token_id) for token_id in next_token[:, 0].tolist()]
        for sample_idx, token_id in enumerate(token_ids):
            generated_by_sample[sample_idx].append(token_id)

        if capture_post_switch_peak:
            first_post_switch_memory = memory_snapshot("first_post_switch_attention")
            first_post_switch_s = elapsed
            capture_post_switch_peak = False

        if stop_on_eos and eos_ids and all(token_id in eos_ids for token_id in token_ids):
            stopped = True

    end_memory = memory_snapshot("end")
    progress["phase"] = "completed"
    component_profile = (
        manager.component_profile_summary()
        if manager is not None and hasattr(manager, "component_profile_summary")
        else {}
    )

    if switched:
        switch_step_index = max(0, int(switch_generated_tokens or 1) - 1)
        post_times = step_times[switch_step_index:]


        if post_times:
            if first_post_switch_s is None:
                first_post_switch_s = post_times[0]
            steady_times = post_times[1:]
        else:
            steady_times = []
    else:
        post_times = []
        steady_times = []

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    return {
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": len(generated_by_sample[0]),
        "generated_token_ids": generated_by_sample[0],
        "generated_token_ids_by_sample": generated_by_sample,
        "decode_step_timing_s": [float(value) for value in step_times],
        "token_latency_s": [float(prefill_s), *[float(value) for value in step_times]],
        "prefill_s": float(prefill_s),
        "pre_switch_tpot_s": mean_or_none(
            step_times[: max(0, int(switch_generated_tokens or 1) - 1)]
            if switched
            else step_times
        ),
        "first_post_switch_s": first_post_switch_s,
        "steady_tpot_s": mean_or_none(steady_times),
        "switch_latency_s": switch_latency_s,
        "prefill_switch_latency_s": prefill_switch_latency_s,
        "prefill_cache_compaction_latency_s": prefill_cache_compaction_latency_s,
        "prefill_allocator_cleanup_latency_s": prefill_allocator_cleanup_latency_s,
        "inverse_rope_latency_s": inverse_rope_latency_s,
        "patch_install_latency_s": patch_install_latency_s,
        "cache_compaction_latency_s": cache_compaction_latency_s,
        "inverse_rope_result": inverse_rope_result,
        "native_fullkv_prefix": native_fullkv_prefix,
        "inverse_rope_on_switch": inverse_rope_on_switch,
        "switch_triggered": switched,
        "switch_trigger_reason": switch_trigger_reason,
        "switch_trigger_observation": switch_trigger_observation,
        "switch_cache_threshold_gib": switch_cache_threshold_gib,
        "switch_allocated_threshold_gib": switch_allocated_threshold_gib,
        "switch_peak_allocated_threshold_gib": switch_peak_allocated_threshold_gib,
        "switch_during_prefill": switched_during_prefill,
        "full_generated_tokens_before_switch": switch_generated_tokens if switched else 0,
        "logical_cache_tokens_at_switch": (
            int(input_ids.shape[1])
            if switched_during_prefill
            else int(input_ids.shape[1] + int(switch_generated_tokens or 0) - 1)
            if switched
            else None
        ),
        "session_tokens_at_switch": (
            int(input_ids.shape[1])
            if switched_during_prefill
            else int(input_ids.shape[1] + int(switch_generated_tokens or 0))
            if switched
            else None
        ),
        "pending_token_ids_at_switch": (
            None
            if switched_during_prefill
            else [sample[int(switch_generated_tokens or 1) - 1] for sample in generated_by_sample]
            if switched
            else None
        ),
        "prefill_memory": prefill_memory,
        "before_switch_memory": switch_memory_before,
        "after_switch_memory": switch_memory_after,
        "switch_peak_memory": switch_peak_memory,
        "pre_switch_cache_bytes": switch_cache_before,
        "first_post_switch_memory": first_post_switch_memory,
        "end_memory": end_memory,
        "end_to_end_peak_gib": max(
            m["peak_gib"]
            for m in [
                prefill_memory,
                switch_peak_memory or {"peak_gib": 0.0},
                first_post_switch_memory or {"peak_gib": 0.0},
                end_memory,
            ]
        ),
        "prefill_cache_bytes": prefill_cache,
        "post_switch_cache_bytes": switch_cache_after or switch_result.get("cache_bytes"),
        "final_cache_bytes": cache_bytes(past_key_values, getattr(manager, "gap_storage", None)),
        "converted_layers": switch_result.get("converted_layers", 0),
        "component_profile": component_profile,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
