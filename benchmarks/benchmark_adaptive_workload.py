#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .switching_common import (
    cache_bytes,
    compact_existing_cache,
    inverse_rope_cache,
    load_runtime,
    memory_snapshot,
    parameter_bytes,
    reset_manager_full,
    run_greedy_session,
    set_seed,
    synthetic_prompt,
    sync_cuda,
    write_json,
)


POLICIES = (
    "fullkv_only",
    "echo_only",
    "adaptive_length",
    "adaptive_memory",
    "palu_static",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--palu_model_module", default="palu.model")
    parser.add_argument("--mlp_path")
    parser.add_argument("--prompt_lengths", type=int, nargs="+", required=True)
    parser.add_argument("--requests_per_length", type=int, default=5)
    parser.add_argument("--order_repeats", type=int, default=1)
    parser.add_argument("--output_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--length_threshold", type=int, default=32768)
    parser.add_argument("--cache_threshold_gib", type=float)
    parser.add_argument("--allocated_threshold_gib", type=float)
    parser.add_argument("--peak_allocated_threshold_gib", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--total_save_dim", type=int, default=0)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--empty_cache_each_layer", action="store_true")
    parser.add_argument(
        "--cache_representation",
        choices=["pre_rope"],
        default="pre_rope",
    )
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--warmup_prompt_length", type=int, default=512)
    parser.add_argument("--warmup_output_length", type=int, default=8)
    parser.add_argument("--empty_cache_between_requests", action="store_true")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    if args.requests_per_length < 1 or args.order_repeats < 1:
        parser.error("request counts and order repeats must be positive")
    if args.output_length < 2:
        parser.error("--output_length must be at least 2")
    if args.batch_size < 1:
        parser.error("--batch_size must be positive")
    if any(length < 1 for length in args.prompt_lengths):
        parser.error("all prompt lengths must be positive")
    if args.policy in {"echo_only", "adaptive_length", "adaptive_memory"} and not args.mlp_path:
        parser.error("--mlp_path is required for EchoKV and adaptive policies")
    if args.policy == "adaptive_memory":
        if (
            args.cache_threshold_gib is None
            and args.allocated_threshold_gib is None
            and args.peak_allocated_threshold_gib is None
        ):
            parser.error(
                "adaptive_memory requires --cache_threshold_gib or "
                "--allocated_threshold_gib or --peak_allocated_threshold_gib"
            )
    elif (
        args.cache_threshold_gib is not None
        or args.allocated_threshold_gib is not None
        or args.peak_allocated_threshold_gib is not None
    ):
        parser.error("memory thresholds are only used by adaptive_memory")
    return args


def build_workload(args: argparse.Namespace) -> list[dict[str, int]]:
    base_requests = []
    request_index = 0
    for length_index, prompt_length in enumerate(args.prompt_lengths):
        for sample_index in range(args.requests_per_length):
            base_requests.append(
                {
                    "base_request_index": request_index,
                    "prompt_length": prompt_length,
                    "prompt_seed": args.seed + length_index * 1000 + sample_index,
                }
            )
            request_index += 1

    workload = []
    for repeat_index in range(args.order_repeats):
        order = list(base_requests)
        rng = np.random.default_rng(args.seed + repeat_index)
        rng.shuffle(order)
        for order_index, request in enumerate(order):
            workload.append(
                {
                    **request,
                    "repeat_index": repeat_index,
                    "order_index": order_index,
                }
            )
    return workload


def prepare_route(
    args: argparse.Namespace,
    manager: Any,
    prompt_length: int,
) -> dict[str, Any]:
    if args.policy == "fullkv_only":
        return {
            "planned_route": "FullKV",
            "native_fullkv_prefix": False,
            "prefill_memory_switch": False,
            "switch_cache_threshold_gib": None,
            "switch_allocated_threshold_gib": None,
            "switch_peak_allocated_threshold_gib": None,
        }

    if args.policy == "palu_static":
        return {
            "planned_route": "PaluKV ratio=0.5",
            "native_fullkv_prefix": False,
            "prefill_memory_switch": False,
            "switch_cache_threshold_gib": None,
            "switch_allocated_threshold_gib": None,
            "switch_peak_allocated_threshold_gib": None,
        }

    reset_manager_full(manager)
    if args.policy == "echo_only":
        manager.set_mode(True)
        return {
            "planned_route": "EchoKV from prefill",
            "native_fullkv_prefix": False,
            "prefill_memory_switch": False,
            "switch_cache_threshold_gib": None,
            "switch_allocated_threshold_gib": None,
            "switch_peak_allocated_threshold_gib": None,
        }

    if args.policy == "adaptive_length":
        if prompt_length >= args.length_threshold:
            manager.set_mode(True)
            return {
                "planned_route": "EchoKV from prefill",
                "native_fullkv_prefix": False,
                "prefill_memory_switch": False,
                "switch_cache_threshold_gib": None,
                "switch_allocated_threshold_gib": None,
                "switch_peak_allocated_threshold_gib": None,
            }
        return {
            "planned_route": "FullKV",
            "native_fullkv_prefix": True,
            "prefill_memory_switch": False,
            "switch_cache_threshold_gib": None,
            "switch_allocated_threshold_gib": None,
            "switch_peak_allocated_threshold_gib": None,
        }


    manager.install_prefill_monitor()
    return {
        "planned_route": "FullKV -> EchoKV if memory threshold is reached",
        "native_fullkv_prefix": False,
        "prefill_memory_switch": True,
        "switch_cache_threshold_gib": args.cache_threshold_gib,
        "switch_allocated_threshold_gib": args.allocated_threshold_gib,
        "switch_peak_allocated_threshold_gib": args.peak_allocated_threshold_gib,
    }


def install_prefill_memory_switch(
    args: argparse.Namespace, model: Any, manager: Any
) -> dict[str, Any]:
    state: dict[str, Any] = {"triggered": False}

    def maybe_switch(layer_idx: int, past_key_values: Any) -> None:
        if state["triggered"] or manager.is_echo_mode:
            return
        current_cache = cache_bytes(past_key_values, None)
        current_cache_gib = current_cache["total_bytes"] / float(1024**3)
        allocated_gib = (
            torch.cuda.memory_allocated() / float(1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        reason = None
        if (
            args.cache_threshold_gib is not None
            and current_cache_gib >= args.cache_threshold_gib
        ):
            reason = "prefill_fullkv_cache_gib"
        elif (
            args.allocated_threshold_gib is not None
            and allocated_gib >= args.allocated_threshold_gib
        ):
            reason = "prefill_allocated_gib"
        elif (
            args.peak_allocated_threshold_gib is not None
            and torch.cuda.is_available()
            and torch.cuda.max_memory_allocated() / float(1024**3)
            >= args.peak_allocated_threshold_gib
        ):
            reason = "prefill_peak_allocated_gib"
        if reason is None:
            return


        memory_before = memory_snapshot("prefill_before_switch")
        switch_start = time.perf_counter()
        inverse_result = {}
        inverse_latency_s = None
        if args.cache_representation == "pre_rope":
            inverse_start = time.perf_counter()
            inverse_result = inverse_rope_cache(model, past_key_values)
            sync_cuda()
            inverse_latency_s = time.perf_counter() - inverse_start
        compaction_start = time.perf_counter()
        compaction_result = compact_existing_cache(manager, past_key_values)
        sync_cuda()
        compaction_latency_s = time.perf_counter() - compaction_start
        allocator_cleanup_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sync_cuda()
        allocator_cleanup_latency_s = time.perf_counter() - allocator_cleanup_start
        patch_start = time.perf_counter()
        manager._apply_patch()
        sync_cuda()
        patch_latency_s = time.perf_counter() - patch_start
        switch_latency_s = time.perf_counter() - switch_start
        memory_after = memory_snapshot("prefill_after_switch")
        manager.prefill_switch_callback = None
        state.update(
            {
                "triggered": True,
                "layer_idx": int(layer_idx),
                "reason": reason,
                "observation": {
                    "layer_idx": int(layer_idx),
                    "layer_number": int(layer_idx + 1),
                    "fullkv_cache_gib": current_cache_gib,
                    "allocated_gib": allocated_gib,
                    "peak_allocated_gib": (
                        torch.cuda.max_memory_allocated() / float(1024**3)
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    "cache_threshold_gib": args.cache_threshold_gib,
                    "allocated_threshold_gib": args.allocated_threshold_gib,
                    "peak_allocated_threshold_gib": args.peak_allocated_threshold_gib,
                },
                "cache_before": current_cache,
                "cache_after": compaction_result["cache_bytes"],
                "converted_layers": compaction_result.get("converted_layers", 0),
                "memory_before": memory_before,
                "memory_after": memory_after,
                "latency_s": switch_latency_s,
                "compaction_latency_s": compaction_latency_s,
                "allocator_cleanup_latency_s": allocator_cleanup_latency_s,
                "inverse_rope_result": inverse_result,
                "inverse_rope_latency_s": inverse_latency_s,
                "patch_install_latency_s": patch_latency_s,
            }
        )

    manager.prefill_switch_state = state
    manager.prefill_switch_callback = maybe_switch
    return state


def warm_up(args: argparse.Namespace, model: Any, manager: Any) -> None:
    if args.no_warmup:
        return
    device = next(model.parameters()).device
    warmup_input = synthetic_prompt(
        model, args.warmup_prompt_length, args.seed + 100000, device
    )

    if manager is None:
        run_greedy_session(
            model=model,
            manager=None,
            input_ids=warmup_input,
            total_new_tokens=args.warmup_output_length,
            switch_after=None,
        )
    else:
        reset_manager_full(manager)
        run_greedy_session(
            model=model,
            manager=manager,
            input_ids=warmup_input,
            total_new_tokens=args.warmup_output_length,
            switch_after=None,
            native_fullkv_prefix=True,
            inverse_rope_on_switch=(args.cache_representation == "pre_rope"),
        )
        reset_manager_full(manager)
        manager.set_mode(True)
        run_greedy_session(
            model=model,
            manager=manager,
            input_ids=warmup_input,
            total_new_tokens=args.warmup_output_length,
            switch_after=None,
        )
        reset_manager_full(manager)
        run_greedy_session(
            model=model,
            manager=manager,
            input_ids=warmup_input,
            total_new_tokens=args.warmup_output_length,
            switch_after=min(2, args.warmup_output_length - 1),
            native_fullkv_prefix=True,
            inverse_rope_on_switch=(args.cache_representation == "pre_rope"),
        )
        reset_manager_full(manager)

    del warmup_input
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def strip_token_sequences(result: dict[str, Any]) -> dict[str, Any]:
    omitted = {
        "generated_token_ids",
        "generated_token_ids_by_sample",
        "decode_step_timing_s",
        "token_latency_s",
    }
    return {key: value for key, value in result.items() if key not in omitted}


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def sample_std(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def cache_gib(cache_record: dict[str, int] | None) -> float | None:
    if not cache_record:
        return None
    return cache_record["total_bytes"] / float(1024**3)


def aggregate_records(records: list[dict[str, Any]], batch_size: int) -> dict[str, Any]:
    statuses = Counter(record["status"] for record in records)
    completed = [record for record in records if record["status"] == "completed"]
    request_latencies = [record["request_wall_s"] for record in completed]
    prefill_latencies = [record["result"]["prefill_s"] for record in completed]
    decode_tpots = [record["decode_tpot_s"] for record in completed if record["decode_tpot_s"] is not None]
    switch_latencies = [
        record["result"]["switch_latency_s"]
        for record in completed
        if record["result"]["switch_latency_s"] is not None
    ]
    switch_cache_before = [
        cache_gib(record["result"].get("pre_switch_cache_bytes"))
        for record in completed
        if record["result"].get("pre_switch_cache_bytes")
    ]
    switch_cache_after = [
        cache_gib(record["result"].get("post_switch_cache_bytes"))
        for record in completed
        if record["result"].get("post_switch_cache_bytes")
    ]
    prefill_switch_layers = [
        record["result"].get("switch_trigger_observation", {}).get("layer_number")
        for record in completed
        if record["result"].get("switch_trigger_observation", {}).get("layer_number")
        is not None
    ]
    converted_layers = [
        record["result"].get("converted_layers")
        for record in completed
        if record["result"].get("converted_layers") is not None
        and record["result"].get("switch_triggered")
    ]
    total_generated = sum(
        record["result"]["generated_tokens"] * batch_size for record in completed
    )
    total_prompt = sum(record["prompt_length"] * batch_size for record in completed)
    measured_wall = sum(record["request_wall_s"] for record in records)

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        group_completed = [record for record in group if record["status"] == "completed"]
        group_latencies = [record["request_wall_s"] for record in group_completed]
        group_switched = [
            record
            for record in group_completed
            if record.get("result", {}).get("switch_triggered")
        ]
        return {
            "attempted": len(group),
            "completed": len(group_completed),
            "oom": sum(record["status"] == "oom" for record in group),
            "switched": sum(
                bool(record.get("result", {}).get("switch_triggered"))
                for record in group_completed
            ),
            "mean_request_wall_s": mean(group_latencies),
            "std_request_wall_s": sample_std(group_latencies),
            "mean_prefill_s": mean(
                [record["result"]["prefill_s"] for record in group_completed]
            ),
            "std_prefill_s": sample_std(
                [record["result"]["prefill_s"] for record in group_completed]
            ),
            "mean_decode_tpot_s": mean(
                [
                    record["decode_tpot_s"]
                    for record in group_completed
                    if record["decode_tpot_s"] is not None
                ]
            ),
            "std_decode_tpot_s": sample_std(
                [
                    record["decode_tpot_s"]
                    for record in group_completed
                    if record["decode_tpot_s"] is not None
                ]
            ),
            "max_peak_gib": max(
                (record["result"]["end_to_end_peak_gib"] for record in group_completed),
                default=None,
            ),
            "mean_final_cache_gib": mean(
                [
                    cache_gib(record["result"].get("final_cache_bytes"))
                    for record in group_completed
                    if record["result"].get("final_cache_bytes")
                ]
            ),
            "mean_switch_latency_s": mean(
                [record["result"]["switch_latency_s"] for record in group_switched]
            ),
            "std_switch_latency_s": sample_std(
                [record["result"]["switch_latency_s"] for record in group_switched]
            ),
            "mean_switch_layer": mean(
                [
                    record["result"].get("switch_trigger_observation", {}).get(
                        "layer_number"
                    )
                    for record in group_switched
                    if record["result"].get("switch_trigger_observation", {}).get(
                        "layer_number"
                    )
                    is not None
                ]
            ),
            "mean_pre_switch_cache_gib": mean(
                [
                    cache_gib(record["result"].get("pre_switch_cache_bytes"))
                    for record in group_switched
                    if record["result"].get("pre_switch_cache_bytes")
                ]
            ),
            "mean_post_switch_cache_gib": mean(
                [
                    cache_gib(record["result"].get("post_switch_cache_bytes"))
                    for record in group_switched
                    if record["result"].get("post_switch_cache_bytes")
                ]
            ),
        }

    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_length[record["prompt_length"]].append(record)
        by_route[record["actual_route"]].append(record)

    return {
        "attempted_requests": len(records),
        "status_counts": dict(statuses),
        "completed_request_rate": len(completed) / len(records) if records else 0.0,
        "switch_count": sum(
            bool(record["result"].get("switch_triggered")) for record in completed
        ),
        "route_counts": dict(Counter(record["actual_route"] for record in records)),
        "total_prompt_tokens_completed": total_prompt,
        "total_generated_tokens_completed": total_generated,
        "measured_request_wall_s_all_attempts": measured_wall,
        "generated_tokens_per_s": total_generated / measured_wall if measured_wall else None,
        "all_tokens_per_s": (total_prompt + total_generated) / measured_wall
        if measured_wall
        else None,
        "mean_request_wall_s": mean(request_latencies),
        "std_request_wall_s": sample_std(request_latencies),
        "p50_request_wall_s": percentile(request_latencies, 50),
        "p95_request_wall_s": percentile(request_latencies, 95),
        "mean_prefill_s": mean(prefill_latencies),
        "std_prefill_s": sample_std(prefill_latencies),
        "mean_decode_tpot_s": mean(decode_tpots),
        "std_decode_tpot_s": sample_std(decode_tpots),
        "mean_switch_latency_s": mean(switch_latencies),
        "std_switch_latency_s": sample_std(switch_latencies),
        "mean_prefill_switch_layer": mean(prefill_switch_layers),
        "mean_converted_layers_at_switch": mean(converted_layers),
        "mean_pre_switch_cache_gib": mean(switch_cache_before),
        "mean_post_switch_cache_gib": mean(switch_cache_after),
        "mean_post_over_pre_switch_cache_ratio": mean(
            [
                after / before
                for before, after in zip(switch_cache_before, switch_cache_after)
                if before
            ]
        ),
        "max_end_to_end_peak_gib": max(
            (record["result"]["end_to_end_peak_gib"] for record in completed),
            default=None,
        ),
        "by_prompt_length": {
            str(length): summarize_group(group) for length, group in sorted(by_length.items())
        },
        "by_actual_route": {
            route: summarize_group(group) for route, group in sorted(by_route.items())
        },
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "request_id",
        "repeat_index",
        "order_index",
        "prompt_length",
        "prompt_seed",
        "policy",
        "planned_route",
        "actual_route",
        "status",
        "oom_phase",
        "generated_tokens",
        "switch_triggered",
        "switch_trigger_reason",
        "switch_during_prefill",
        "prefill_switch_layer",
        "prefill_switch_peak_allocated_gib",
        "full_generated_tokens_before_switch",
        "prefill_s",
        "decode_tpot_s",
        "switch_latency_s",
        "inverse_rope_latency_s",
        "prefill_cache_compaction_latency_s",
        "prefill_allocator_cleanup_latency_s",
        "patch_install_latency_s",
        "converted_layers",
        "request_wall_s",
        "end_to_end_peak_gib",
        "prefill_cache_gib",
        "pre_switch_cache_gib",
        "post_switch_cache_gib",
        "final_cache_gib",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            result = record.get("result", {})
            writer.writerow(
                {
                    "request_id": record["request_id"],
                    "repeat_index": record["repeat_index"],
                    "order_index": record["order_index"],
                    "prompt_length": record["prompt_length"],
                    "prompt_seed": record["prompt_seed"],
                    "policy": record["policy"],
                    "planned_route": record["planned_route"],
                    "actual_route": record["actual_route"],
                    "status": record["status"],
                    "oom_phase": record.get("oom_phase"),
                    "generated_tokens": result.get("generated_tokens"),
                    "switch_triggered": result.get("switch_triggered"),
                    "switch_trigger_reason": result.get("switch_trigger_reason"),
                    "switch_during_prefill": result.get("switch_during_prefill"),
                    "prefill_switch_layer": (
                        result.get("switch_trigger_observation", {}).get("layer_number")
                    ),
                    "prefill_switch_peak_allocated_gib": result.get(
                        "switch_trigger_observation", {}
                    ).get("peak_allocated_gib"),
                    "full_generated_tokens_before_switch": result.get(
                        "full_generated_tokens_before_switch"
                    ),
                    "prefill_s": result.get("prefill_s"),
                    "decode_tpot_s": record.get("decode_tpot_s"),
                    "switch_latency_s": result.get("switch_latency_s"),
                    "inverse_rope_latency_s": result.get("inverse_rope_latency_s"),
                    "prefill_cache_compaction_latency_s": result.get(
                        "prefill_cache_compaction_latency_s"
                    ),
                    "prefill_allocator_cleanup_latency_s": result.get(
                        "prefill_allocator_cleanup_latency_s"
                    ),
                    "patch_install_latency_s": result.get(
                        "patch_install_latency_s"
                    ),
                    "converted_layers": result.get("converted_layers"),
                    "request_wall_s": record.get("request_wall_s"),
                    "end_to_end_peak_gib": result.get("end_to_end_peak_gib"),
                    "prefill_cache_gib": cache_gib(result.get("prefill_cache_bytes")),
                    "pre_switch_cache_gib": cache_gib(
                        result.get("pre_switch_cache_bytes")
                    ),
                    "post_switch_cache_gib": cache_gib(result.get("post_switch_cache_bytes")),
                    "final_cache_gib": cache_gib(result.get("final_cache_bytes")),
                }
            )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workload = build_workload(args)
    if args.policy == "fullkv_only":
        runtime_mode = "fullkv"
    elif args.policy == "palu_static":
        runtime_mode = "palu"
    else:
        runtime_mode = "switch"
    tokenizer = None
    model = None
    manager = None
    records: list[dict[str, Any]] = []
    benchmark_started = None

    config = {
        "policy": args.policy,
        "model_path": args.model_path,
        "mlp_path": args.mlp_path,
        "prompt_lengths": args.prompt_lengths,
        "requests_per_length": args.requests_per_length,
        "order_repeats": args.order_repeats,
        "output_length": args.output_length,
        "batch_size": args.batch_size,
        "length_threshold": args.length_threshold,
        "cache_threshold_gib": args.cache_threshold_gib,
        "allocated_threshold_gib": args.allocated_threshold_gib,
        "peak_allocated_threshold_gib": args.peak_allocated_threshold_gib,
        "cache_representation": args.cache_representation,
        "total_save_dim": args.total_save_dim,
        "seed": args.seed,
        "warmup": not args.no_warmup,
        "empty_cache_between_requests": args.empty_cache_between_requests,
    }

    try:
        tokenizer, model, manager = load_runtime(args, runtime_mode)
        del tokenizer
        device = next(model.parameters()).device
        config["model_parameter_bytes"] = parameter_bytes(model)
        config["predictor_parameter_bytes"] = (
            parameter_bytes(manager.echo_mlps) if manager is not None else 0
        )
        config["memory_after_load"] = memory_snapshot("after_load")
        warm_up(args, model, manager)
        config["memory_after_warmup"] = memory_snapshot("after_warmup")
        benchmark_started = time.perf_counter()

        for global_index, request in enumerate(workload):
            request_id = (
                f"r{request['repeat_index']}_l{request['prompt_length']}_"
                f"n{request['base_request_index']}"
            )
            route = prepare_route(args, manager, request["prompt_length"])
            prefill_switch_state = None
            if route.get("prefill_memory_switch"):
                prefill_switch_state = install_prefill_memory_switch(
                    args, model, manager
                )
            progress = {"phase": "input_allocation"}
            record: dict[str, Any] = {
                **request,
                "request_id": request_id,
                "global_index": global_index,
                "policy": args.policy,
                "planned_route": route["planned_route"],
                "actual_route": route["planned_route"],
                "status": "started",
            }
            input_ids = None
            request_started = None
            request_setup_started = time.perf_counter()

            try:
                input_ids = synthetic_prompt(
                    model, request["prompt_length"], request["prompt_seed"], device
                )
                if args.batch_size > 1:
                    input_ids = input_ids.expand(args.batch_size, -1).clone()
                request_started = time.perf_counter()
                result = run_greedy_session(
                    model=model,
                    manager=manager,
                    input_ids=input_ids,
                    total_new_tokens=args.output_length,
                    switch_after=None,
                    stop_on_eos=False,
                    progress=progress,
                    native_fullkv_prefix=route["native_fullkv_prefix"],
                    inverse_rope_on_switch=(args.cache_representation == "pre_rope"),
                    switch_cache_threshold_gib=route["switch_cache_threshold_gib"],
                    switch_allocated_threshold_gib=route[
                        "switch_allocated_threshold_gib"
                    ],
                    switch_peak_allocated_threshold_gib=route[
                        "switch_peak_allocated_threshold_gib"
                    ],
                    prefill_switch_state=prefill_switch_state,
                )
                sync_cuda()
                record["request_wall_s"] = time.perf_counter() - request_started
                record["status"] = "completed"
                if result["switch_triggered"]:
                    record["actual_route"] = "FullKV -> EchoKV"
                elif args.policy == "adaptive_memory":
                    record["actual_route"] = "FullKV (threshold not reached)"
                record["decode_tpot_s"] = (
                    float(np.mean(result["decode_step_timing_s"]))
                    if result["decode_step_timing_s"]
                    else None
                )
                record["result"] = strip_token_sequences(result)
            except torch.cuda.OutOfMemoryError as exc:
                sync_cuda()
                record.update(
                    {
                        "status": "oom",
                        "actual_route": f"{route['planned_route']} (OOM)",
                        "error": str(exc),
                        "oom_phase": progress["phase"],
                        "request_wall_s": time.perf_counter()
                        - (request_started or request_setup_started),
                        "memory_at_oom": memory_snapshot("oom"),
                        "result": {},
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                record.update(
                    {
                        "status": "failed",
                        "actual_route": f"{route['planned_route']} (failed)",
                        "error": repr(exc),
                        "failure_phase": progress["phase"],
                        "request_wall_s": time.perf_counter()
                        - (request_started or request_setup_started),
                        "traceback": traceback.format_exc(),
                        "result": {},
                    }
                )

            records.append(record)
            del input_ids
            reset_manager_full(manager)
            gc.collect()
            if args.empty_cache_between_requests and torch.cuda.is_available():
                torch.cuda.empty_cache()

            write_json(
                args.output_dir / "workload.json",
                {"status": "running", "config": config, "requests": records},
            )
            print(
                f"[{global_index + 1}/{len(workload)}] {request_id}: "
                f"{record['status']} via {record['actual_route']}"
            )
    except Exception as exc:
        write_json(
            args.output_dir / "workload.json",
            {
                "status": "failed_during_setup",
                "config": config,
                "requests": records,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise

    summary = aggregate_records(records, args.batch_size)
    summary["benchmark_wall_s_excluding_model_load_and_warmup"] = (
        time.perf_counter() - benchmark_started if benchmark_started is not None else None
    )
    write_json(args.output_dir / "summary.json", {"config": config, "summary": summary})
    write_json(
        args.output_dir / "workload.json",
        {"status": "completed", "config": config, "summary": summary, "requests": records},
    )
    write_csv(args.output_dir / "requests.csv", records)
    print(f"Saved workload results under {args.output_dir}")
    return 0 if summary["status_counts"].get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
