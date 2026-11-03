#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import traceback
from pathlib import Path

import torch

from .switching_common import (
    cache_bytes,
    load_runtime,
    memory_snapshot,
    parameter_bytes,
    reset_manager_full,
    run_greedy_session,
    set_seed,
    synthetic_prompt,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fullkv", "switch"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--palu_model_module", default="palu.model")
    parser.add_argument("--mlp_path")
    parser.add_argument("--prompt_length", type=int, required=True)
    parser.add_argument("--output_length", type=int, default=256)
    parser.add_argument("--switch_after", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--total_save_dim", type=int, default=0)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--empty_cache_each_layer", action="store_true")
    parser.add_argument("--patched_fullkv_prefix", action="store_true")
    parser.add_argument(
        "--cache_representation",
        choices=["pre_rope"],
        default="pre_rope",
    )
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup_prompt_length", type=int, default=512)
    parser.add_argument("--warmup_output_length", type=int, default=8)
    parser.add_argument(
        "--profile_components",
        action="store_true",
        help="Record CUDA-event timings for Echo reconstruction/assembly/RoPE/attention",
    )
    parser.add_argument("--profile_steps", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "switch" and not args.mlp_path:
        parser.error("--mlp_path is required in switch mode")
    if args.switch_after < 1 or args.switch_after >= args.output_length:
        parser.error("--switch_after must be in [1, output_length)")
    if args.warmup and args.warmup_output_length < 2:
        parser.error("--warmup_output_length must be at least 2")
    return args


def main() -> int:
    args = parse_args()
    args.native_fullkv_prefix = args.mode == "switch" and not args.patched_fullkv_prefix
    args.inverse_rope_on_switch = (
        args.native_fullkv_prefix and args.cache_representation == "pre_rope"
    )
    set_seed(args.seed)
    tokenizer = model = manager = None
    progress = {"phase": "model_load"}
    if args.mode == "switch":
        prefix_tag = "native" if args.native_fullkv_prefix else "patched"
        result_name = f"switch_{prefix_tag}_S{args.switch_after}"
    else:
        result_name = args.mode
    output_path = args.output_dir / f"{result_name}_L{args.prompt_length}_B{args.batch_size}.json"
    payload = {
        "status": "started",
        "mode": args.mode,
        "model_path": args.model_path,
        "mlp_path": args.mlp_path,
        "prompt_length": args.prompt_length,
        "output_length": args.output_length,
        "switch_after": args.switch_after if args.mode == "switch" else None,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "block_size": args.block_size,
        "total_save_dim": args.total_save_dim,
        "sink_size": args.sink_size,
        "local_window_size": args.local_window_size,
        "attn_implementation": args.attn_implementation,
        "empty_cache_each_layer": args.empty_cache_each_layer,
        "native_fullkv_prefix": args.native_fullkv_prefix,
        "cache_representation": (
            args.cache_representation if args.mode == "switch" else None
        ),
        "inverse_rope_on_switch": (
            args.inverse_rope_on_switch if args.mode == "switch" else None
        ),
        "warmup": args.warmup,
        "profile_components": args.profile_components,
        "profile_steps": args.profile_steps,
    }

    try:
        runtime_mode = "switch" if args.mode == "switch" or args.mlp_path else "fullkv"
        tokenizer, model, manager = load_runtime(args, runtime_mode)
        if args.mode == "fullkv" and manager is not None:
            manager.is_echo_mode = False
        if args.mode == "switch":
            payload["runtime_mode"] = (
                "native_fullkv_to_echo"
                if args.native_fullkv_prefix
                else "patched_fullkv_to_echo"
            )
        else:
            payload["runtime_mode"] = "patched_fullkv" if manager is not None else "vanilla_fullkv"
        device = next(model.parameters()).device
        input_ids = synthetic_prompt(model, args.prompt_length, args.seed, device)
        if args.batch_size > 1:
            input_ids = input_ids.expand(args.batch_size, -1).clone()

        payload["model_parameter_bytes"] = parameter_bytes(model)
        payload["predictor_parameter_bytes"] = parameter_bytes(manager.echo_mlps) if manager else 0
        payload["memory_after_load"] = memory_snapshot("after_load")
        if args.warmup:
            warmup_input = synthetic_prompt(
                model, args.warmup_prompt_length, args.seed + 1000, device
            )
            warmup_switch_after = (
                min(4, args.warmup_output_length - 1) if args.mode == "switch" else None
            )
            run_greedy_session(
                model=model,
                manager=manager,
                input_ids=warmup_input,
                total_new_tokens=args.warmup_output_length,
                switch_after=warmup_switch_after,
                stop_on_eos=False,
                native_fullkv_prefix=args.native_fullkv_prefix,
                inverse_rope_on_switch=args.inverse_rope_on_switch,
            )
            reset_manager_full(manager)
            del warmup_input
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            payload["memory_after_warmup"] = memory_snapshot("after_warmup")

        result = run_greedy_session(
            model=model,
            manager=manager,
            input_ids=input_ids,
            total_new_tokens=args.output_length,
            switch_after=args.switch_after if args.mode == "switch" else None,
            stop_on_eos=False,
            progress=progress,
            native_fullkv_prefix=args.native_fullkv_prefix,
            inverse_rope_on_switch=args.inverse_rope_on_switch,
        )
        result["generated_text_first_sample"] = tokenizer.decode(
            result["generated_token_ids"], skip_special_tokens=False
        )
        payload.update(result)
        payload["status"] = "completed"
    except torch.cuda.OutOfMemoryError as exc:
        payload.update({
            "status": "oom",
            "error": str(exc),
            "oom_phase": progress["phase"],
            "memory_at_oom": memory_snapshot("oom"),
        })
        torch.cuda.empty_cache()
    except Exception as exc:
        payload.update({
            "status": "failed",
            "error": repr(exc),
            "failure_phase": progress["phase"],
            "traceback": traceback.format_exc(),
        })

    write_json(output_path, payload)
    print(f"Saved {payload['status']} result to {output_path}")
    return 0 if payload["status"] in {"completed", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
