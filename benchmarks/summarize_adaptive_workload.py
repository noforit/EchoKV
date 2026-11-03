#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_POLICIES = (
    "fullkv_only",
    "echo_only",
    "adaptive_length",
    "adaptive_memory",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES))
    return parser.parse_args()


def load_result(root: Path, policy: str) -> dict[str, Any]:
    path = root / policy / "summary.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_pm(mean: Any, std: Any, digits: int = 2) -> str:
    if mean is None:
        return "NA"
    return f"{fmt(mean, digits)} +/- {fmt(std, digits)}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    overall_rows = []
    length_rows = []

    for policy in args.policies:
        payload = load_result(args.results_root, policy)
        summary = payload["summary"]
        config = payload["config"]
        overall_rows.append(
            {
                "policy": policy,
                "completed": summary["status_counts"].get("completed", 0),
                "attempted": summary["attempted_requests"],
                "oom": summary["status_counts"].get("oom", 0),
                "switched": summary["switch_count"],
                "generated_tokens_per_s": summary["generated_tokens_per_s"],
                "all_tokens_per_s": summary["all_tokens_per_s"],
                "mean_request_wall_s": summary["mean_request_wall_s"],
                "std_request_wall_s": summary["std_request_wall_s"],
                "p95_request_wall_s": summary["p95_request_wall_s"],
                "mean_prefill_s": summary["mean_prefill_s"],
                "std_prefill_s": summary["std_prefill_s"],
                "mean_decode_tpot_ms": (
                    summary["mean_decode_tpot_s"] * 1000
                    if summary["mean_decode_tpot_s"] is not None
                    else None
                ),
                "std_decode_tpot_ms": (
                    summary["std_decode_tpot_s"] * 1000
                    if summary["std_decode_tpot_s"] is not None
                    else None
                ),
                "mean_switch_latency_ms": (
                    summary["mean_switch_latency_s"] * 1000
                    if summary["mean_switch_latency_s"] is not None
                    else None
                ),
                "std_switch_latency_ms": (
                    summary["std_switch_latency_s"] * 1000
                    if summary["std_switch_latency_s"] is not None
                    else None
                ),
                "mean_pre_switch_cache_gib": summary["mean_pre_switch_cache_gib"],
                "mean_post_switch_cache_gib": summary["mean_post_switch_cache_gib"],
                "mean_prefill_switch_layer": summary["mean_prefill_switch_layer"],
                "mean_converted_layers_at_switch": summary[
                    "mean_converted_layers_at_switch"
                ],
                "max_peak_gib": summary["max_end_to_end_peak_gib"],
                "predictor_gib": config["predictor_parameter_bytes"] / float(1024**3),
            }
        )

        for prompt_length, group in summary["by_prompt_length"].items():
            length_rows.append(
                {
                    "policy": policy,
                    "prompt_length": int(prompt_length),
                    "completed": group["completed"],
                    "attempted": group["attempted"],
                    "oom": group["oom"],
                    "switched": group["switched"],
                    "mean_request_wall_s": group["mean_request_wall_s"],
                    "std_request_wall_s": group["std_request_wall_s"],
                    "mean_prefill_s": group["mean_prefill_s"],
                    "std_prefill_s": group["std_prefill_s"],
                    "mean_decode_tpot_ms": (
                        group["mean_decode_tpot_s"] * 1000
                        if group["mean_decode_tpot_s"] is not None
                        else None
                    ),
                    "std_decode_tpot_ms": (
                        group["std_decode_tpot_s"] * 1000
                        if group["std_decode_tpot_s"] is not None
                        else None
                    ),
                    "max_peak_gib": group["max_peak_gib"],
                    "mean_final_cache_gib": group["mean_final_cache_gib"],
                    "mean_switch_latency_ms": (
                        group["mean_switch_latency_s"] * 1000
                        if group["mean_switch_latency_s"] is not None
                        else None
                    ),
                    "std_switch_latency_ms": (
                        group["std_switch_latency_s"] * 1000
                        if group["std_switch_latency_s"] is not None
                        else None
                    ),
                    "mean_switch_layer": group["mean_switch_layer"],
                    "mean_pre_switch_cache_gib": group[
                        "mean_pre_switch_cache_gib"
                    ],
                    "mean_post_switch_cache_gib": group[
                        "mean_post_switch_cache_gib"
                    ],
                }
            )

    write_csv(args.results_root / "policy_summary.csv", overall_rows)
    write_csv(args.results_root / "policy_by_length.csv", length_rows)

    overall_markdown = markdown_table(
        [
            "Policy",
            "Completed",
            "OOM",
            "Switched",
            "Workload goodput (tok/s)",
            "Latency (s)",
            "TTFT (s)",
            "TPOT (ms)",
            "Switch (ms)",
            "Switch layer",
            "Switch cache (GiB)",
            "Peak (GiB)",
        ],
        [
            [
                row["policy"],
                f"{row['completed']}/{row['attempted']}",
                str(row["oom"]),
                str(row["switched"]),
                fmt(row["generated_tokens_per_s"]),
                fmt_pm(row["mean_request_wall_s"], row["std_request_wall_s"]),
                fmt_pm(row["mean_prefill_s"], row["std_prefill_s"]),
                fmt_pm(row["mean_decode_tpot_ms"], row["std_decode_tpot_ms"]),
                fmt_pm(row["mean_switch_latency_ms"], row["std_switch_latency_ms"]),
                fmt(row["mean_prefill_switch_layer"], 1),
                (
                    f"{fmt(row['mean_pre_switch_cache_gib'], 3)} -> "
                    f"{fmt(row['mean_post_switch_cache_gib'], 3)}"
                    if row["mean_pre_switch_cache_gib"] is not None
                    else "NA"
                ),
                fmt(row["max_peak_gib"], 3),
            ]
            for row in overall_rows
        ],
    )
    length_markdown = markdown_table(
        [
            "Policy",
            "Prompt",
            "Completed",
            "OOM",
            "Switched",
            "Latency (s)",
            "TTFT (s)",
            "TPOT (ms)",
            "Switch (ms)",
            "Switch layer",
            "Switch cache (GiB)",
            "Final cache (GiB)",
            "Peak (GiB)",
        ],
        [
            [
                row["policy"],
                f"{row['prompt_length'] // 1024}K",
                f"{row['completed']}/{row['attempted']}",
                str(row["oom"]),
                str(row["switched"]),
                fmt_pm(row["mean_request_wall_s"], row["std_request_wall_s"]),
                fmt_pm(row["mean_prefill_s"], row["std_prefill_s"]),
                fmt_pm(row["mean_decode_tpot_ms"], row["std_decode_tpot_ms"]),
                fmt_pm(
                    row["mean_switch_latency_ms"],
                    row["std_switch_latency_ms"],
                ),
                fmt(row["mean_switch_layer"], 1),
                (
                    f"{fmt(row['mean_pre_switch_cache_gib'], 3)} -> "
                    f"{fmt(row['mean_post_switch_cache_gib'], 3)}"
                    if row["mean_pre_switch_cache_gib"] is not None
                    else "NA"
                ),
                fmt(row["mean_final_cache_gib"], 3),
                fmt(row["max_peak_gib"], 3),
            ]
            for row in length_rows
        ],
    )
    report = "# Adaptive workload summary\n\n" + overall_markdown
    report += "\n\n## By prompt length\n\n" + length_markdown + "\n"
    if "palu_static" in args.policies:
        report += (
            "\n> `palu_static` uses the repository's official Palu HuggingFace "
            "loader. Its low-rank projections are reconstructed before the "
            "standard attention cache update, so this row is a static Palu "
            "throughput/capacity reference rather than a memory-matched "
            "low-rank-cache measurement.\n"
        )
    (args.results_root / "report.md").write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
