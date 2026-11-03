# Experiment interfaces

The main LongBench/RULER interfaces support all three paper backbones. Ablation,
Hybrid, switching, and adaptive-workload interfaces target Llama-3.1-8B only.
Add `--dry-run` to preview commands.

## Models and cache settings

| `--model` | Hugging Face model ID |
|---|---|
| `llama8b` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `mistral7b` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `mistral24b` | `mistralai/Mistral-Small-24B-Instruct-2501` |

Use `--model-path /path/to/model` to override the ID with a local copy.

| `--ratio` | Group size | Local dimensions | Nominal ratio in compressed region |
|---|---:|---:|---:|
| `0.7` | 2 | 384 | 0.6875 |
| `0.5` | 2 | 0 | 0.5 |
| `0.3` | 4 | 64 | 0.296875 |

All three models use flattened KV dimension 1,024. These settings are stored
once in `configs/experiments.json`; full sink/window tokens add cache storage.

## Data preparation

```bash
python -m pip install -r requirements-data.txt
python prepare_data.py longbench
```

This downloads the official THUDM/LongBench `data.zip` and extracts the 16 tasks
used in the paper. To use a local archive, pass `--archive /path/to/data.zip`.
The shell entrypoints default to all 16 tasks; use
`DATASETS=narrativeqa` for the bundled single-task example.

Prepare RULER with the official task generators:

```bash
git clone https://github.com/NVIDIA/RULER.git third_party/RULER
python -m nltk.downloader punkt punkt_tab
(cd third_party/RULER/scripts/data/synthetic/json && \
    python download_paulgraham_essay.py && bash download_qa_dataset.sh)

python prepare_data.py ruler --ruler-root third_party/RULER \
    --model llama8b --model-path /path/to/Meta-Llama-3.1-8B-Instruct \
    --length 31500 --num-samples 500 --output-dir data/ruler
```

Repeat the last command with `--model mistral7b` and `--model mistral24b`, using
the corresponding tokenizer paths. Generated files are placed at
`data/ruler/<model>/<length>/<task>/validation.jsonl`. The main runner's 32K
setting uses length 31,500; length and samples per task are explicit arguments.
Select the training JSON with `--data /path/to/LongAlpaca-12k.json`.
The default is the bundled `data/LongAlpaca-1k.json` subset.

## Main training and evaluation

```bash
python experiments.py train --model llama8b --ratio 0.5 \
    --data /path/to/LongAlpaca-12k.json --output runs/llama8b/0.5

python experiments.py evaluate --model llama8b --ratio 0.5 \
    --checkpoint runs/llama8b/0.5/stage2_o_mse/checkpoint-1000.pt \
    --benchmark longbench --data-dir data/LongBench \
    --output runs/llama8b/0.5/longbench

python experiments.py evaluate --model llama8b --ratio 0.5 \
    --checkpoint runs/llama8b/0.5/stage2_o_mse/checkpoint-1000.pt \
    --benchmark ruler --data-dir data/ruler/llama8b/31500 \
    --output runs/llama8b/0.5/ruler
```

The same commands support the other two model aliases and ratios. Main training
uses 600 KV-MSE updates followed by 1,000 O-MSE updates, with cosine scheduler
horizons 1,000 and 2,000. Each stage
loads the preceding stage's final checkpoint. Use a separate output directory
for each model/configuration.

Both evaluation commands run all tasks by default (16 LongBench / 13 RULER),
then write per-task scores, sample counts, and a macro-average to
`scores_echo.json` and `scores_echo.csv`. Use `--tasks ...` to select tasks
or `--method base` to evaluate FullKV without a predictor checkpoint.

Score existing predictions independently:

```bash
python score.py --benchmark ruler \
    --results-dir runs/llama8b/0.5/ruler/meta-llama-3.1-8b-instruct --method echo
```

## Llama ablations and Hybrid

All settings reuse the normal train/evaluate entrypoints with linear predictors
and contiguous local dimensions. Pass the same input and cache settings during
both stages and evaluation.
`--ratio` supplies defaults; `--block-size` and `--local-dim` override them.

| Experiment | Interface |
|---|---|
| Loss objectives | `--schedule kv_only`, `o_only`, `qk_kl_only`, `kv_o`, or `kv_qk_kl` (training only) |
| Input features | `--input-mode joint`, `anchor_only`, or `target_only` |
| Group/local dimension | `--block-size S --local-dim m` |
| EchoKV + ThinK | `--method hybrid --key-ratio 0.5` |

Loss-only presets train for 1,000 steps. Two-objective ablations use 700+300
steps, separately from the main 600+1,000 schedule. Ablation scheduler horizons
default to their stage lengths; override them with `--scheduler-steps H1 H2`.

For the ratio-0.75 input comparison, use `--block-size 2 --local-dim 512` and vary
`--input-mode`. The predictor's input changes while the retained cache budget
stays fixed.

Examples:

```bash
python experiments.py train --model llama8b --ratio 0.5 --schedule kv_o \
    --data /path/to/LongAlpaca-12k.json --output runs/loss/kv_o
python experiments.py train --model llama8b --ratio 0.7 \
    --block-size 2 --local-dim 512 --input-mode anchor_only \
    --data /path/to/LongAlpaca-12k.json --output runs/input/anchor_only
python experiments.py train --model llama8b --ratio 0.5 \
    --method hybrid --key-ratio 0.5 --output runs/hybrid
```

Predictors accept both the public `linear_k/v` and internal `project_k/v`
checkpoint key names.

## Llama switching and mixed workload

Switching uses pre-RoPE predictors and applies inverse RoPE to the native FullKV
cache during conversion. Timing runs disable per-layer allocator cleanup by default.

```bash
python experiments.py switch --model llama8b --ratio 0.5 \
    --checkpoint runs/llama8b/0.5/stage2_o_mse/checkpoint-1000.pt \
    --switch-after 32 --repeats 5 --output runs/switch32

python experiments.py adaptive --model llama8b --ratio 0.5 \
    --checkpoint runs/llama8b/0.5/stage2_o_mse/checkpoint-1000.pt \
    --output runs/adaptive
```

Switching compares FullKV with Switch@32 (or `--switch-after 64`) at 16K/32K,
batch 1, generating 256 tokens. Each repetition warms up independently and
writes to a separate directory. Its summary reports mean latency, TPOT,
persistent cache, predictor storage, peak memory, and completion counts.

Adaptive defaults are 4K/8K/32K/36K, five **batches** per length, eight sequences
per batch, and 16 output tokens: 20 batches / 160 sequences per policy. Policies
are FullKV-only, static EchoKV, length routing, and memory-triggered routing.
Length routing uses threshold 32,768; memory routing uses peak allocated memory
threshold 62 GiB, configurable with `--memory-threshold-gib`. The shuffled
workload/seed is shared across policies, and warm-up is excluded from results.

Each run stores status (`completed`, `oom`, or `failed`), TPOT, switch timing,
cache bytes, predictor bytes, and allocated/reserved/peak memory separately.
Fields ending in `*_bytes`, `*_gib`, and `*_s` use bytes, GiB, and seconds.
Completion counts refer to batches, with batch size recorded alongside them.

The launchers generate summaries automatically. To regenerate them:

```bash
python -m benchmarks.summarize_switching --results-root runs/switch32
python -m benchmarks.summarize_adaptive_workload --results_root runs/adaptive \
    --policies fullkv_only echo_only adaptive_length adaptive_memory
```

The optional low-level `palu_static` policy requires the separately installed
Palu implementation and a pre-compressed checkpoint; `--palu_model_module`
selects its model-registration module.

## Acknowledgments

RULER scoring functions are adapted from [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
under the Apache 2.0 license; see [LICENSE-RULER.txt](benchmarks/LICENSE-RULER.txt).
