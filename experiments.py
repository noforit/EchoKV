"""Portable launch interface: shared configs, no model imports until execution."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from benchmarks.tasks import LONGBENCH_TASKS, RULER_TOKENS

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'configs/experiments.json').read_text())
POLICIES = ['fullkv_only', 'echo_only', 'adaptive_length', 'adaptive_memory']


def command(args, *parts):
    cmd = [sys.executable, *map(str, parts)]
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def predictor_flags(args):
    config = CONFIG['ratios'][args.ratio].copy()
    if args.block_size is not None:
        config['block_size'] = args.block_size
    if args.local_dim is not None:
        config['total_save_dim'] = args.local_dim
    flags = ['--block_size', config['block_size'], '--total_save_dim', config['total_save_dim'],
             '--echo_input_mode', args.input_mode]
    if args.method == 'hybrid':
        return flags + ['--compress_ratio', args.key_ratio]
    return flags


def train(args):
    stages = CONFIG['schedules'][args.schedule]
    if args.scheduler_steps and len(args.scheduler_steps) != len(stages):
        raise ValueError('--scheduler-steps must provide one value per stage')
    entry = ROOT / ('train_think.py' if args.method == 'hybrid' else 'train.py')
    common = ['--model_path', args.model_path, '--dataset_path', args.data.resolve(),
              '--batch_size', 1, '--lr', args.lr, '--seed', args.seed, '--save_per_step', 100,
              *predictor_flags(args)]
    checkpoint = None
    for i, (loss, steps, scheduler) in enumerate(stages):
        output = args.output / f'stage{i+1}_{loss}'
        if checkpoint and not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
        extra = ['--mlp_path', checkpoint] if checkpoint else []
        scheduler = args.scheduler_steps[i] if args.scheduler_steps else scheduler
        command(args, entry, *common, '--loss_type', loss, '--num_steps', steps,
                '--scheduler_steps', scheduler, '--output_dir', output, *extra)
        checkpoint = output / f'checkpoint-{steps}.pt'
    if not args.dry_run and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(f'Final checkpoint: {checkpoint}')


def evaluate(args):
    tasks = args.tasks or (LONGBENCH_TASKS if args.benchmark == 'longbench' else list(RULER_TOKENS))
    method = 'base' if args.method == 'base' else 'echo'
    if args.benchmark == 'longbench':
        entry = 'inference_longbench_think.py' if args.method == 'hybrid' else 'inference_longbench.py'
        extra = []
    else:
        entry = 'inference_ruler.py'
        extra = ['--hybrid'] if args.method == 'hybrid' else []
    flags = predictor_flags(args) if method == 'echo' else []
    if args.checkpoint:
        flags += ['--mlp_path', args.checkpoint.resolve()]
    command(args, ROOT / entry, '--model_path', args.model_path, '--method', method,
            '--data_dir', args.data_dir.resolve(), '--save_dir', args.output,
            '--datasets', *tasks, '--sink_size', 4, '--local_window_size', 128,
            '--attn_implementation', 'flash_attention_2', '--seed', args.seed, *flags, *extra)
    model_name = args.model_path.rstrip('/').split('/')[-1].lower()
    command(args, ROOT / 'score.py', '--benchmark', args.benchmark,
            '--results-dir', args.output / model_name, '--method', method, '--tasks', *tasks)


def switch(args):
    cfg = CONFIG['ratios'][args.ratio]
    for repeat in range(1, args.repeats+1):
        for length in args.lengths:
            for mode in ('fullkv', 'switch'):
                extra = ['--mlp_path', args.checkpoint.resolve()] if mode == 'switch' else []
                command(args, '-m', 'benchmarks.benchmark_switching', '--mode', mode,
                        '--model_path', args.model_path, '--prompt_length', length,
                        '--output_length', 256, '--switch_after', args.switch_after,
                        '--block_size', cfg['block_size'], '--total_save_dim', cfg['total_save_dim'],
                        '--seed', args.seed, '--warmup', '--output_dir', args.output / f'run_{repeat}', *extra)
    command(args, '-m', 'benchmarks.summarize_switching', '--results-root', args.output)


def adaptive(args):
    cfg = CONFIG['ratios'][args.ratio]
    for policy in args.policies:
        extra = []
        if policy == 'adaptive_length':
            extra += ['--length_threshold', args.length_threshold]
        elif policy == 'adaptive_memory':
            extra += ['--peak_allocated_threshold_gib', args.memory_threshold_gib]
        if policy != 'fullkv_only':
            extra += ['--mlp_path', args.checkpoint.resolve()]
        command(args, '-m', 'benchmarks.benchmark_adaptive_workload', '--policy', policy,
                '--model_path', args.model_path, '--prompt_lengths', *args.lengths,
                '--requests_per_length', args.batches_per_length, '--batch_size', args.batch_size,
                '--output_length', 16, '--seed', args.seed,
                '--block_size', cfg['block_size'], '--total_save_dim', cfg['total_save_dim'],
                '--empty_cache_between_requests', '--output_dir', args.output / policy, *extra)
    command(args, '-m', 'benchmarks.summarize_adaptive_workload',
            '--results_root', args.output, '--policies', *args.policies)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    for action in ('train', 'evaluate', 'switch', 'adaptive'):
        c = sub.add_parser(action)
        c.add_argument('--model', choices=CONFIG['models'], default='llama8b')
        c.add_argument('--model-path', help='Override the Hugging Face model ID with a local path')
        c.add_argument('--ratio', choices=CONFIG['ratios'], default='0.5')
        c.add_argument('--output', type=Path, required=True)
        c.add_argument('--seed', type=int, default=42)
        c.add_argument('--dry-run', action='store_true')
        if action != 'train':
            c.add_argument('--checkpoint', type=Path)
        if action in ('train', 'evaluate'):
            c.add_argument('--method', choices=['echo','hybrid','base'] if action=='evaluate' else ['echo','hybrid'], default='echo')
            c.add_argument('--block-size', type=int)
            c.add_argument('--local-dim', type=int)
            c.add_argument('--input-mode', choices=['joint','anchor_only','target_only'], default='joint')
            c.add_argument('--key-ratio', type=float, default=0.5)
        if action == 'train':
            c.add_argument('--data', type=Path, default=ROOT / 'data/LongAlpaca-1k.json')
            c.add_argument('--schedule', choices=CONFIG['schedules'], default='main')
            c.add_argument('--scheduler-steps', type=int, nargs='+')
            c.add_argument('--lr', type=float, default=5e-4)
        elif action == 'evaluate':
            c.add_argument('--benchmark', choices=['longbench','ruler'], required=True)
            c.add_argument('--data-dir', type=Path, required=True)
            c.add_argument('--tasks', nargs='+')
        elif action == 'switch':
            c.add_argument('--lengths', nargs='+', type=int, default=[16384,32768])
            c.add_argument('--switch-after', type=int, choices=[32,64], default=32)
            c.add_argument('--repeats', type=int, default=5)
        else:
            c.add_argument('--lengths', nargs='+', type=int, default=[4096,8192,32768,36864])
            c.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
            c.add_argument('--batch-size', type=int, default=8)
            c.add_argument('--batches-per-length', type=int, default=5)
            c.add_argument('--length-threshold', type=int, default=32768)
            c.add_argument('--memory-threshold-gib', type=float, default=62)
    args = p.parse_args()
    args.model_path = args.model_path or CONFIG['models'][args.model]
    if Path(args.model_path).is_dir():
        args.model_path = str(Path(args.model_path).resolve())
    args.output = args.output.resolve()
    if args.action in ('switch', 'adaptive') and args.model != 'llama8b':
        p.error('Switching and workload experiments target llama8b')
    if args.action in ('train','evaluate'):
        advanced = (args.method == 'hybrid' or args.block_size is not None or args.local_dim is not None
                    or args.input_mode != 'joint'
                    or (args.action == 'train' and args.schedule != 'main'))
        if args.model != 'llama8b' and advanced:
            p.error('Ablation and hybrid interfaces target llama8b; use main configurations for Mistral')
        if (args.method == 'hybrid' and args.action == 'train'
                and args.schedule not in ('main','kv_only','o_only','kv_o')):
            p.error('Hybrid supports KV-MSE/O-MSE schedules')
        if args.block_size is not None and args.block_size < 1:
            p.error('--block-size must be positive')
        if args.local_dim is not None and not 0 <= args.local_dim <= 1024:
            p.error('--local-dim must be in [0,1024] for these backbones')
    needs_checkpoint = (args.action == 'switch'
                        or args.action == 'evaluate' and args.method != 'base'
                        or args.action == 'adaptive' and any(x != 'fullkv_only' for x in args.policies))
    if needs_checkpoint and not args.checkpoint:
        p.error('--checkpoint is required for this action')
    if args.action == 'evaluate':
        allowed = LONGBENCH_TASKS if args.benchmark == 'longbench' else RULER_TOKENS
        if args.tasks and any(task not in allowed for task in args.tasks):
            p.error('Unknown task for the selected benchmark')
    if args.action == 'switch' and args.repeats < 1:
        p.error('--repeats must be positive')
    return args


if __name__ == '__main__':
    args = parse_args()
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        (args.output / f'{args.action}_config.json').write_text(json.dumps(metadata, indent=2) + '\n')
    {'train': train, 'evaluate': evaluate, 'switch': switch, 'adaptive': adaptive}[args.action](args)
