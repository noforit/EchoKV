"""Prepare all 16 LongBench tasks or generate the 13 RULER tasks."""
import argparse
import json
import runpy
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from benchmarks.tasks import LONGBENCH_TASKS, RULER_TOKENS

ROOT = Path(__file__).resolve().parent
MODELS = json.loads((ROOT / 'configs/experiments.json').read_text())['models']
TEMPLATES = {
    'llama8b': '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{task_template}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    'mistral7b': '<s>[INST] {task_template}[/INST]',
    'mistral24b': '<s>[INST] {task_template}[/INST]',
}


def prepare_longbench(archive, output, tasks):
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for task in tasks:
            matches = [name for name in bundle.namelist() if Path(name).name == f'{task}.jsonl']
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {task}.jsonl in {archive}')
            data = bundle.read(matches[0])
            path = output / f'{task}.jsonl'
            path.write_bytes(data)
            print(f'Saved {path}')


def prepare_ruler(args):

    import yaml
    scripts = args.ruler_root.resolve() / 'scripts'
    constants = runpy.run_path(str(scripts / 'data/synthetic/constants.py'))['TASKS']
    customized = yaml.safe_load((scripts / 'synthetic.yaml').read_text())
    output = args.output_dir.resolve() / args.model / str(args.length)
    for task in args.tasks or list(RULER_TOKENS):
        config = customized[task]
        base = constants[config['task']]
        template = TEMPLATES[args.model].format(task_template=base['template']) + base.get('answer_prefix', '')
        cmd = [sys.executable, str(scripts / 'data/synthetic' / f"{config['task']}.py"),
               '--save_dir', str(output), '--save_name', task, '--subset', 'validation',
               '--tokenizer_path', args.model_path or MODELS[args.model], '--tokenizer_type', 'hf',
               '--max_seq_length', str(args.length), '--tokens_to_generate', str(base['tokens_to_generate']),
               '--num_samples', str(args.num_samples), '--random_seed', str(args.seed), '--template', template]
        for key, value in config.get('args', {}).items():
            cmd.extend([f'--{key}', str(value)])
        if config['task'] == 'qa':
            cmd.extend(['--pre_samples', '0'])
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=scripts / 'data', check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='benchmark', required=True)
    lb = sub.add_parser('longbench')
    lb.add_argument('--archive', type=Path, help='Optional existing official data.zip')
    lb.add_argument('--output-dir', type=Path, default=ROOT / 'data/LongBench')
    lb.add_argument('--tasks', nargs='+', choices=LONGBENCH_TASKS, default=LONGBENCH_TASKS)
    ru = sub.add_parser('ruler')
    ru.add_argument('--ruler-root', type=Path, required=True, help='Official NVIDIA/RULER checkout')
    ru.add_argument('--model', choices=list(MODELS), default='llama8b')
    ru.add_argument('--model-path')
    ru.add_argument('--output-dir', type=Path, default=ROOT / 'data/ruler')
    ru.add_argument('--length', type=int, default=31500)
    ru.add_argument('--num-samples', type=int, default=500)
    ru.add_argument('--seed', type=int, default=42)
    ru.add_argument('--tasks', nargs='+', choices=list(RULER_TOKENS))
    ru.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.benchmark == 'longbench':
        archive = args.archive
        if archive is None:
            from huggingface_hub import hf_hub_download
            archive = Path(hf_hub_download('THUDM/LongBench', 'data.zip', repo_type='dataset'))
        prepare_longbench(archive, args.output_dir, args.tasks)
    else:
        if args.length < 1 or args.num_samples < 1:
            p.error('length and num-samples must be positive')
        prepare_ruler(args)
