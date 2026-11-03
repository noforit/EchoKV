"""RULER generation using the original input + answer-prefix protocol."""
import argparse
import json
from pathlib import Path
import torch
from runtime import load_model
from benchmarks.tasks import RULER_TOKENS
from benchmarks.switching_common import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_path', required=True)
    p.add_argument('--method', choices=['base', 'echo'], default='echo')
    p.add_argument('--hybrid', action='store_true')
    p.add_argument('--mlp_path')
    p.add_argument('--data_dir', type=Path, required=True, help='<task>/validation.jsonl parent')
    p.add_argument('--datasets', nargs='+', choices=list(RULER_TOKENS), default=list(RULER_TOKENS))
    p.add_argument('--save_dir', type=Path, required=True)
    p.add_argument('--eval_batch_size', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--attn_implementation', default='flash_attention_2')
    p.add_argument('--block_size', type=int, default=2)
    p.add_argument('--total_save_dim', type=int, default=0)
    p.add_argument('--echo_input_mode', default='joint', choices=['joint','anchor_only','target_only'])
    p.add_argument('--compress_ratio', type=float, default=0.5)
    p.add_argument('--sink_size', type=int, default=4)
    p.add_argument('--local_window_size', type=int, default=128)
    args = p.parse_args()
    if args.method == 'echo' and not args.mlp_path:
        p.error('--mlp_path is required for EchoKV')
    if args.eval_batch_size < 1:
        p.error('--eval_batch_size must be positive')
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    set_seed(args.seed)
    for task in args.datasets:
        if not (args.data_dir / task / 'validation.jsonl').is_file():
            raise FileNotFoundError(args.data_dir / task / 'validation.jsonl')
    tokenizer, model, manager = load_model(args, hybrid=args.hybrid)
    model_name = args.model_path.rstrip('/').split('/')[-1].lower()
    for task in args.datasets:
        with (args.data_dir / task / 'validation.jsonl').open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        output = args.save_dir / model_name / task / f'{args.method}.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w') as handle:
            for start in range(0, len(rows), args.eval_batch_size):
                batch = rows[start:start + args.eval_batch_size]
                if manager is not None:
                    manager.set_mode(True)
                prompts = [row['input'] + row.get('answer_prefix', '') for row in batch]
                tokens = tokenizer(prompts, padding=True, return_tensors='pt', add_special_tokens=True).to(model.device)
                context = tokens.input_ids.shape[-1]
                output_ids = model.generate(**tokens, max_new_tokens=RULER_TOKENS[task],
                                            do_sample=False, num_beams=1, min_length=context+1,
                                            eos_token_id=tokenizer.eos_token_id)
                texts = tokenizer.batch_decode(output_ids[:, context:], skip_special_tokens=True)
                for row, prompt, pred in zip(batch, prompts, texts):
                    record = dict(prompt=prompt, answers=row['outputs'], pred=pred,
                                  length=row.get('length'), _id=row['index'])
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
        print(output)


if __name__ == '__main__':
    main()
