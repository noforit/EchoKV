import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Training_Config

from models.llama_patch import EchoLlamaPartialManager as EchoLlamaManager
from models.mistral_patch import EchoMistralPartialManager as EchoMistralManager
from data_loader import get_datasets, get_datasets_c4

from trainer import train_echo
import argparse
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation to use")
    parser.add_argument("--block_size", type=int, default=2, help="block_size n to save KV cache (every n layers)")
    parser.add_argument("--total_save_dim", type=int, default=1024)
    parser.add_argument("--echo_input_mode", type=str, default='joint')
    parser.add_argument("--k_compress_dim", type=int, default=512)
    parser.add_argument("--compress_ratio", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="./saved_echo_mlps" )


    parser.add_argument("--dataset_path", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=200, help="Number of training steps")
    parser.add_argument("--save_per_step", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--mlp_path", type=str, default=None, help="Path to the trained MLP checkpoint (.pt) [Required for echo]")
    parser.add_argument("--loss_type", type=str, default='o_mse')


    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler_steps", type=int, default=None)
    args = parser.parse_args()


    return args


def main():


    args = parse_args()
    set_seed(args.seed)

    print(f"Configurations:")
    print(f"  Model: {args.model_path}")
    print(f"  Block_size: {args.block_size}")
    print(f"  Total_save_dim: {args.total_save_dim}")
    print(f"  Echo_input_mode: {args.echo_input_mode}")
    print(f"  Dataset_path: {args.dataset_path}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Loss_type: {args.loss_type}")

    print(f"Loading Model from {args.model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )


    print("Applying Echo Patch...")
    if model.config.model_type == 'llama':
        echo_manager = EchoLlamaManager(model, args)
    elif model.config.model_type == 'mistral':
        echo_manager = EchoMistralManager(model, args)
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")

    echo_manager.to(model.device)

    if args.mlp_path:
        echo_manager.load_mlps(args.mlp_path)
        print(f"Loaded Echo MLPs from {args.mlp_path}")

    print("Preparing Data...")
    if 'c4' in args.dataset_path.lower():
        datasets = get_datasets_c4(tokenizer, args)
    else:
        datasets = get_datasets(tokenizer, args)


    train_echo(echo_manager, datasets, tokenizer, args)


if __name__ == "__main__":
    main()
