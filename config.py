# config.py
import argparse
import torch

class Training_Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Echo Llama Training Config")
        
        self.parser.add_argument("--model_path", type=str, required=True, help="Path to the Llama model")
        self.parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation to use")
        self.parser.add_argument("--block_size", type=int, default=2, help="block_size n to save KV cache (every n layers)")
        self.parser.add_argument("--total_save_dim", type=int, default=1024)
        self.parser.add_argument("--echo_input_mode", type=str, default='joint')
        self.parser.add_argument("--k_compress_dim", type=int, default=512)
        self.parser.add_argument("--compress_ratio", type=float, default=0.5)
        self.parser.add_argument("--output_dir", type=str, default="./saved_echo_mlps" )
        
        # === training ===
        self.parser.add_argument("--dataset_path", type=str, default='')
        self.parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
        self.parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for training")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        self.parser.add_argument("--num_steps", type=int, default=200, help="Number of training steps")
        self.parser.add_argument("--save_per_step", type=int, default=50, help="Save checkpoint every N steps")
        self.parser.add_argument("--mlp_path", type=str, default=None, help="Path to the trained MLP checkpoint (.pt) [Required for echo]")
        self.parser.add_argument("--loss_type", type=str, default='o_mse')

        # === system ===
        self.parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    def parse(self):
        return self.parser.parse_args()


class Inference_Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Echo Llama Inference Config")
        
        # === Model & Weights ===
        self.parser.add_argument("--model_path", type=str, required=True, help="Path to the base Llama model")
        self.parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation to use")
        self.parser.add_argument("--mlp_path", type=str, required=True, help="Path to the trained .pt file (EchoMLPs)")
        self.parser.add_argument("--block_size", type=int, default=2, help="Must match the block size used in training")
        self.parser.add_argument("--total_save_dim", type=int, default=1024)

        # === System ===
        self.parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    def parse(self):
        return self.parser.parse_args()
    