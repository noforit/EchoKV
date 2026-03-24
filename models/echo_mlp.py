import torch
import torch.nn as nn
from transformers import LlamaConfig

class EchoPartialMLP(nn.Module):
    """
    EchoPartialMLP:
    Input: Anchor(Full) + Target(Partial)
    Output: Target(Remaining)
    """
    def __init__(self, config, total_save_dim: int = 1024, block_size: int = 2):
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        
        # Flatten 
        self.feature_dim = self.num_kv_heads * self.head_dim
        self.total_save_dim = total_save_dim
        
        if self.total_save_dim > self.feature_dim:
            raise ValueError(f"total_save_dim ({self.total_save_dim}) must be <= feature_dim ({self.feature_dim})")
        
        self.target_remain_dim = self.feature_dim - self.total_save_dim
        
        # input: Anchor global(feature_dim) + Target local(total_save_dim)
        if block_size > 1:
            self.input_dim = self.feature_dim + self.total_save_dim
        else:
            self.input_dim = self.total_save_dim
        self.output_dim = self.target_remain_dim
        
        self.linear_k = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.linear_v = nn.Linear(self.input_dim, self.output_dim, bias=True)

    def forward(self, anchor_k, anchor_v, target_k_partial_flat, target_v_partial_flat):

        if anchor_k is not None or anchor_v is not None:
            if anchor_k is not None:
                bsz, _, seq_len, _ = anchor_k.shape
                k_anchor_flat = anchor_k.transpose(1, 2).reshape(bsz, seq_len, -1)
                k_in = torch.cat([k_anchor_flat, target_k_partial_flat], dim=-1)
            else:
                k_in = None

            if anchor_v is not None:
                bsz, _, seq_len, _ = anchor_v.shape
                v_anchor_flat = anchor_v.transpose(1, 2).reshape(bsz, seq_len, -1)
                v_in = torch.cat([v_anchor_flat, target_v_partial_flat], dim=-1)
            else:
                v_in = None
        
        else:
            # 2. Concat Input (no anchor)
            k_in = target_k_partial_flat
            v_in = target_v_partial_flat

        # 3. Predict Remaining
        if k_in is not None:
            rec_k_remain_flat = self.linear_k(k_in)
        else:
            rec_k_remain_flat = None
        if v_in is not None:
            rec_v_remain_flat = self.linear_v(v_in)
        else:
            rec_v_remain_flat = None

        # rec_k_remain_flat = self.linear_k(k_in)
        # rec_v_remain_flat = self.linear_v(v_in)
        
        return rec_k_remain_flat, rec_v_remain_flat


    def get_projected_kv(self, anchor_k, anchor_v, stored_k_partial, stored_v_partial, self_attn_module=None):

        # bsz, n_keep, seq_len, head_dim = stored_k_partial.shape
        if stored_k_partial is not None:
            bsz, seq_len, total_dim = stored_k_partial.shape
        
        # 1. Flatten Storage -> [bsz, seq, total_save_dim]
        # k_partial_flat = stored_k_partial.transpose(1, 2).reshape(bsz, seq_len, -1)
        # v_partial_flat = stored_v_partial.transpose(1, 2).reshape(bsz, seq_len, -1)
            k_partial_flat = stored_k_partial
        else:
            k_partial_flat = None

        if stored_v_partial is not None:
            bsz, seq_len, total_dim = stored_v_partial.shape
            v_partial_flat = stored_v_partial
        else:
            v_partial_flat = None
        
        # 2. Predict Remaining Flattened
        k_remain_flat, v_remain_flat = self.forward(anchor_k, anchor_v, k_partial_flat, v_partial_flat)

        # 3. Concat to Full Flattened
        if k_remain_flat is not None:
            k_recon_flat = torch.cat([k_partial_flat, k_remain_flat], dim=-1)
            rec_k_full = k_recon_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            rec_k_full = None

        if v_remain_flat is not None:
            v_recon_flat = torch.cat([v_partial_flat, v_remain_flat], dim=-1)
            rec_v_full = v_recon_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            rec_v_full = None
        
        
        
        return rec_k_full, rec_v_full