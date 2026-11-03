import torch

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    if kv_states.shape[1] != q_states.shape[1]:
        kv_states = repeat_kv(kv_states, q_states.shape[1]//kv_states.shape[1])
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)

    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm

    del q_states
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    del key
    mask = mask.scatter_(-1, keep_idx, 1)
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k], kv_states[:, :, seqlen - recent_size:, :], ~mask


def recover_cache(key_states_pruned, mask, ratio=0.4):
    _, heads, head_dim  = mask.shape
    k = head_dim - int(head_dim * ratio)
    sqlen = int(key_states_pruned.shape[-1]/(k*heads))
    mask = mask.unsqueeze(2).expand(-1, -1, sqlen, -1)
    recovered_key_states = torch.zeros(1, heads, sqlen, head_dim, dtype=key_states_pruned.dtype, device = key_states_pruned.device)
    recovered_key_states[mask] = key_states_pruned
    del mask
    return recovered_key_states
