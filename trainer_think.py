import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from transformers import get_scheduler
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from models.think_utils import repeat_kv
import time
def train_echo(echo_manager, datasets, tokenizer, args):

    total_samples = len(datasets)
    args.num_steps = min(args.num_steps, total_samples // args.batch_size)

    optimizer = torch.optim.AdamW(echo_manager.echo_mlps.parameters(), lr=args.lr)

    scheduler = get_scheduler(
        name="cosine",             
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_steps
    )


    model = echo_manager.model
    
    model.eval() 
    model.requires_grad_(False) 
    
    echo_manager.echo_mlps.requires_grad_(True)
    echo_manager.set_mode(False) 
    echo_manager.is_training = True

    global_step = 0 
    loss_type = getattr(args, 'loss_type', 'kv_mse') 
    
    print(f"Start Training... Total Steps: {args.num_steps} | Loss Type: {loss_type}")

    stop_training = False
    

    
    torch.cuda.synchronize()
    training_start = time.time()
    
    while not stop_training:
        for i in range(0, len(datasets), args.batch_size):


            if global_step >= args.num_steps:
                stop_training = True
                break

            batch_prompts = datasets[i:i+args.batch_size]


            tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            input_ids = tokenized_prompts.input_ids
            if args.loss_type in ['qk_kl']:
                input_ids = input_ids[:, :16384]
            

            with torch.no_grad():
                model(input_ids)
                
            total_loss_log = 0.0 
            log_metrics = {'k': 0.0, 'v': 0.0, 'main': 0.0}
            trained_layers_count = 0

            optimizer.zero_grad()

            
            for layer_idx in range(model.config.num_hidden_layers):

                block_id, anchor_idx = echo_manager.get_anchor_info(layer_idx)
                
                if layer_idx == anchor_idx and echo_manager.block_size > 1:
                    continue
                    
                target_k = echo_manager.training_buffer[layer_idx]['k'].detach()
                pred_k = echo_manager.training_buffer[layer_idx]['think_k'].detach()
                target_v = echo_manager.training_buffer[layer_idx]['v'].detach()

                target_q = echo_manager.training_buffer[layer_idx]['q'].detach() 
                cos = echo_manager.training_buffer[layer_idx]['cos']         
                sin = echo_manager.training_buffer[layer_idx]['sin']          
                mask = echo_manager.training_buffer[layer_idx]['mask']
                target_o = echo_manager.training_buffer[layer_idx]['o'].detach()
                
                # get KV (Anchor)
                if echo_manager.block_size > 1:
                    anchor_v = echo_manager.training_buffer[anchor_idx]['v'].detach()
                else:
                    anchor_v = None
                

                mlp = echo_manager.echo_mlps[layer_idx]
                curr_layer_attn = model.model.layers[layer_idx].self_attn

                bsz, _, seq_len, _ = target_k.shape
                target_v_save = target_v.transpose(1, 2).reshape(bsz, seq_len, -1)[:, :, :echo_manager.total_save_dim]


                _, pred_v = mlp.get_projected_kv(None, anchor_v, None, target_v_save, curr_layer_attn)
                
                with torch.no_grad():
                    l_k_mon = F.mse_loss(pred_k, target_k)
                    l_v_mon = F.mse_loss(pred_v, target_v)
                    log_metrics['k'] += l_k_mon.item()
                    log_metrics['v'] += l_v_mon.item()

                bsz, n_heads, seq_len, head_dim = target_q.shape
                
                layer_loss = 0.0


                if loss_type == 'o_mse':
                    q_rotated, k_pred_rotated = apply_rotary_pos_emb(target_q, pred_k, cos, sin)

                    position_ids = torch.arange(seq_len, device=target_q.device).unsqueeze(0)

                    attn_kwargs = {
                        'position_ids': position_ids,
                        'use_cache': False
                    }
                    attention_interface = ALL_ATTENTION_FUNCTIONS['flash_attention_2'] # 或者 'eager_attention'
                    # print(q_rotated.shape, k_pred_rotated.shape, pred_v.shape, mask.shape)
                    # input()
                    
                    # final_value_states = repeat_kv(final_value_states, self_attn_module.num_key_value_groups)
                    pred_v = repeat_kv(pred_v, curr_layer_attn.num_key_value_groups)
                    attn_output, attn_weights = attention_interface(
                        curr_layer_attn,
                        query = q_rotated, 
                        key = k_pred_rotated,  
                        value = pred_v,        
                        # value = target_v,          
                        attention_mask = mask, 
                        scaling=curr_layer_attn.scaling,
                        dropout_p=0.0,
                        **attn_kwargs
                    )
                    
                    attn_output = attn_output.reshape(bsz, seq_len, -1)

                    pred_o = curr_layer_attn.o_proj(attn_output)


                    
                    layer_loss = F.mse_loss(pred_o, target_o)
                    log_metrics['main'] += layer_loss.item()
                    del target_o, attn_output, pred_o


                elif loss_type == 'kv_mse':
                    layer_loss = F.mse_loss(pred_v, target_v)
                    log_metrics['main'] += layer_loss.item()
                

                layer_loss.backward()
                
                total_loss_log += layer_loss.item()
                
                trained_layers_count += 1

                del pred_k, pred_v, target_k, target_v
                del layer_loss

            




            optimizer.step()
            scheduler.step()

            echo_manager.training_buffer = {}

            global_step += 1
            
            if global_step % 5 == 0:
                # avg_k = sum_loss_k / trained_layers_count if trained_layers_count > 0 else 0.0
                # avg_v = sum_loss_v / trained_layers_count if trained_layers_count > 0 else 0.0
                
                current_lr = optimizer.param_groups[0]['lr']
                div = trained_layers_count if trained_layers_count > 0 else 1
                
                stage_info = loss_type

                print(f"Step {global_step}/{args.num_steps} | "
                      f"Mode: {stage_info} | "
                      f"Total_Loss: {total_loss_log:.4f} | "
                      f"Main_Loss_Avg: {log_metrics['main']/div:.5f} | "
                      f"K_MSE: {log_metrics['k']/div:.5f} | "
                      f"V_MSE: {log_metrics['v']/div:.5f} | "
                      f"LR: {current_lr:.2e}")

            if args.save_per_step > 0 and global_step % args.save_per_step == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                echo_manager.save_mlps(save_path)
