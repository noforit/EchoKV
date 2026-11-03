from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(args, hybrid=False):
    tokenizer_kwargs = {}
    if 'mistral-small' in args.model_path.lower():
        tokenizer_kwargs['fix_mistral_regex'] = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True, padding_side='left', **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype='auto', device_map='auto', trust_remote_code=True,
        low_cpu_mem_usage=True, use_cache=True, attn_implementation=args.attn_implementation)
    manager = None
    if args.method == 'echo':
        if model.config.model_type == 'llama':
            if hybrid:
                from models.llama_patch_think import EchoLlamaPartialManager as Manager
            else:
                from models.llama_patch import EchoLlamaPartialManager as Manager
        elif model.config.model_type == 'mistral' and not hybrid:
            from models.mistral_patch import EchoMistralPartialManager as Manager
        else:
            raise ValueError('EchoKV supports Llama/Mistral; the hybrid experiment targets Llama')
        manager = Manager(model, args)
        manager.load_mlps(args.mlp_path)
        manager.to(model.device)
        manager.set_mode(True)
    model.eval()
    return tokenizer, model, manager
