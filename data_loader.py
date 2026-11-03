import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from tqdm import tqdm

def get_datasets(tokenizer, args):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading data from {args.dataset_path} ...")
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {args.dataset_path}")

    datasets = []

    datalen = []

    for item in tqdm(data, desc="Processing samples"):

        instruction = item.get("instruction", "")
        user_input = item.get("input", "")
        output = item.get("output", "")
        if 'pred' in args.dataset_path.lower():
            output = item.get("pred", "")

        if user_input and user_input.strip():
            user_content = f"{instruction}\n{user_input}"
        else:
            user_content = instruction

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]

        prompts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


        datasets.append(prompts)


    return datasets

def get_datasets_c4(tokenizer, args):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading data from {args.dataset_path} ...")
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {args.dataset_path}")

    datasets = []
    datalen = []
    for item in tqdm(data, desc="Processing samples"):

        instruction = item.get("text", "")
        user_input = item.get("input", "")
        output = item.get("output", "")
        if 'pred' in args.dataset_path.lower():
            output = item.get("pred", "")

        if user_input and user_input.strip():
            user_content = f"{instruction}\n{user_input}"
        else:
            user_content = instruction

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]

        prompts = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


        datasets.append(prompts)


    return datasets
