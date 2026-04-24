import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, AutoModelForCausalLM
from datasets import Dataset


def make_data_collator(tokenizer: PreTrainedTokenizer):
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    
    def collate_fn(batch: list[dict]) -> dict:
        input_ids = [torch.tensor(e["input_ids"]) for e in batch]
        attention_mask = [torch.tensor(e["attention_mask"]) for e in batch]
        labels = [torch.tensor(e["labels"]) for e in batch]
        
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    return collate_fn


def prepare_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    apply_chat_template: bool,
    chat_template_args: dict | None = None,
    num_workers: int = 0,
    ignore_index: int = -100
) -> DataLoader:
    assert (
        "prompt" in dataset.column_names and
        "response" in dataset.column_names
    )
    chat_template_args = chat_template_args or {}
    
    def tokenize(example):
        prompt = example["prompt"]
        response = example["response"]
        
        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_args
            )
        
        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(response)
        input_ids = prompt_ids + response_ids
        input_ids = input_ids[:max_length]
        
        seqlen = len(prompt_ids) + len(response_ids)
        attention_mask = [1 for _ in range(seqlen)]
        
        labels = [ignore_index for _ in prompt_ids] + response_ids
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        collate_fn=make_data_collator(tokenizer),
        num_workers=num_workers
    )
    return dataloader


def apply_pruning_masks(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
):
    for name, module in model.named_modules():
        if name in masks:
            w = module.weight.data
            m = masks[name].to(w.device, w.dtype)
            module.weight.data = w * m


def load_model_and_apply_masks(
    model_path: str,
    masks: dict[str, torch.Tensor],
    **model_init_kwargs
) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_init_kwargs
    )
    apply_pruning_masks(model, masks)
    return model