import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import load_dataset
import click
import json

from oaas.prune import prune


def get_next_version_dir(root_dir, prefix="v"):
    os.makedirs(root_dir, exist_ok=True)
    base_path = Path(root_dir)
    
    v_num = 0
    while True:
        subfolder_name = f"{prefix}{v_num}"
        full_path = base_path / subfolder_name
        
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            return full_path
        
        v_num += 1


@click.command()
@click.option("--model_path", type=str)
@click.option("--target_modules", type=str, multiple=True)
@click.option("--save_dir", type=str)
@click.option("--hallu_data", type=str)
@click.option("--truth_data", type=str)
@click.option("--batch_size", type=int, default=4)
@click.option("--max_length", type=int, default=4096)
@click.option("--oa_weight", type=float, default=1e-2)
@click.option("--pruning_ratio", type=float, default=0.1)
@click.option(
    "--saliency_method",
    type=click.Choice(["first_order", "second_order"]),
    default="second_order",
    show_default=True,
)
@click.option("--apply_chat_template", is_flag=True)
@click.option("--save_masks_only", is_flag=True)
def main(
    model_path,
    target_modules,
    save_dir,
    hallu_data,
    truth_data,
    batch_size,
    max_length,
    oa_weight,
    pruning_ratio,
    saliency_method,
    apply_chat_template,
    save_masks_only
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="bfloat16",
        attn_implementation="sdpa",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    datasets = {
        "hallu": load_dataset("json", split="train", data_files=hallu_data),
        "truth": load_dataset("json", split="train", data_files=truth_data)
    }

    config = dict(
        batch_size=batch_size,
        max_length=max_length,
        oa_weight=oa_weight,
        pruning_ratio=pruning_ratio,
        saliency_method=saliency_method,
        apply_chat_template=apply_chat_template
    )

    model, masks = prune(
        model,
        tokenizer,
        datasets,
        target_modules,
        **config
    )

    save_dir = get_next_version_dir(save_dir)
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        data = json.dumps(config, ensure_ascii=False)
        f.write(data + "\n")
    
    print("Saving outputs to:", save_dir)
    if not save_masks_only:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    else:
        torch.save(masks, os.path.join(save_dir, "masks.pt"))
    
    print("Done")


if __name__ == "__main__":
    main()
