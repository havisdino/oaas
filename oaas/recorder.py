from typing import Callable, Iterable

import torch
from torch import nn
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class NeuronRecorder:

    def __init__(self, model: nn.Module, target_modules: Iterable[str]):
        self.model = model
        self.target_modules = set(target_modules)
        self._hooks = []

        self._abs_act_sums = {}
        self._token_counts = {}
        
        self._response_mask = None

    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            act = input[0]
            
            if self._response_mask is None or act.dim() != 3:
                return

            mask = self._response_mask.to(act.device)
            abs_act = act.detach().float().abs()

            mask = mask.float().unsqueeze(-1)
            masked_sum = (abs_act * mask).sum(dim=(0, 1))
            token_count = mask.sum().item()

            if name not in self._abs_act_sums:
                self._abs_act_sums[name] = masked_sum
                self._token_counts[name] = token_count
            else:
                self._abs_act_sums[name] += masked_sum
                self._token_counts[name] += token_count

        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_modules:
                hook_fn = self._make_hook(name)
                self._hooks.append(
                    module.register_forward_hook(hook_fn)
                )

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
    
    def _prepare_inputs(
        self,
        prompts: list[str],
        responses: list[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        apply_chat_template: bool = False,
        chat_template_args: dict | None = None
    ) -> dict[str, torch.Tensor]:
        if apply_chat_template:
            assert tokenizer.chat_template is not None
            chat_template_args = chat_template_args or {}
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_args
                ) for p in prompts
            ]
        
        full_texts = [p + r for p, r in zip(prompts, responses)]
        full_enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        prompt_lengths = []
        for prompt in prompts:
            enc = tokenizer(
                prompt,
                return_tensors="pt",
            )
            prompt_lengths.append(enc["input_ids"].shape[1])

        bsz, seqlen = full_enc["input_ids"].shape
        response_mask = torch.zeros(bsz, seqlen, dtype=torch.float32)
        
        for i, pl in enumerate(prompt_lengths):
            seqlen = int(full_enc["attention_mask"][i].sum().item())
            if pl < seqlen:
                response_mask[i, pl:None] = 1.0

        device = self.model.device
        self._response_mask = response_mask.to(device)
        
        return full_enc

    def reset(self):
        self._abs_act_sums.clear()
        self._token_counts.clear()

    def get_average_activations(self) -> dict[str, torch.Tensor]:
        return {
            name: self._abs_act_sums[name] / max(self._token_counts[name], 1)
            for name in self._abs_act_sums
        }

    @torch.no_grad()
    def record(
        self,
        dataset: Iterable[dict],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        max_length: int = 2048,
        apply_chat_template: bool = False,
        chat_template_args: dict | None = None
    ) -> dict[str, torch.Tensor]:
        self.model.eval()
        self.reset()
        self._register_hooks()

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"

        try:
            for start in tqdm(
                range(0, len(dataset), batch_size),
                desc="Recording activations"
            ):
                batch = dataset[start : start + batch_size]
                prompts = batch["prompt"]
                responses = batch["response"]

                inputs = self._prepare_inputs(
                    prompts,
                    responses,
                    tokenizer,
                    max_length,
                    apply_chat_template,
                    chat_template_args
                )

                self.model(**inputs)

        finally:
            self._remove_hooks()
            self._response_mask = None
            tokenizer.padding_side = original_padding_side

        return self.get_average_activations()