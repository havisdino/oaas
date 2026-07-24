from collections import defaultdict
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class OAASaliency:

    def __init__(
        self,
        model: nn.Module,
        oa_scores: dict[str, torch.Tensor],
        oa_weight: float,
        saliency_method: Literal["first_order", "second_order"] = "second_order"
    ):
        if saliency_method not in {"first_order", "second_order"}:
            raise ValueError(
                "saliency_method must be either 'first_order' or 'second_order', "
                f"got {saliency_method}"
            )

        self.model = model
        self.target_modules = set(oa_scores.keys())
        self.oa_scores = oa_scores
        self.oa_weight = oa_weight
        self.saliency_method = saliency_method
        
        self._hooks = []
        self._activations = {}
    
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_modules:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            self._activations[name] = input[0]
            input[0].retain_grad()
        return hook

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _activation_mask_saliency(self, act: torch.Tensor) -> torch.Tensor:
        first_order_sal = act.detach().float() * act.grad.detach().float()

        if self.saliency_method == "first_order":
            saliency = first_order_sal
        else:
            saliency = 0.5 * first_order_sal.square()

        return saliency.reshape(-1, saliency.shape[-1]).mean(dim=0).cpu()
    
    def compute(
        self,
        dataloader: DataLoader,
        loss_fn: Callable[[nn.Module, dict], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        accumulated = defaultdict(lambda: torch.tensor(0.0))
        counts = defaultdict(int)
        
        self._register_hooks()
        self.model.eval()

        try:
            for batch in tqdm(dataloader, desc="Computing saliency scores"):
                self.model.zero_grad()
                self._activations.clear()

                loss = loss_fn(self.model, batch)
                loss.backward()

                for name, act in self._activations.items():
                    saliency = self._activation_mask_saliency(act)

                    accumulated[name] = accumulated[name] + saliency
                    counts[name] += 1

        finally:
            self._remove_hooks()
        
        scores = {}
        
        for name in accumulated:
            oa_score = self.oa_scores[name].cpu()
            w_oa = torch.relu(oa_score) * self.oa_weight
            scores[name] = accumulated[name] / counts[name] - w_oa
        
        return scores
