from collections import defaultdict
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class OAASaliency:

    def __init__(
        self,
        model: nn.Module,
        oa_scores: dict[str, torch.Tensor],
        oa_weight: float
    ):
        self.model = model
        self.target_modules = set(oa_scores.keys())
        self.oa_scores = oa_scores
        self.oa_weight = oa_weight
        
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

                loss = loss_fn(self.model, batch)
                loss.backward()

                for name, act in self._activations.items():
                    first_order_sal = (act.detach() * act.grad.detach())
                    first_order_sal = (
                        first_order_sal
                        .reshape(-1, first_order_sal.shape[-1])
                        .mean(dim=0)
                        .cpu()
                    )

                    accumulated[name] = accumulated[name] + first_order_sal
                    counts[name] += 1

        finally:
            self._remove_hooks()

        scores = {
            name: (
                accumulated[name] / counts[name] - 
                self.oa_weight * self.oa_scores[name].cpu()
            )
            for name in accumulated
        }
        return scores
