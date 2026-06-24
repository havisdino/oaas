from typing import Iterable, Literal

import torch
from torch import nn
from transformers import PreTrainedTokenizer

from oaas.overactivity import compute_oa_scores
from oaas.saliency import OAASaliency
from oaas.utils import prepare_dataloader
from oaas.utils import apply_pruning_masks


def cross_entropy_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    return model(**batch).loss


def hallucination_surplus_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    log_p_pos = -model(**batch["truth"]).loss
    log_p_neg = -model(**batch["hallu"]).loss
    return log_p_neg - log_p_pos 


def get_pruning_masks(
    saliency_scores: dict[str, torch.Tensor],
    pruning_ratio: float,
) -> dict[str, torch.Tensor]:
    if not (0.0 <= pruning_ratio <= 1.0):
        raise ValueError(
            f"pruning_ratio must be in [0, 1], got {pruning_ratio}"
        )

    all_scores = torch.cat([s for s in saliency_scores.values()])
    total_neurons = all_scores.numel()
    n_prune = int(total_neurons * pruning_ratio)

    threshold = torch.kthvalue(all_scores, k=n_prune).values

    masks = {}
    for name, scores in saliency_scores.items():
        masks[name] = (scores > threshold).float()

    return masks


def prune(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    datasets: dict[Literal["hallu", "truth"], Iterable[dict]],
    target_modules: Iterable[str],
    batch_size: int,
    max_length: int,
    oa_weight: float,
    pruning_ratio: float,
    saliency_method: Literal["first_order", "second_order"] = "second_order",
    eps: float = 1e-6,
    **kwargs
) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    num_workers = (
        kwargs.pop("num_workers")
        if "num_workers" in kwargs else 0
    )
    
    oa_scores = compute_oa_scores(
        model,
        tokenizer,
        datasets,
        target_modules,
        batch_size,
        max_length,
        eps,
        **kwargs
    )
    
    saliency = OAASaliency(model, oa_scores, oa_weight, saliency_method)
    dataloader = prepare_dataloader(
        datasets["truth"],
        tokenizer,
        batch_size,
        max_length,
        shuffle=True,
        num_workers=num_workers,
        **kwargs
    )
    sal_scores = saliency.compute(
        dataloader,
        loss_fn=cross_entropy_loss
    )
    
    masks = get_pruning_masks(sal_scores, pruning_ratio)
    apply_pruning_masks(model, masks)
    
    return model, masks
