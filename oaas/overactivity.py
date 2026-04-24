from typing import Iterable, Literal

import torch
from torch import nn
from transformers import PreTrainedTokenizer

from oaas.recorder import NeuronRecorder


def compute_oa_scores(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    datasets: dict[Literal["hallu", "truth"], Iterable[dict]],
    target_modules: Iterable[str],
    batch_size: int,
    max_length: int,
    eps: float = 1e-6,
    **kwargs
) -> dict[str, torch.Tensor]:
    assert "hallu" in datasets and "truth" in datasets
    
    recorder = NeuronRecorder(model, target_modules)
    mean_abs_act = {}
    
    for key, dataset in datasets.items():
        mean_abs_act[key] = recorder.record(
            dataset,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            **kwargs
        )
    
    oa_scores = {}
    for key in mean_abs_act[key].keys():
        maa_hallu = mean_abs_act["hallu"][key] + eps
        maa_truth = mean_abs_act["truth"][key] + eps
        oa_scores[key] = torch.log(maa_hallu / maa_truth)
    
    return oa_scores
