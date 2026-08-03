# OAAS: Overactivity Neuron Saliency

## Requirements

Install the core dependencies:

```bash
pip install torch transformers datasets click tqdm
```

For large models, make sure your local PyTorch installation matches your CUDA, MPS, or CPU environment. The CLI loads models with `device_map="auto"` and `dtype="bfloat16"`, so a GPU setup is recommended for multi-billion-parameter models.

## Data Format

The CLI expects two JSON or JSONL datasets:

- `hallu_data`: examples containing hallucination-style responses
- `truth_data`: examples containing truthful/reference responses

Each row must contain `prompt` and `response` fields:

```json
{"prompt": "Who wrote Pride and Prejudice?", "response": "Jane Austen wrote Pride and Prejudice."}
```

The response tokens are used for activation recording and loss computation. Prompt tokens are masked from the training labels with `-100`.

## Usage

Run the pruning pipeline from the repository root:

```bash
python -m oaas.cli \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --target_modules model.layers.0.mlp.down_proj \
  --target_modules model.layers.1.mlp.down_proj \
  --save_dir outputs/Llama-3.1-8B-Instruct \
  --hallu_data data/hallu.jsonl \
  --truth_data data/truth.jsonl \
  --batch_size 4 \
  --max_length 4096 \
  --oa_weight 6e-4 \
  --pruning_ratio 0.001 \
  --saliency_method second_order \
  --ranking_scope global \
  --apply_chat_template \
  --save_masks_only
```

You can also use the included example script:

```bash
bash scripts/prune_llama_3.1_8b_instruct.sh
```

The example targets every `mlp.down_proj` module in the 32-layer Llama 3.1 8B Instruct model.

## CLI Options

| Option | Description | Default |
| --- | --- | --- |
| `--model_path` | Hugging Face model ID or local model path. | Required |
| `--target_modules` | Module names to score and prune. May be passed multiple times. | Required |
| `--save_dir` | Root directory for outputs. A versioned subdirectory is created automatically. | Required |
| `--hallu_data` | JSON/JSONL file for hallucination examples. | Required |
| `--truth_data` | JSON/JSONL file for truth examples. | Required |
| `--batch_size` | Batch size for activation recording and saliency computation. | `4` |
| `--max_length` | Maximum tokenized sequence length. | `4096` |
| `--oa_weight` | Weight applied to positive over-activation scores before pruning. | `1e-2` |
| `--pruning_ratio` | Fraction of scored neurons to prune. Must be between `0` and `1`. | `0.1` |
| `--saliency_method` | Saliency method: `first_order` or `second_order`. | `second_order` |
| `--ranking_scope` | Rank pruning candidates `global`ly or `layerwise`. | `global` |
| `--apply_chat_template` | Apply the tokenizer chat template to prompts. | Off |
| `--save_masks_only` | Save `masks.pt` instead of saving the full pruned model and tokenizer. | Off |

## Outputs

Each run creates a new versioned output directory under `--save_dir`, such as:

```text
outputs/Llama-3.1-8B-Instruct/v0/
```

The output directory always contains:

- `metadata.json`: run configuration

When `--save_masks_only` is set, it also contains:

- `masks.pt`: pruning masks keyed by target module name

Without `--save_masks_only`, the CLI saves:

- the pruned model weights
- the tokenizer files

## Applying Saved Masks

Use `load_model_and_apply_masks` to apply saved masks to a model later:

```python
import torch

from oaas.utils import load_model_and_apply_masks

masks = torch.load("outputs/Llama-3.1-8B-Instruct/v0/masks.pt")

model = load_model_and_apply_masks(
    "meta-llama/Llama-3.1-8B-Instruct",
    masks,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

## Notes

- Target module names must exactly match names returned by `model.named_modules()`.
- The provided script assumes a Llama-style architecture with `model.layers.<i>.mlp.down_proj`.
- `--apply_chat_template` requires a tokenizer with a chat template.
- The current CLI is intended to be run from the repository root.
