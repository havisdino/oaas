num_layers=32
target_modules=()

for ((i=0; i<=N; i++)); do
    target_modules+=(--target_modules "model.layers.$i.mlp.down_proj")
done

python -m oaas.cli \
    --model_path meta/Llama-3.1-8B-Instruct \
    ${target_modules[@]} \
    --save_dir outputs/Llama-3.1-8B-Instruct \
    --hallu_data data/hallu.jsonl \
    --truth_data data/truth.jsonl \
    --batch_size 4 \
    --max_length 4096 \
    --oa_weight 1e-2 \
    --pruning_ratio 0.05 \
    --apply_chat_template \
    --save_masks_only
