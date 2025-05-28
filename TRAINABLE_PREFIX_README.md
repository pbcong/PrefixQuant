# Trainable Prefix for Outlier Reduction in Quantization

## Overview

The trainable prefix feature learns optimal prefix token embeddings that minimize outliers in model activations, improving quantization quality. Unlike the original approach that selects discrete tokens, this method optimizes continuous embeddings directly.

## How It Works

### 1. Loss Function Design

The loss function targets outlier reduction through multiple components:

- **Outlier Loss**: Penalizes activations exceeding the threshold using:
  - Direct penalty for values above threshold
  - Smooth max approximation (LogSumExp) to target largest values
  - Percentile-based loss for top 1% of activations
  
- **Smoothness Loss**: Encourages smooth activation patterns by minimizing differences between consecutive positions

- **Diversity Loss**: Prevents prefix embedding collapse by penalizing high similarities between different prefix tokens

- **Magnitude Regularization**: Keeps prefix embeddings within reasonable bounds

### 2. Training Process

1. **Initialization**: The prefix embeddings are initialized using embeddings from good discrete tokens (if available) or random initialization.

2. **Forward Pass**: During training, the prefix embeddings are prepended to input sequences, and the model processes them to generate hidden states across all layers.

3. **Loss Computation**: The loss is computed on non-prefix positions across all layers (or specified target layers).

4. **Optimization**: AdamW optimizer with cosine learning rate scheduling updates the prefix embeddings to minimize outliers.

## Usage

### Basic Command

```bash
python main.py \
    --model_path <model_path> \
    --trainable_prefix \
    --prefix_len 8 \
    --prefix_lr 1e-3 \
    --prefix_epochs 20 \
    --outlier_threshold 64 \
    --wbits 4 \
    --input_bits 8 \
    ...other quantization args...
```

### Key Arguments

- `--trainable_prefix`: Enable trainable prefix training
- `--prefix_len`: Number of prefix tokens to learn (default: 8)
- `--prefix_lr`: Learning rate for prefix training (default: 1e-3)
- `--prefix_epochs`: Number of training epochs (default: 20)
- `--outlier_threshold`: Threshold for considering activations as outliers (default: 64)

## Implementation Details

### Device Management

- The model is automatically moved to GPU if available for efficient training
- Prefix embeddings are kept on the same device as the model
- After training, the model is moved back to its original device

### Loss Weights

The total loss combines different objectives:
```
total_loss = 1.0 * outlier_loss + 0.1 * smoothness_loss + 0.05 * diversity_loss + 0.01 * magnitude_loss
```

### Best Model Selection

The training process tracks the best loss and restores the best prefix embeddings at the end.

### Conversion to Discrete Tokens

After training, the continuous embeddings are converted back to discrete tokens by finding the nearest tokens in the vocabulary using cosine similarity.

## Advantages

1. **Direct Optimization**: Directly optimizes for outlier reduction instead of relying on heuristic token selection
2. **Multi-Layer Awareness**: Considers outliers across all layers simultaneously
3. **Smooth Optimization**: Uses differentiable approximations for better gradient flow
4. **Regularization**: Built-in regularization prevents degenerate solutions

## Example Results

The trainable prefix method typically achieves:
- Lower maximum activation values
- More uniform activation distributions
- Better quantization accuracy
- Improved perplexity after quantization

## Tips for Best Results

1. **Learning Rate**: Start with 1e-3 and adjust based on loss curves
2. **Prefix Length**: 8-16 tokens usually work well; longer may not improve results
3. **Outlier Threshold**: Set based on your quantization bit-width (lower bits need lower thresholds)
4. **Epochs**: 20-30 epochs are usually sufficient; monitor for overfitting
5. **Target Layers**: Consider focusing on layers with most outliers for efficiency \

Experiments:

10 prefix tokens trained
python main.py --model_path meta-llama/Llama-2-7b-hf --model_name Llama-2-7b-hf --save_quant_dir ./quantized_models/llama2-7b-w4a4kv4-trained-prefix --output_dir ./logs/llama7-7b-w4a4kv4-trained-prefix --cache_dir ./cache --calib_dataset wikitext2 --train_size 512 --val_size 64 --training_seqlen 1024 --ppl_seqlen 2048 --seed 42 --wbits 4 --w_group_size 128 --input_bits 4 --input_mode static --k_bits 4 --v_bits 4 --kv_group_size 128 --kv_mode static --pre_rotate --rotate_mode hadamard --down_online_had --qk_online_had --trainable_prefix --prefix_len 8 --prefix_lr 1e-3 --prefix_epochs 8 --outlier_threshold 64 --batch_size 1 --activation_clipping --mse_init --epochs 10 --quant_lr 5e-5 --weight_lr 5e-6  --eval_ppl --eval_tasks "winogrande,hellaswag,arc_challenge,arc_easy,piqa,gsm8k" --eval_batch_size 4 --max_memory 65GiB


3 prefix tokens trained
CUDA_VISIBLE_DEVICES=1 python main.py --model_path meta-llama/Llama-2-7b-hf --model_name Llama-2-7b-hf --save_quant_dir ./quantized_models/llama2-7b-w4a4kv4-trained-prefix --output_dir ./logs/llama7-7b-w4a4kv4-trained-prefix --cache_dir ./cache --calib_dataset wikitext2 --train_size 512 --val_size 64 --training_seqlen 1024 --ppl_seqlen 2048 --seed 42 --wbits 4 --w_group_size 128 --input_bits 4 --input_mode static --k_bits 4 --v_bits 4 --kv_group_size 128 --kv_mode static --pre_rotate --rotate_mode hadamard --down_online_had --qk_online_had --trainable_prefix --prefix_len 3 --prefix_lr 1e-3 --prefix_epochs 8 --outlier_threshold 64 --batch_size 4 --activation_clipping --mse_init --epochs 10 --quant_lr 5e-5 --weight_lr 5e-6  --eval_ppl --eval_tasks "winogrande,hellaswag,arc_challenge,arc_easy,piqa,gsm8k" --eval_batch_size 4 --max_memory 65GiB
