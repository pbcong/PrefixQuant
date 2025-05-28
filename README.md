# PrefixQuant
Official PyTorch implement for [PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization](https://arxiv.org/abs/2410.05265). 



## News

[2025/1] Support the learnable activation cliping for dynamic quantization.

[2024/10] We release PrefixQuant, the first work to let static activation quantization outperforms dynamic ones in LLM. We only open the fake quantization code now, and the inference kernels will be released later.

## Contents
- [Installation](#Installation)
- [Quantization](#quantization)
- [Inference](#Inference)
- [Citation](#citation)


## Installation
```
conda create -n prefixquant python==3.9

conda activate prefixquant

pip install -r requirements.txt
```

## Quantization
We provide an example command to quantized `Llama-3-8B` without fine-tuning:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  \
--model_name DeepSeek-R1-Distill-Qwen-7B \
--output_dir ./log/deepseek-qwen-7b-w4a4kv4 \
--wbits 4 \
--input_bits 4 \
--input_mode static \
--v_bits 4 \
--k_bits 4 \
--kv_group_size 128 \
--kv_mode static \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande \
--save_quant_dir ./pre_quantized_models/deepseek-qwen-7b-w4a4kv4
```
You can find the detailed fine-tuning setting in the paper. There are some useful information as follows:
- For dynamic quantization, you should add `--activation_clipping` to enhance the perfomance.
- You can add `--epochs 20` to introduce fine-tuning for W4A4KV4 quantization, and `--epochs 10` for W4A8KV4 quantization. 
- For Llama-3-70B(-Instruct) models, you should change the default learning rate to `--quant_lr 2e-5 --weight_lr 2e-6`. 
- For Llama-2-70B, you should set `--loss_type skip_mse` for the training stability.

## Inference
We provide an example command to evaluate the quantize models:
```
CUDA_VISIBLE_DEVICES=0 python eval.py \
--quant_model ./pre_quantized_models/llama-3-8b-w4a4kv4 \
--eval_ppl \
--eval_tasks  piqa,arc_easy,arc_challenge,hellaswag,winogrande
```

## Plot Activation Distribution
We provide an example command to visualize token-wsie maximum values for linear inputs:
```
CUDA_VISIBLE_DEVICES=0 python plot_activation.py \
--model_path path/to/llama-2-7b \
--model_name llama-2-7b \
--plot_linear_input
```
You can add `--pre_rotate --down_online_had --qk_online_had` to apply hadamard rotation, and add `--set_prefixed_tokens` to set the proposed prefixed tokens in our paper.
Additionally, you can also change `--plot_linear_input` to other plotting choices, details are as follows:
- `--plot_linear_output`: plot token-wsie maximum values for linear outputs (such as Q/K/V).
- `--plot_outlier_token_position`: count the token index of outlier tokens.
- `--plot_outlier_token`: count the token content of outlier tokens
- `--plot_layer_wise_outlier_token_number`: plot layer-wise outlier token number
- `--plot_layer_input_3d` : plot the 3D image of layer inputs.
- `--plot_block_output_3d` : plot the 3D image of block outputs.

More examples can be found in `./examples/plot.sh`.


## Citation
If you use our PrefixQuant approach in your research, please cite our paper:
```
@article{prefixquant,
  title={PrefixQuant: Eliminating Outliers by Prefixed Tokens for Large Language Models Quantization},
  author={Chen, Mengzhao and  Liu, Yi and Wang, Jiahao and Bin, Yi and Shao, Wenqi and Luo, Ping},
  journal={arXiv preprint arXiv:2410.05265},
  year={2024}
}
```

# PrefixQuant: Trainable Prefix Integration

## Example Command with All Optimizations Including Trainable Prefix

Here's a comprehensive command that utilizes all performance optimizations including the new trainable prefix functionality:

```bash
python main.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --save_quant_dir ./quantized_models/llama2-7b-w4g128-trainable-prefix \
    --output_dir ./logs/llama2-7b-trainable-prefix \
    --cache_dir ./cache \
    --calib_dataset c4 \
    --train_size 512 \
    --val_size 64 \
    --training_seqlen 1024 \
    --ppl_seqlen 2048 \
    --seed 42 \
    --wbits 4 \
    --w_group_size 128 \
    --w_asym \
    --input_bits 8 \
    --input_group_size 64 \
    --input_mode dynamic \
    --input_asym \
    --k_bits 8 \
    --v_bits 8 \
    --kv_group_size 64 \
    --kv_mode dynamic \
    --kv_asym \
    --pre_rotate \
    --rotate_mode hadamard \
    --down_online_had \
    --qk_online_had \
    --trainable_prefix \
    --prefix_len 8 \
    --prefix_lr 1e-3 \
    --prefix_epochs 20 \
    --prefix_loss_type l2 \
    --activation_clipping \
    --mse_init \
    --mse_init_size 8 \
    --epochs 10 \
    --quant_lr 5e-5 \
    --weight_lr 5e-6 \
    --min_lr_factor 10 \
    --clip_grad 0.3 \
    --batch_size 4 \
    --loss_type mse \
    --training_target fp_input \
    --eval_ppl \
    --eval_tasks "winogrande,hellaswag,arc_challenge,arc_easy,piqa" \
    --eval_batch_size 16 \
    --max_memory 65GiB
```

to train the prefix:
python main.py \
    --model_path meta-llama/Meta-Llama-3-8B \
    --model_name Llama-3-8b \
    --save_quant_dir ./quantized_models/llama3-8b-w4a4kv4-trained-prefix \
    --output_dir ./logs/llama3-8b-w4a4kv4-trained-prefix \
    --cache_dir ./cache \
    --calib_dataset wikitext2 \
    --train_size 512 \
    --val_size 64 \ 
    --training_seqlen 1024 \ 
    --ppl_seqlen 2048 \
    --seed 42 \
    --wbits 4 \
    --w_group_size 128 \
    --w_asym \
    --input_bits 8 \
    --input_group_size 64 \
    --input_mode dynamic \
    --input_asym \
    --k_bits 8 \
    --v_bits 8 \
    --kv_group_size 64 \
    --kv_mode dynamic \
    --kv_asym \
    --pre_rotate \
    --rotate_mode hadamard \
    --down_online_had \
    --qk_online_had \
    --trainable_prefix \
    --prefix_len 8 \
    --prefix_lr 1e-3 \
    --prefix_epochs 20 \
    --outlier_threshold 64 \
    --batch_size 4 \
    --activation_clipping \
    --mse_init \
    --mse_init_size 8 \
    --epochs 10 \
    --quant_lr 5e-5 \
    --weight_lr 5e-6 \
    --min_lr_factor 10 \
    --clip_grad 0.3 \
    --loss_type mse \
    --training_target fp_input \
    --eval_ppl \
    --eval_tasks "winogrande,hellaswag,arc_challenge,arc_easy,piqa,gsm8k" \
    --eval_batch_size 16 \
    --max_memory 65GiB
    
python main.py --model_path meta-llama/Llama-2-7b-hf --model_name Llama-2-7b-hf --save_quant_dir ./quantized_models/llama2-7b-w4a4kv4-trained-prefix --output_dir ./logs/llama7-7b-w4a4kv4-trained-prefix --cache_dir ./cache --calib_dataset wikitext2 --train_size 512 --val_size 64 --training_seqlen 1024 --ppl_seqlen 2048 --seed 42 --wbits 4 --w_group_size 128 --w_asym --input_bits 8 --input_group_size 64 --input_mode dynamic --input_asym --k_bits 8 --v_bits 8 --kv_group_size 64 --kv_mode dynamic --kv_asym --pre_rotate --rotate_mode hadamard --down_online_had --qk_online_had --trainable_prefix --prefix_len 8 --prefix_lr 1e-3 --prefix_epochs 20 --outlier_threshold 64 --batch_size 4 --activation_clipping --mse_init --mse_init_size 8 --epochs 10 --quant_lr 5e-5 --weight_lr 5e-6 --min_lr_factor 10 --clip_grad 0.3 --loss_type mse --training_target fp_input --eval_ppl --eval_tasks "winogrande,hellaswag,arc_challenge,arc_easy,piqa,gsm8k" --eval_batch_size 16 --max_memory 65GiB
    

## Key Features Enabled:

1. **Trainable Prefix (`--trainable_prefix`)**: Learns optimal prefix embeddings instead of using discrete tokens
2. **Rotation (`--pre_rotate`, `--down_online_had`, `--qk_online_had`)**: Hadamard rotation for better quantization
3. **Quantization Training (`--epochs 10`)**: Fine-tunes quantization parameters
4. **MSE Initialization (`--mse_init`)**: Optimal quantizer initialization
5. **Activation Clipping (`--activation_clipping`)**: Layer-wise activation clipping
6. **Multi-bit Quantization**: 4-bit weights, 8-bit activations and KV cache
7. **Comprehensive Evaluation**: Both perplexity and downstream tasks

## Trainable Prefix Parameters:

- `--prefix_len 8`: Number of learnable prefix tokens
- `--prefix_lr 1e-3`: Learning rate for prefix embeddings
- `--prefix_epochs 20`: Training epochs for prefix optimization
- `--prefix_loss_type l2`: Loss function type (l2 or var)

The trainable prefix will automatically initialize using good discrete tokens (if available) and then optimize them to flatten attention outputs for better quantization performance.