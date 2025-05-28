#!/bin/bash

# Example script to demonstrate trainable prefix for quantization

MODEL_PATH="meta-llama/Llama-2-7b-hf"  # Replace with your model path
OUTPUT_DIR="./log/trainable_prefix_test"
SAVE_DIR="./quantized_models/llama2-7b-w4a8-trainable-prefix"

# Run quantization with trainable prefix
python main.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --save_quant_dir $SAVE_DIR \
    --trainable_prefix \
    --prefix_len 8 \
    --prefix_lr 1e-3 \
    --prefix_epochs 20 \
    --outlier_threshold 64 \
    --wbits 4 \
    --w_group_size 128 \
    --input_bits 8 \
    --input_mode dynamic \
    --k_bits 8 \
    --v_bits 8 \
    --kv_group_size 128 \
    --kv_mode dynamic \
    --mse_init \
    --epochs 3 \
    --train_size 512 \
    --val_size 64 \
    --training_seqlen 1024 \
    --batch_size 4 \
    --quant_lr 5e-5 \
    --weight_lr 5e-6 \
    --eval_ppl \
    --eval_tasks "piqa,arc_easy,arc_challenge,hellaswag,winogrande" \
    --calib_dataset pile \
    --seed 42 