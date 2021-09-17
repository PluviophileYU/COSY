#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7
python run_squad.py --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_eval \
    --do_test \
    --dev_file ../data/dev/dev-context-en-question-en.json \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../output/MLQA \
    --per_gpu_eval_batch_size 128 \
    --fp16 \
    --eval_all_checkpoints