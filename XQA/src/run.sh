#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
python run_squad.py --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_test \
    --train_file ../data/train/train-context-en-question-en.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../output/MLQA \
    --per_gpu_eval_batch_size 4  \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --eval_all_checkpoints \
    --seed 2020 \
    --loss_scale_1 0.1 \
    --loss_scale_2 0.1 &&

python run_squad.py --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_test \
    --train_file ../data/train/train-context-en-question-en.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../output/MLQA \
    --per_gpu_eval_batch_size 4  \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --eval_all_checkpoints \
    --seed 2030 \
    --loss_scale_1 0.1 \
    --loss_scale_2 0.1 &&

python run_squad.py --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_test \
    --train_file ../data/train/train-context-en-question-en.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../output/MLQA \
    --per_gpu_eval_batch_size 4  \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --eval_all_checkpoints \
    --seed 2040 \
    --loss_scale_1 0.1 \
    --loss_scale_2 0.1 &&

python run_squad.py --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_test \
    --train_file ../data/train/train-context-en-question-en.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../output/MLQA \
    --per_gpu_eval_batch_size 4  \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --eval_all_checkpoints \
    --seed 2050 \
    --loss_scale_1 0.1 \
    --loss_scale_2 0.1 &&

python run_squad.py --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_test \
    --train_file ../data/train/train-context-en-question-en.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../output/MLQA \
    --per_gpu_eval_batch_size 4  \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --eval_all_checkpoints \
    --seed 2060 \
    --loss_scale_1 0.1 \
    --loss_scale_2 0.1 &&

python run_squad.py --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_train \
    --do_eval \
    --do_test \
    --train_file ../data/train/train-context-en-question-en.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ../output/MLQA \
    --per_gpu_eval_batch_size 4  \
    --per_gpu_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --eval_all_checkpoints \
    --seed 2070 \
    --loss_scale_1 0.1 \
    --loss_scale_2 0.1