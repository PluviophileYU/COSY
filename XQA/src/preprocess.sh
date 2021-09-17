#!/usr/bin/env bash

python run_squad.py --model_type bert \
    --model_name_or_path bert-base-multilingual-cased \
    --do_preprocess \
    --train_file ../data/train/train-context-en-question-en.json \
    --max_seq_length 384 \
    --doc_stride 128
