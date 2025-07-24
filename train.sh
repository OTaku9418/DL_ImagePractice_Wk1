#!/bin/bash

python main.py \
    --mode both \
    --epochs 10 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --weight_decay 1e-4 \
    --validation_split 0.1 \
    --save_every 10 \
    --device auto \
    --num_workers 0

echo "finish！"
