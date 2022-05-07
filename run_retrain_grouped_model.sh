#!/bin/bash

python3 retrain_grouped_model.py \
        -a lstm_cell_level \
        --dataset cifar10  \
        --resume pruned_models/ \
        --train_batch 1000 \
        --epochs 25 \
        --num_gpus 1
