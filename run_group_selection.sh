#!/bin/bash

python3 rnn_group_selection.py \
        --arch lstm_cell_level \
        --resume pretrained/cifar10/checkpoint_lstm_cell_level.pth \
        --dataset cifar10 \
        --ngroups 2 \
        --gpu_num 1

