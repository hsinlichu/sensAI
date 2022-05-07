#!/bin/bash

python3 rnn_prune_and_get_model.py \
        -a lstm_cell_level \
        --dataset cifar10  \
        --resume pretrained/cifar10/checkpoint_lstm_cell_level.pth\
        -c prune_candidate_logs/ \
        -s pruned_models/
