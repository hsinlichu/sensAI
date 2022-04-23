#!/bin/bash

python3 group_selection.py \
        --arch resnet110 \
        --dataset cifar10 \
        --ngroups 2 \
        --gpu_num 2

