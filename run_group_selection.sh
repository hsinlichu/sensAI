#!/bin/bash

python3 group_selection.py \
        --arch lstm\
        --dataset cifar10 \
        --ngroups 2 \
        --gpu_num 2

