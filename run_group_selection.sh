#!/bin/bash

python3 group_selection.py \
        --arch lstm\
        --dataset cifar10 \
        --ngroups 2 \
        --gpu_num 2 \
	--resume log/0425_214518_relu/best_checkpoint.pth.tar
