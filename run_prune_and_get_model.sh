#!/bin/bash

python3 prune_and_get_model.py \
        -a $ARCH \
        --dataset $DATASET \
        --resume $pretrained_model \
        -c ./prune_candidate_logs/ \
        -s ./{TO_SAVE_PRUNED_MODEL_DIR}/
