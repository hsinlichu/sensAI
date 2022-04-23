#!/bin/bash

python3 retrain_grouped_model.py \
        -a $ARCH \
        --dataset $DATASET \
        --resume ./{TO_SAVE_PRUNED_MODEL_DIR}/ \
        --train_batch $batch_size \
        --epochs $number_of_epochs \
        --num_gpus $number_of_gpus
