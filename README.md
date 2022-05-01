# sensAI
sensAI: ConvNets Decomposition via Class Parallelism for Fast Inference on Live Data

step 1: pretrain一个lstm_cell_level
```
python pretrain.py -d cifar10 -a lstm_cell_level
```

step 2: group selection and get pruned candidates
```
python3 rnn_group_selection.py \
        --arch lstm_cell_level \
        --resume pretrained/cifar10/checkpoint_lstm_cell_level.pth \
        --dataset cifar10 \
        --ngroups 2 \
        --gpu_num 2
        
```
