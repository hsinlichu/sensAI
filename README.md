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

step 3:  Prune model
```
python3 rnn_prune_and_get_model.py \
        -a lstm_cell_level \
        --dataset cifar10  \
        --resume pretrained/cifar10/checkpoint_lstm_cell_level.pth\
        -c prune_candidate_logs/ \
        -s pruned_models/
        
```


step 4:  retrain pruned models
```
python3 retrain_grouped_model.py \
        -a lstm_cell_level \
        --dataset cifar10  \
        --resume pruned_models/ \
        --train_batch 1000 \
        --epochs 25 \
        --num_gpus 2
```


step 5:  evaluate
```
python3 evaluate.py \
        -a lstm_cell_level \
        --dataset=cifar10  \
        --retrained_dir pruned_models_retrained/ \
        --test-batch 1000
```



# 文本分类

我目前写了一个文本分类的RNN，放在models/text/rnn.py里。
目前train下来准确率能上75%，目前还在训练中。
要复现的化，首先建一个叫data的folder，然后到这个链接下载压缩包（https://download.pytorch.org/tutorial/data.zip

之后把 data.zip解压到这个data folder下面，并将解压出来的文件夹重命名为nameLan

命令行里输入“python pretrain.py -d nameLan -a rnn”，就可以用这个nameLan数据集train RNN了

Step 1: train完之后，输入以下命令可以group selection:
```
python3 rnn_group_selection.py \
        --arch rnn\
        --resume pretrained/nameLan/checkpoint_rnn.pth \
        --dataset nameLan \
        --ngroups 2 \
        --gpu_num 1
```




