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

# 文本分类

我目前写了一个文本分类的RNN，放在models/text/rnn.py里。
目前train下来准确率能上75%，目前还在训练中。
要复现的化，首先建一个叫data的folder，然后到这个链接下载压缩包（https://download.pytorch.org/tutorial/data.zip

之后把 data.zip解压到这个data folder下面，并将解压出来的文件夹重命名为nameLan

命令行里输入“python pretrain.py -d nameLan -a rnn”，就可以用这个nameLan数据集train RNN了

train完之后，输入以下命令可以group selection:
```
python3 rnn_group_selection.py \
        --arch rnn\
        --resume pretrained/nameLan/checkpoint_rnn.pth \
        --dataset nameLan \
        --ngroups 2 \
        --gpu_num 1
```



