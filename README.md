# RNN for Name Classification Dataset

The model is inside models/text/rnn.py 

First create a folder called 'data/', and then download the zip from here: https://download.pytorch.org/tutorial/data.zip. \
Unzip data.zip and rename it nameLan, and put it under the data/ folder.


step 1: pretrain the RNN model (The pretrained acc is about 75%)
```
python pretrain.py -d nameLan -a RNN
```

step 2: group selection and get pruned candidates (ngroups=1 if you only want one group)
```
python3 rnn_group_selection.py \
        -a RNN \
        --resume pretrained/nameLan/checkpoint_RNN.pth \
        -d nameLan \
        --ngroups 2 
```

step 3:  Prune model
```
python3 rnn_prune_and_get_model.py \
        -a RNN \
        -d nameLan \
        --resume pretrained/cifar10/checkpoint_RNN.pth\
        -c prune_candidate_logs/ \
        -s pruned_models/
        
```


step 4:  retrain pruned models
```
python3 retrain_grouped_model.py \
        -a RNN \
        -d nameLan  \
        --resume pruned_models/ \
        --train_batch 1 \
        --epochs 25 
```


step 5:  evaluate
```
python3 evaluate.py \
        -a RNN \
        -d nameLan  \
        --retrained_dir pruned_models_retrained/ \
        --test-batch 1
