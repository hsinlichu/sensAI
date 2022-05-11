# LSTM for Name Classification Dataset

The model is inside models/cifar/rnn.py 

First create a folder called 'data/', and then download the zip from here: https://download.pytorch.org/tutorial/data.zip
Unzip data.zip and rename it nameLan, and put it under the data/ folder.


step 1: pretrain the lstm model (The pretrained acc is about 79%)
```
python pretrain.py -d nameLan -a lstm_cell_level
```

step 2: group selection and get pruned candidates (ngroups=1 if you only want one group)
```
python3 rnn_group_selection.py \
        -alstm_cell_level \
        --resume pretrained/nameLan/checkpoint_lstm_cell_level.pth \
        -d nameLan \
        --ngroups 2 
```

step 3:  Prune model
```
python3 rnn_prune_and_get_model.py \
        -a lstm_cell_level \
        -d nameLan \
        --resume pretrained/cifar10/checkpoint_lstm_cell_level.pth\
        -c prune_candidate_logs/ \
        -s pruned_models/
        
```


step 4:  retrain pruned models
```
python3 retrain_grouped_model.py \
        -a lstm_cell_level \
        -d nameLan  \
        --resume pruned_models/ \
        --train_batch 1 \
        --epochs 25 
```


step 5:  evaluate
```
python3 evaluate.py \
        -a lstm_cell_level \
        -d nameLan  \
        --retrained_dir pruned_models_retrained/ \
        --test-batch 1
```


