# sensAI_experiments
l1 norm pruning taken from https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/cifar/l1-norm-pruning

## Experiment Results for L1 Norm Weight Level Pruning + APOZ feature map pruning
Test accuracy of vgg16 (default settings from original repo) after 160 epochs: 93%. <br/>
Test accuracy after l1 norm weight pruning: 37%

Total number of params before l1 norm prune: 14.99M <br/>
Total number of params after l1 norm prune: 5.39M

Test accuracy of vgg16 after retraining for 1 epoch: 93%

Total number of params per model after apoz prune: ~3M <br/>
Test accuracy after pruning w/o retraining: 36.2%

Test accuracy after retraining binary models 1epoch: 70% <br/>
Test accuracy after retraining binary models 40epoch: 90%

## To reproduce

Perform step 0 - 2 inside l1-norm-pruning directory

Step 0)
Train base model
```shell
python main.py --dataset cifar10 --arch vgg --depth 16
```

Step 1)
Perform l1 norm pruning
```shell
python vggprune.py --dataset cifar10 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
Step 2)
Fine tune pruned model for at least 1 epoch to regain accuracy
```shell
python main_finetune.py --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 16 
```

(Inside l1-norm-pruning directory) Pruned model checkpoint should now be stored at ./logs


From l1-norm-pruning directory, copy the pruned model to parent directory (prune2).
```shell
cp -r logs ../pruned_checkpoint
```

Step 3)
Compute the feature map activations for the pruned model.
From prune2 directory,
```shell
./scripts/activations
```

Feature map activation data should now be stored at ./feature_map_data

Step 4)
Generate pruning candidates according to APOZ policy.
From prune2 directory,
```shell
rm -r prune_candidate_logs 
mkdir prune_candidate_logs
python3 apoz_policy.py
```

Pruning candidates should now be stored at ./prune_candidate_logs

Step 5)
Further prune the model according to the generated pruning candidates
From prune2 directory,
```shell
python3 prune_and_get_model.py -a vgg -r ./pruned_checkpoint -c ./prune_candidate_logs -s ./pruned2_models
```

Evaluate the pruned models to check accuracy prior to retraining
```shell
python3 evaluate.py -a vgg --test-batch 100 --pruned --resume ./pruned2_models/vgg --evaluate --binary
```

Step 6)
Retrain the model for some number of epochs
```shell
./scripts/train_binary_pruned.sh
```

Step 7)
Evaluate the retrained binary models
```shell
python3 evaluate.py -a vgg --test-batch 100 --resume ./pruned2_retrain/vgg/ --evaluate --binary --pruned --refined
```