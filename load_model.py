import os
import torch

import models.cifar as cifar_models
import models.text as text_models


def model_arches(dataset):
    if dataset == 'cifar':
        return sorted(name for name in cifar_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar_models.__dict__[name]))
    elif dataset == 'nameLan':
        return sorted(name for name in text_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(text_models.__dict__[name]))
    else:
        raise NotImplementedError



def load_pretrain_model(arch, dataset, resume_checkpoint, num_classes, use_cuda, input_size=None):
    #assert os.path.isfile(resume_checkpoint), 'Error: no checkpoint found!'
    if(os.path.isfile(resume_checkpoint)):
        print('==> Resuming from checkpoint..')
        print(resume_checkpoint)
        if use_cuda:
            checkpoint = torch.load(resume_checkpoint)
        else:
            checkpoint = torch.load( 
                resume_checkpoint, map_location=torch.device('cpu'))
    if arch.startswith('lstm'):
        if dataset.startswith('cifar'):
            model = cifar_models.__dict__[arch](num_classes=num_classes)  
        elif dataset == 'nameLan':
            model = cifar_models.__dict__[arch](input_size=input_size, num_classes=num_classes, dataset=dataset)
    elif arch == 'RNN':
        model = text_models.__dict__[arch](input_size, num_classes)  
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}.")
    print(model)

    if use_cuda:
        model.cuda()
    state_dict = {}
    # deal with old torch version
    if(os.path.isfile(resume_checkpoint)):
        if arch != 'mobilenetv2' and arch != 'shufflenetv2':
            for k, v in checkpoint['state_dict'].items():
                state_dict[k.replace('module.', '')] = v
            model.load_state_dict(state_dict)
        else:
            for k, v in checkpoint['net'].items():
                state_dict[k.replace('module.', '')] = v
            model.load_state_dict(state_dict)
    return model
