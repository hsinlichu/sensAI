import argparse
import os
import shutil
import time
import random
import warnings
import threading



from tqdm import tqdm
import torch
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from utils import Logger, AverageMeter, accuracy, savefig
from torch.utils.data import Dataset, DataLoader
import glob
import re
import itertools
from compute_flops import print_model_param_flops
import torchvision.models as models
from imagenet_evaluate_grouped import main_worker
import torch.multiprocessing as mp
import load_model

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names += ["resnet110", "resnet164", "mobilenetv2", "shufflenetv2", "lstm_cell_level", "gru_cell_level"]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100/ImageNet Testing')
# Checkpoints
parser.add_argument('--retrained_dir', type=str, metavar='PATH',
                    help='path to the directory of pruned models (default: none)')
# Checkpoints
parser.add_argument('--resume', type=str, metavar='PATH',
                    help='path to the directory of pretrained models (default: none)')

# Datasets
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--data', metavar='DIR', required=False,
                    help='path to imagenet dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--bce', default=False, action='store_true',
                    help='Use binary cross entropy loss')
best_acc1 = 0

# Miscs
parser.add_argument('--seed', type=int, default=42, help='manual seed')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'imagenet', 'Dataset can only be cifar10, cifar100 or imagenet.'

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

torch.set_printoptions(threshold=10000)

def main():
    # imagenet evaluation
    if args.dataset == 'imagenet':
        imagenet_evaluate()
        return
    
    # cifar 10/100 evaluation
    print('==> Preparing dataset %s' % args.dataset)
    if args.dataset == 'cifar10':
        dataset_loader = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        dataset_loader = datasets.CIFAR100
    else:
        raise NotImplementedError

    trainDataset = dataset_loader(
            root='data',
            download=False,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
    testDataset = dataset_loader(
            root='data',
            download=False,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
    trainloader = data.DataLoader(
        trainDataset,
        batch_size = args.test_batch,
        shuffle = True,
        num_workers = args.workers)

    testloader = data.DataLoader(
        testDataset,
        batch_size = args.test_batch,
        shuffle = True,
        num_workers = args.workers)

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    if args.pretrained:
        model = load_model.load_pretrain_model(args.arch, 'cifar', args.resume, len(trainDataset.classes), use_cuda)
        if use_cuda:
            model = model.cuda()
    else:
        model = load_pruned_models(args.retrained_dir+'/'+args.arch+'/')
    print(model)

    if not args.pretrained:
        if len(model.group_info) == 10 and args.dataset == 'cifar10':
            args.bce = True

    print("On training set:")
    train_acc = test_list(trainloader, model, criterion, use_cuda)
    print("On testing set:")
    test_acc = test_list(testloader, model, criterion, use_cuda)

def imagenet_evaluate():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def test_list(testloader, model, criterion, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    moving_times = AverageMeter()
    computation_times = AverageMeter()
    addition_times = AverageMeter()


    model.eval()
    start = end = time.time()

    if args.dataset == 'cifar10':
        confusion_matrix = np.zeros((10, 10))
    elif args.dataset == 'cifar100':
        confusion_matrix = np.zeros((100, 100))
    else:
        raise NotImplementedError

    bar = tqdm(total=len(testloader))
    # pdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        bar.update(1)
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            targets = targets.cuda()
            inputs = inputs.cuda()

        with torch.no_grad():
            if args.pretrained:
                outputs, computation_time = model(inputs, output_time=True)
                computation_times.update(computation_time)
            else:
                outputs, (moving_time, computation_time, addition_time) = model(inputs, output_time=True)
                moving_times.update(moving_time)
                computation_times.update(computation_time)
                addition_times.update(addition_time)

            loss = criterion(outputs, targets)
            for output, target in zip(outputs, targets):
                gt = target.item()
                dt = np.argmax(output.cpu().numpy())
                confusion_matrix[gt, dt] += 1
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.set_description('({batch}/{size}) Data: {data:.4f} | Computation: {computation:.4f} | Moving: {moving:.4f} | Addition: {addition:.4f} | Batch: {batch_time:.4f} | Total: {total:.4f} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.sum,
            batch_time=batch_time.sum,
            computation=computation_times.sum,
            moving=moving_times.sum,
            addition=addition_times.sum,
            total=time.time() - start,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        ))
    bar.close()

    np.set_printoptions(precision=3, linewidth=96)

    '''
    print("\n===== Full Confusion Matrix ==================================\n")
    if confusion_matrix.shape[0] < 20:
        print(confusion_matrix)
    else:
        print("Warning: The original confusion matrix is too big to fit into the screen. "
              "Skip printing the matrix.")

    if all([len(group) > 1 for group in model.group_info]):
        print("\n===== Inter-group Confusion Matrix ===========================\n")
        print(f"Group info: {[group for group in model.group_info]}")
        n_groups = len(model.group_info)
        group_confusion_matrix = np.zeros((n_groups, n_groups))
        for i in range(n_groups):
            for j in range(n_groups):
                cols = model.group_info[i]
                rows = model.group_info[j]
                group_confusion_matrix[i, j] += confusion_matrix[cols[0], rows[0]]
                group_confusion_matrix[i, j] += confusion_matrix[cols[0], rows[1]]
                group_confusion_matrix[i, j] += confusion_matrix[cols[1], rows[0]]
                group_confusion_matrix[i, j] += confusion_matrix[cols[1], rows[1]]
        group_confusion_matrix /= group_confusion_matrix.sum(axis=-1)[:, np.newaxis]
        print(group_confusion_matrix)

    print("\n===== In-group Confusion Matrix ==============================\n")
    for group in model.group_info:
        print(f"group {group}")
        inter_group_matrix = confusion_matrix[group, :][:, group]
        inter_group_matrix /= inter_group_matrix.sum(axis=-1)[:, np.newaxis]
        print(inter_group_matrix)
    '''
    return (losses.avg, top1.avg)

class GroupedModel(nn.Module):
    def __init__(self, model_list, group_info):
        super().__init__()
        self.group_info = group_info
        # flatten list of list
        permutation_indices = list(itertools.chain.from_iterable(group_info))
        self.permutation_indices = torch.eye(len(permutation_indices))[permutation_indices]
        if use_cuda:
            self.permutation_indices = self.permutation_indices.cuda()
        self.model_list = nn.ModuleList(model_list)
        for i, model in enumerate(model_list):
            print(group_info[i])
            print(model.valid_timestep)
            print("Pruned {} timestamps".format(32 - len(model.valid_timestep)))

    def forward(self, inputs, output_time=False):
        moving_time = 0
        computation_time = 0
        addition_time = 0
        output_list = []
      
        if args.bce:
            for model_idx, model in enumerate(self.model_list):
                inputs = inputs.cuda(model_idx)
                output = model(inputs)[:, 0]
                output_list.append(output)
            output_list = torch.softmax(torch.stack(output_list, dim=1).squeeze(), dim=1)
        else:
            lock = threading.Lock()
            output_list = [None for i in range(len(self.model_list))]
            #grad_enabled = torch.is_grad_enabled()

            def _worker(i, module, input, device):
                # torch.set_grad_enabled(grad_enabled)
                with torch.no_grad():
                    try:
                        with torch.cuda.device(device):
                            output = module(input)
                        output = torch.softmax(output, dim=1)[:, 1:].cuda(0)
                        with lock:
                            output_list[i] = output
                    except Exception as e:
                        with lock:
                            output_list[i] = e

            moving_start = time.time()
            data = [inputs.clone().cuda(i) for i in range(len(self.model_list))]
            torch.cuda.synchronize()
            moving_time += time.time() - moving_start

            threads = [threading.Thread(target=_worker,
                                        args=(i, model, data[i], i))
                       for i, model in enumerate(self.model_list)]
            start = time.time()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            computation_time = time.time() - start
            '''

            data = [inputs.clone().cuda(i) for i in range(len(self.model_list))]
            for model_idx, model in enumerate(self.model_list): 

                moving_start = time.time()
                inputs = inputs.cuda(model_idx)
                moving_time += time.time() - moving_start

                computation_start = time.time()
                output = model(data[model_idx])
                #output = model(inputs)
                computation_time += time.time() - computation_start

                addition_start = time.time()
                output = torch.softmax(output, dim=1)[:, 1:]
                addition_time += time.time() - addition_start

                moving_start = time.time()
                output = output.cuda(0)
                moving_time += time.time() - moving_start

                addition_start = time.time()
                output_list.append(output)
                addition_time += time.time() - addition_start
            '''
            '''
            moving_start = time.time()
            data = [inputs.clone().cuda(i) for i in range(len(self.model_list))]
            a0 = torch.zeros(inputs.size()).cuda(0)
            a1 = torch.zeros(inputs.size()).cuda(1)
            moving_time += time.time() - moving_start

            computation_start = time.time()
            
            output0 = self.model_list[0](a0)
            output1 = self.model_list[1](a1)
            #output0 = self.model_list[0](data[0])
            #output1 = self.model_list[1](data[1])

            #output = model(inputs)
            computation_time += time.time() - computation_start

            addition_start = time.time()
            output0 = torch.softmax(output0, dim=1)[:, 1:]
            output1 = torch.softmax(output1, dim=1)[:, 1:]
            addition_time += time.time() - addition_start

            moving_start = time.time()
            output1 = output1.cuda(0)
            moving_time += time.time() - moving_start

            addition_start = time.time()
            output_list.append(output0)
            output_list.append(output1)
            addition_time += time.time() - addition_start
            '''

            addition_start = time.time()
            output_list = torch.cat(output_list, 1)
            addition_time += time.time() - addition_start

        addition_start = time.time()
        final_output = torch.mm(output_list, self.permutation_indices)
        addition_time += time.time() - addition_start
        
        if output_time:
            return final_output, (moving_time, computation_time, addition_time)
        return final_output

    def print_statistics(self):
        num_params = []
        num_flops = []

        print("\n===== Metrics for grouped model ==========================\n")

        for group_id, model in zip(self.group_info, self.model_list):
            n_params = sum(p.numel() for p in model.parameters()) / 10**6
            num_params.append(n_params)
            print(f'Grouped model for Class {group_id} '
                  f'Total params: {n_params:2f}M')
            num_flops.append(print_model_param_flops(model, 32))

        print(f"Average number of flops: {sum(num_flops) / len(num_flops) / 10**9 :3f} G")
        print(f"Average number of param: {sum(num_params) / len(num_params)} M")


def load_pruned_models(model_dir):
    group_dir = model_dir[:-(len(args.arch)+1)]
    if not model_dir.endswith('/'):
        model_dir += '/'
    file_names = [f for f in glob.glob(model_dir + "*.pth", recursive=False)]
    model_list = [torch.load(file_name, map_location=lambda storage, loc: storage.cuda(i)) for i, file_name in enumerate(file_names)]
    groups = np.load(open(group_dir + "grouping_config.npy", "rb"))
    group_info = []
    for file in file_names:
        group_id = filename_to_index(file)
        print(f"Group number is: {group_id}")
        class_indices = groups[group_id]
        group_info.append(class_indices.tolist()[0])
    model = GroupedModel(model_list, group_info)
    #model.print_statistics()
    return model


def filename_to_index(filename):
    filename = [int(s) for s in filename.split('_') if s.isdigit()]
    return filename

if __name__ == '__main__':
    main()


