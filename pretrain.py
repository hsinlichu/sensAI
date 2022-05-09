import argparse
import pickle

from datasets.nameLan import TextDataset

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import load_model
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import subprocess as sp
import os
import time

from even_k_means import kmeans_lloyd
import models.cifar as cifar_models
from models.cifar.rnn import lstm_cell_level
from models.text.rnn import RNN



BATCH_SIZE = 64

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/100/Imagenet Generate Group Info')
# Datasets
parser.add_argument('-d', '--dataset', required=True, type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='/home/ubuntu/imagenet', required=False, type=str,
                    help='location of the imagenet dataset that includes train/val')
# Architecture
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet20',
                    #choices=load_model.model_arches('cifar'),
                    help='model architecture: ' +
                    ' | '.join(load_model.model_arches('cifar')) +
                    ' (default: resnet18)')      
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')  
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')

# Miscs
parser.add_argument('--seed', type=int, default=42, help='manual seed')
args = parser.parse_args()
use_cuda = torch.cuda.is_available() and True

# Random seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

max_epoch = 500



def main():
    print('==> Preparing dataset %s' % args.dataset)
    best_prec1 = 0
    img_width = 28
    # cifar10/100 group selection
    if args.dataset in ['cifar10', 'cifar100']:
        if args.dataset == 'cifar10':
            dataset_loader = datasets.CIFAR10
        elif args.dataset == 'cifar100':
            dataset_loader = datasets.CIFAR100
        img_width=32*3

        pretrained_model_path = "pretrained/"
        save_dir = pretrained_model_path+args.dataset
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = save_dir+'/'

        train_dataset = dataset_loader(
            root='data/',
            download=True,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
        val_dataset = dataset_loader(
            root='data/',
            download=True,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=args.workers,
            pin_memory=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=args.workers,
            pin_memory=False)

        
        model = cifar_models.__dict__[args.arch](num_classes=len(train_dataset.classes))
        if args.resume!="":
            if os.path.exists(args.resume):
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['state_dict'])

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), 0.02,
                                momentum=args.lr,
                                weight_decay=5e-4)
        
        model = model.cuda()


        for epoch in range(max_epoch):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(save_dir, 'checkpoint_{}.pth'.format(args.arch)))
    elif args.dataset == 'nameLan':
        # load dataset
        trainset = TextDataset('data/nameLan/names/',isTest=False)
        testset = TextDataset('data/nameLan/names/',isTest=True)
        BATCH_SIZE = 1
        train_loader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
        val_loader = torch.utils.data.DataLoader(testset,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
        print("nameLan loaded!")
        # create model
        if args.arch == 'rnn':
            model = RNN(input_size=trainset.n_letters,output_size=trainset.n_categories).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), 0.005)

        pretrained_model_path = "pretrained/"
        save_dir = pretrained_model_path+args.dataset
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_dir = save_dir+'/'
        if args.resume!="":
            if os.path.exists(args.resume):
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['state_dict'])
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Start training ... ")
        for epoch in range(max_epoch):
            # adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train_rnn(model, train_loader, criterion, optimizer,epoch)

            # evaluate on validation set
            prec1 = validate_rnn(model, val_loader)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(save_dir, 'checkpoint_{}.pth'.format(args.arch)))





def train_rnn(model, trainloader, criterion, optimizer,epoch):
    top1 = AverageMeter()
    losses = AverageMeter()
    model.train()
    for idx, batch in enumerate(trainloader):
        line_tensor, category_tensor = batch
        target = torch.reshape(category_tensor, (-1,)).cuda()
        optimizer.zero_grad()
        output = model(line_tensor.cuda())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        output = output.float()
        loss = loss.float()
        prec1 = accuracy(output.data.cpu(), category_tensor)[0]
        losses.update(loss.item(), category_tensor.size(0))
        top1.update(prec1.item(), category_tensor.size(0))

    print('Epoch: [{0}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  epoch, loss=losses, top1=top1))

def validate_rnn(model, testloader):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    for idx, batch in enumerate(testloader):
        category_tensor, line_tensor = batch
        with torch.no_grad():
            output, hidden = model(line_tensor.cuda())

        output = output.float()
        prec1 = accuracy(output.data.cpu(), category_tensor)[0]
        top1.update(prec1.item(), category_tensor.size(0))
    print('test * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
