"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
torchvision
"""
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
from torch import nn
from datasets.FordA import FordADataset
import matplotlib.pyplot as plt
import models.cifar.rnn as rnn

import logging
from datetime import datetime
logger = logging.getLogger()

parser = argparse.ArgumentParser(
    description='FordA Generate Group Info')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--comment', type=str, default="test")
args = parser.parse_args()

logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

comment = "{}_{}".format(str(datetime.now().strftime(r'%m%d_%H%M%S')), args.comment)
resultDirPath = Path("log") / comment
resultDirPath.mkdir(parents=True, exist_ok=True)

fileHandler = logging.FileHandler(resultDirPath / "info.log")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 500             # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 5e-2                # learning rate

logger.info("Epoch: {} | batch size: {} | LR: {}".format(EPOCH, BATCH_SIZE, LR))

trainset = FordADataset(path='./data/FordA/FordA_TRAIN.tsv', train=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=args.workers)

testset = FordADataset(path='./data/FordA/FordA_TEST.tsv', train=False, mean=trainset.mean, std=trainset.std)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=args.workers)

model = rnn.lstm(num_classes=2)
logger.info(model)
model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
best_test_loss = 1e10
for epoch in range(EPOCH):
    # training 
    model.train()
    trange = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train|Epoch {}".format(epoch))
    train_total = 0
    train_correct = 0
    train_loss = 0
    for step, (inputs, labels) in trange:
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()                           # clear gradients for this training step
        outputs = model(inputs)                         # rnn output
        loss = loss_func(outputs, labels)               # cross entropy loss
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss += loss.item() * labels.size(0)
        trange.set_postfix(loss=loss.item(), Acc=train_correct / train_total)
    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total

    model.eval()
    test_total = 0
    test_correct = 0
    test_loss = 0
    with torch.no_grad():
        trange = tqdm(enumerate(test_loader), total=len(test_loader), desc="Train|Epoch {}".format(epoch))
        for step, (inputs, labels) in trange:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)                               # rnn output
            loss = loss_func(outputs, labels)                   # cross entropy loss

            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_loss += loss.item() * labels.size(0)
            trange.set_postfix(loss=loss.item(), Acc=test_correct / test_total)
    epoch_test_loss = test_loss / test_total
    epoch_test_acc = test_correct / test_total

    if epoch_test_loss < best_test_loss:
        best_test_loss = epoch_test_loss
        filename = resultDirPath / "best_checkpoint.pth.tar"
        saved_data = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                }
        torch.save(saved_data, filename)
        logger.info("Current Best(loss: {:.4f}, acc: {:.2f}) Save to: {}".format(epoch_test_loss, epoch_test_acc, filename))

    logger.info('Epoch: {}'.format(epoch))
    logger.info('Train | loss: {:.4f} | accuracy {:.2f}'.format(epoch_train_loss, epoch_train_acc))
    logger.info('Test | loss: {:.4f} | accuracy {:.2f}'.format(epoch_test_loss, epoch_test_acc))
