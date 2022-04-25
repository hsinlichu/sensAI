"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
matplotlib
torchvision
"""
import argparse
from tqdm import tqdm
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import models.cifar.rnn as rnn

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/100/Imagenet Generate Group Info')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 100               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
LR = 0.01               # learning rate

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

trainset = dsets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=args.workers)

testset = dsets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=args.workers)

model = rnn.lstm(num_classes=10)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    # training 
    model.train()
    trange = tqdm(enumerate(train_loader), total=len(train_loader), desc="Train|Epoch {}".format(epoch))
    train_total = 0
    train_correct = 0
    train_loss = 0
    for step, (inputs, labels) in trange:
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = inputs.contiguous().view(-1, 32, 32 * 3)

        outputs = model(inputs)                         # rnn output
        loss = loss_func(outputs, labels)               # cross entropy loss
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss += loss.item() * labels.size(0)
        trange.set_postfix(loss=loss.item(), Acc=train_correct / train_total)

    model.eval()
    test_total = 0
    test_correct = 0
    test_loss = 0
    with torch.no_grad():
        trange = tqdm(enumerate(test_loader), total=len(test_loader), desc="Train|Epoch {}".format(epoch))
        for step, (inputs, labels) in trange:
            inputs = inputs.permute(0, 2, 3, 1)
            inputs = inputs.contiguous().view(-1, 32, 32 * 3)

            outputs = model(inputs)                               # rnn output
            loss = loss_func(outputs, labels)                   # cross entropy loss

            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            test_loss += loss.item() * labels.size(0)
            trange.set_postfix(loss=loss.item(), Acc=test_correct / test_total)
    print('Epoch: {}'.format(epoch))
    print('Train | loss: {:.4f} | accuracy {:.2f}'.format(train_loss / train_total, train_correct / train_total))
    print('Test | loss: {:.4f} | accuracy {:.2f}'.format(test_loss / test_total, test_correct / test_total))
