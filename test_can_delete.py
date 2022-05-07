from torchvision import datasets

DATASET_ROOT = 'data/'
d = datasets.CIFAR10(root=DATASET_ROOT, train=True,
                                   download=True)
print(d.targets)
