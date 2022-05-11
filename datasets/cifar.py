import os
import pickle
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from typing import List, Tuple

from datasets import utils


# Transformations
RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([RC, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])


DATASET_ROOT = './data/'


class Cifar8(datasets.CIFAR10):
    def __init__(
        self,
        root,
        train = True,
        transform = None,
        target_transform = None,
        download = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []


        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.array(self.targets)

        index = self.targets != 8  
        self.targets = self.targets[index]
        self.data = self.data[index]
        index = self.targets != 9 
        self.targets = self.targets[index]
        self.data = self.data[index]

        self._load_meta()
        self.classes = set(self.targets)




class CIFAR10TrainingSetWrapper(utils.DataSetWrapper):
    def __init__(self, class_group: Tuple[int], negative_samples=False):
        #dataset = datasets.CIFAR10(root=DATASET_ROOT, train=True,
        #                           download=True, transform=transform_with_aug)
        dataset = Cifar8(root=DATASET_ROOT, train=True,
                                   download=True, transform=transform_with_aug)

        super().__init__(dataset, class_group, negative_samples)


class CIFAR10TestingSetWrapper(utils.DataSetWrapper):
    def __init__(self, class_group: Tuple[int], negative_samples=False):
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=False,
                                   download=True, transform=transform_no_aug)
        super().__init__(dataset, class_group, negative_samples)


class CIFAR100TrainingSetWrapper(utils.DataSetWrapper):
    def __init__(self, class_group: Tuple[int], negative_samples=False):
        dataset = datasets.CIFAR100(root=DATASET_ROOT, train=True,
                                    download=True, transform=transform_with_aug)
        super().__init__(dataset, class_group, negative_samples)


class CIFAR100TestingSetWrapper(utils.DataSetWrapper):
    def __init__(self, class_group: Tuple[int], negative_samples=False):
        dataset = datasets.CIFAR100(root=DATASET_ROOT, train=False,
                                    download=True, transform=transform_no_aug)
        super().__init__(dataset, class_group, negative_samples)


