import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.utils import data
from torch.autograd import Variable
import torch.nn as nn

from datasets import utils
from typing import List, Tuple

import glob
import unicodedata
import string
import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

class nameLanTrainingSetWrapper(utils.DataSetWrapper):
    def __init__(self, class_group: Tuple[int], negative_samples=False):
        dataset = TextDataset('data/nameLan/names/',isTest=False)
        super().__init__(dataset, class_group, negative_samples, 'nameLan')

class nameLanTestingSetWrapper(utils.DataSetWrapper):
    def __init__(self, class_group: Tuple[int], negative_samples=False):
        dataset = TextDataset('data/nameLan/names/',isTest=True)
        super().__init__(dataset, class_group, negative_samples, 'nameLan')

class TextDataset(data.Dataset):
	def __init__(self,data_dir,isTest):
		self.all_letters = string.ascii_letters + " .,;'-"
		self.n_letters = len(self.all_letters)
		self.category_lines = {}
		self.all_categories = []
		self.data_dir = data_dir
		self.isTest = isTest
		self.line_category_set = []
		self.rand_seed = 17
		self.train_val_percent = 0.8
		self.max_name_len = 30

		self.load_files()
		self.splitTrainVal()
		self.shuffle_set()

		self.targets = [self.all_categories.index(d['category']) for d in self.line_category_set]

	def load_files(self):
		for filename in self.findFiles(self.data_dir+'*.txt'):
		    category = filename.split('/')[-1].split('.')[0]
		    self.all_categories.append(category)
		    lines = self.readLines(filename)
		    # shuffle
		    np.random.seed(self.rand_seed)
		    idxs = np.arange(len(lines))
		    np.random.shuffle(idxs)
		    self.category_lines[category] = [lines[i] for i in idxs]

		self.n_categories = len(self.all_categories)

	def splitTrainVal(self):
		for category in self.all_categories:
			lines = self.category_lines[category]
			n_lines = int(len(lines)*self.train_val_percent)
			if self.isTest:
				lines = lines[n_lines:]
			else:
				lines = lines[:n_lines]
			for line in lines:
				self.line_category_set.append({
					"name":line,
					"category":category
					})

	def shuffle_set(self):
		random.shuffle(self.line_category_set)

	def __len__(self):
		return len(self.line_category_set)
		

	def findFiles(self,path): return glob.glob(path)

	# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
	def unicodeToAscii(self,s):
	    return ''.join(
	        c for c in unicodedata.normalize('NFD', s)
	        if unicodedata.category(c) != 'Mn'
	        and c in self.all_letters
	    )

	# Read a file and split into lines
	def readLines(self,filename):
	    lines = open(filename).read().strip().split('\n')
	    return [self.unicodeToAscii(line) for line in lines]

	# Find letter index from all_letters, e.g. "a" = 0
	def letterToIndex(self,letter):
	    return self.all_letters.find(letter)

	# Turn a line into a <line_length x 1 x n_letters>,
	# or an array of one-hot letter vectors
	# def lineToTensor(self,line):
	#     tensor = torch.zeros(self.max_name_len, self.n_letters)
	#     for li, letter in enumerate(line):
	#         tensor[li][self.letterToIndex(letter)] = 1
	#     return tensor
	def lineToTensor(self,line):
	    tensor = torch.zeros(len(line), self.n_letters)
	    for li, letter in enumerate(line):
	        tensor[li][self.letterToIndex(letter)] = 1
	    return tensor

	def categoryFromOutput(self,output):
	    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
	    category_i = top_i[0][0]
	    return self.all_categories[category_i], category_i


	def __getitem__(self, index):
		line_cat = self.line_category_set[index]
		line = line_cat["name"]
		category = line_cat["category"]
		category_tensor = Variable(torch.LongTensor([self.all_categories.index(category)]))
		line_tensor = Variable(self.lineToTensor(line))
		return line_tensor, category_tensor