import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import _pickle as cPickle
import numpy as np
import random
from operator import itemgetter
import os

LIMIT_SIZE = 2500
CIFAR10_NUM_CLASSES = 10

# download CIFAR10
def prepare_cifar10(path):
    CIFAR10(root=path, download=True)

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)

# Limit the size of traing data of the specefied classes
def limit_specified_classes(X_data, y_data, classes, size=2500):
    idxes = []
    for i in range(0, 10):
        idx = np.where(y_data == i)[0]
        if i in classes:
            idx = np.random.choice(idx, size=(size), replace=False)
        idxes.extend(idx)

    ret_X, ret_y = X_data[idxes], y_data[idxes]
    return ret_X, ret_y

# Weighted sampler for sampling from inbalanced data
def set_weighted_sampler(label_data):
    class_count = [len(np.where(label_data == i)[0]) for i in range(CIFAR10_NUM_CLASSES)]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    
    class_weights[3] *= 1.5
    
    class_weights_all = class_weights[label_data]
    
    # print(class_weights)
    # print(label_data)
    # print(class_weights_all, len(class_weights_all))
    
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    ) 
    return weighted_sampler

def load_cifar10(data_dir):
    special_classes = [2, 4, 9] # bird, deer, truck class labels

    for i in range(1, 6):
        fname = os.path.join(data_dir, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            X_train = data_dict['data']
            y_train = data_dict['labels']
        else:
            X_train = np.vstack((X_train, data_dict['data']))
            y_train = np.hstack((y_train, data_dict['labels']))

    data_dict = unpickle(os.path.join(data_dir, 'test_batch'))
    X_test = data_dict['data']
    y_test = np.array(data_dict['labels'])

    X_train = [conv_data2image(x) for x in X_train]
    X_test = [conv_data2image(x) for x in X_test]
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    X_train, y_train = limit_specified_classes(X_train, y_train, classes=special_classes)
    return X_train, y_train, X_test, y_test

class MyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        super(Dataset, self).__init__()
        self.transform = transform
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        # preprocess image
        if self.transform:
            x = self.transform(x)    

        return x, y

# if __name__ == "__main__":
#     X_train, y_train, _, _ = load_cifar10("data/cifar-10-batches-py")
#     print(X_train.shape, y_train.shape)
#     print(np.where(y_train == 2))