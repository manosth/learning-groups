# system imports
import os
import time
from datetime import datetime
from skimage import io
import math

# pythom imports
import numpy as np
import scipy.io as sio
import argparse

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, random_split
from torch.nn.modules.utils import _pair, _quadruple

import torchvision
import torchvision.datasets as ds
import torchvision.transforms.functional as tf

from model import *

def log_gradients_in_model(model, logger, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)

class custom_loss(torch.nn.Module):
    def __init__(self, lam_loss, base_loss=torch.nn.MSELoss()):
        super(custom_loss, self).__init__()
        self.base_loss = base_loss
        self.lam_loss = lam_loss

    def forward(self, x, out, code, params):
        base_loss = self.base_loss(x, out)
        # l1_loss = self.lam_loss * code.abs().sum(dim=1).mean()
        # the thing below looks digusting, but basically you take the norm over each filtermap
        # (you could also do the sum, this is personal preference I think, there is no consensus).
        # If there are no filtermaps and we have a dense model, then this will just give the absolute
        # value. Then, you have to take the norm over the group (in which case, if it is a sparse model
        # then this dimension will be one and again nothing will happen). Finally, you take the sum over
        # the number of groups, finalizing the L2,1 norm (and in the case of sparsity just L1). Then, you just take the mean over the batch.
        orig_shape = code.shape
        code = code.view(orig_shape[0], params.num_groups, params.group_size, orig_shape[2], orig_shape[3])
        l1_loss = self.lam_loss * code.abs().sum(dim=(-2, -1)).norm(dim=2).sum(dim=1).mean()
        loss = base_loss + l1_loss
        return loss

def report_statistics(start, idx, total_len, val=0.0):
    current = time.time()
    total = current - start
    seconds = int(total % 60)
    minutes = int((total // 60) % 60)
    hours = int((total // 60) // 60)

    if idx == -1:
        print("")
        print(f"Total time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
    else:
        remain = (total_len - idx - 1) / (idx + 1) * total
        seconds_r = int(remain % 60)
        minutes_r = int((remain // 60) % 60)
        hours_r = int((remain // 60) // 60)
        print(f"progress: {(idx + 1) / total_len * 100:5.2f}%\telapsed: {hours:02d}:{minutes:02d}:{seconds:02d}\tremaining: {hours_r:02d}:{minutes_r:02d}:{seconds_r:02d}\tval: {val}", end="\r")

class Params():
        def __init__(self, params):
            # optimizer and training params
            self.lr = 1e-2
            self.eps = 1e-8

            self.batch_size = 128
            self.epochs = 300

            self.tensorboard = True
            self.dataset = "cifar"

            # model params
            self.model = "final_suul" # sparse, untied, whitened/unormalized, learned bias
            self.group_size = 4
            self.num_groups = 5
            self.kernel_size = 6
            self.stride = 1

            self.num_layers = 4

            self.step = 0.01
            self.lmbd = 0

            self.lam_loss = 0

            # data params
            self.n_classes = 10
            self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

            self.n_channels = 3
            self.input_size = 1024
            self.input_width = 32
            self.input_height = 32

            # classification params
            self.pool_size = 4

# self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
def parse_args():
    parser = argparse.ArgumentParser()
    # training params
    parser.add_argument('--lr', type=float, default=1e-2, help='optimizer learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='optimizer epsilon')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--tensorboard', type=bool, default=True, help='use tensorboard')
    parser.add_argument('--comments', type=str, default=None, help='comments to prepend to the folder')

    # data params
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar10c', 'cifar10g', 'mnist', 'rmnist', 'fmnist'], help='testing dataset')
    parser.add_argument('--experiment', type=str, default='class', choices=['class', 'recon', 'rot_p'], help='experiment')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--n_channels', type=int, default=3, help='number of channels')
    parser.add_argument('--input_size', type=int, default=1024, help='size of the input images')
    parser.add_argument('--input_width', type=int, default=32, help='width of the input images')
    parser.add_argument('--input_height', type=int, default=32, help='height of the input images')
    parser.add_argument('--norm', type=str, default="whiten", choices=['whiten', 'standard', 'none'], help='data normalization')

    # architecture params
    parser.add_argument('--model', type=str, default="lgn", help='model name')
    parser.add_argument('--n_groups', type=int, default=5, help='number of groups')
    parser.add_argument('--group_size', type=int, default=4, help='group size')
    parser.add_argument('--kernel_size', type=int, default=6, help='kernel size')
    parser.add_argument('--stride', type=int, default=1, help='convolution stride')
    parser.add_argument('--n_layers', type=int, default=4, help='number of layers')
    parser.add_argument('--eta', type=float, default=0.01, help='unfolding step size')
    parser.add_argument('--lmbd', type=float, default=0, help='lambda for soft thresholding')
    parser.add_argument('--pool_size', type=int, default=4, help='pooling size for classification')

    args = parser.parse_args()

    return args

class Names():
        def __init__(self, params, folder=None):
            if folder is None:
                if params.comments is None:
                    name = str(params.model) + "_groups=" + str(params.group_size) + "_kernel=" + str(params.kernel_size) + "_layers=" + str(params.n_layers) + "_lam=" + str(params.lmbd) + "_lr=" + str(params.lr)
                else:
                    name = params.comments + "_" + str(params.model) + "_groups=" + str(params.group_size) + "_kernel=" + str(params.kernel_size) + "_layers=" + str(params.n_layers) + "_lam=" + str(params.lmbd) + "_lr=" + str(params.lr)

                date = datetime.now().strftime("%Y_%m_%d_T%H%M%S")
                model = name + "_" + date
                path = "results/" + params.experiment + "/" + params.dataset + "/"
            else:
                model = folder.split("/")[3]
                path = "results/" + folder.split("/")[1] + "/" + folder.split("/")[2] + "/"

            # if params.n_channels > 1:
            #     path += "color_"
            # else:
            #     path += "_"

            # if params.kernel_size < params.input_width:
            #     path += "conv_"

            self.model = model
            self.folder_path = path + model + "/"
            self.model_path = path + model + "/" + model + ".pth"
            self.figs_path = path + model + "/figs/"

def gen_names(params, folder=None):
    names = Names(params, folder=folder)
    return names

def gen_loaders(params, workers, patch=False):
    if params.dataset == "mnist":
        X_tr, Y_tr, X_te, Y_te = load_mnist(params)
    elif params.dataset == "rmnist":
        X_tr, Y_tr, X_te, Y_te = load_rmnist(params)
    elif params.dataset == "cifar10":
        if params.n_channels > 1:
            X_tr, Y_tr, X_te, Y_te = load_cifar(params, color=True, patch=patch)
        else:
            X_tr, Y_tr, X_te, Y_te = load_cifar(params, patch=patch)
    else:
        raise Exception("Dataset not implemeneted")
    train_dl = make_loader(TensorDataset(X_tr, Y_tr), batch_size=params.batch_size, num_workers=workers)
    test_dl = make_loader(TensorDataset(X_te, Y_te), batch_size=params.batch_size, num_workers=workers)
    return train_dl, test_dl

def gen_model(params, device, init_B=None):
    # for now, just hack it out
    # return LearnGroupAction_patch(params, device).to(device)
    # return LearnGroupAction(params, device).to(device)
    # return LearnGroupAction_inv(params, device).to(device)
    return LearnGroupAction_recon_inv(params, device).to(device)

def standardize(X, mean=None, std=None):
    "Expects data in NxCxWxH."
    if mean is None:
        mean = X.mean(axis=(0,2,3))
        std = X.std(axis=(0,2,3))

    X = torchvision.transforms.Normalize(mean, std)(X)
    return X, mean, std

def whiten(X, zca=None, mean=None, eps=1e-8):
    "Expects data in NxCxWxH."
    os = X.shape
    X = X.reshape(os[0], -1)

    if zca is None:
        mean = X.mean(dim=0)
        cov = np.cov(X, rowvar=False)
        U, S, V = np.linalg.svd(cov)
        zca = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))
    X = torch.Tensor(np.dot(X - mean, zca.T).reshape(os))
    return X, zca, mean

def load_mnist(params, datadir="~/data", five_digits=False):
    train_ds = ds.MNIST(root=datadir, train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    test_ds = ds.MNIST(root=datadir, train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    def to_xy(dataset):
        Y = dataset.targets.long()
        # this size is necessary to work with the matmul broadcasting when using channels
        X = dataset.data.view(dataset.data.shape[0], params.n_channels, params.input_width, -1) / 255.0
        return X, Y

    def get_five_digits(X, Y):
        digit_0 = (Y == 0)
        digit_3 = (Y == 3)
        digit_4 = (Y == 4)
        digit_6 = (Y == 6)
        digit_7 = (Y == 7)
        indexes = digit_0 | digit_3 | digit_4 | digit_6 | digit_7
        return X[indexes], Y[indexes]

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)

    X_tr, mean, std = standardize(X_tr)
    X_te, _, _ = standardize(X_te, mean, std)

    X_tr, zca, mean = whiten(X_tr)
    X_te, _, _ = whiten(X_te, zca, mean)

    return X_tr, Y_tr, X_te, Y_te

def load_rmnist(params, datadir="/home/manos/data/rmnist/data.mat"):
    data = sio.loadmat(datadir)
    X_tr = torch.Tensor(data['x'])
    Y_tr = torch.Tensor(data['y']).squeeze().long()

    X_te = torch.Tensor(data['x_test'])
    Y_te = torch.Tensor(data['y_test']).squeeze().long()

    X_tr = standardize(X_tr)
    X_te = standardize(X_te)

    X_tr = whiten(X_tr)
    X_te = whiten(X_te)

    return X_tr, Y_tr, X_te, Y_te

def load_cifar(params, datadir='~/data', three_class=False, color=False, patch=False):
    train_ds = ds.CIFAR10(root=datadir, train=True,
                           download=True, transform=None)
    test_ds = ds.CIFAR10(root=datadir, train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        Y = torch.Tensor(np.array(dataset.targets)).long()
        X = torch.Tensor(np.transpose(dataset.data, (0, 3, 1, 2))).float() / 255.0 # [0, 1]
        if not color:
            X = torchvision.transforms.Grayscale()(X).view(X.shape[0], 1, X.shape[2], -1)
        return X, Y

    def get_three_classes(X, Y):
        cats = (Y == 3)
        horses = (Y == 7)
        boats = (Y == 8)

        indexes = cats | horses | boats
        return X[indexes], Y[indexes]

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)

    if params.norm == "standrad":
        X_tr, mean, std = standardize(X_tr)
        X_te, _, _ = standardize(X_te, mean, std)
    elif params.norm == "whiten":
        X_tr, mean, std = standardize(X_tr)
        X_te, _, _ = standardize(X_te, mean, std)

        X_tr, zca, mean = whiten(X_tr)
        X_te, _, _ = whiten(X_te, zca, mean)

    if patch:
        angle = 60
        med_filt = MedianPool2d(6, same=True)
        # kernel_size = 3
        # stride = 1
        # ih, iw = params.input_height, params.input_width
        # if ih % 1 == 0:
        #     ph = max(kernel_size - stride, 0)
        # else:
        #     ph = max(kernel_size - (ih % stride), 0)
        # if iw % stride == 0:
        #     pw = max(kernel_size - stride, 0)
        # else:
        #     pw = max(kernel_size - (iw % stride), 0)
        # pl = pw // 2
        # pr = pw - pl
        # pt = ph // 2
        # pb = ph - pt
        # padding = (pl, pt)
        # med_filt = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        X_patch_tr = torch.zeros(X_tr.shape[0], X_tr.shape[1], params.kernel_size, params.kernel_size)
        Y_patch_tr = torch.zeros(X_tr.shape[0], X_tr.shape[1], params.kernel_size, params.kernel_size)
        for idx in range(X_tr.shape[0]):
            rand_x = np.random.choice(X_tr.shape[2] - params.kernel_size)
            rand_y = np.random.choice(X_tr.shape[3] - params.kernel_size)
            X_patch_tr[idx] = X_tr[idx, :, rand_x:rand_x+params.kernel_size, rand_y:rand_y+params.kernel_size]
            Y_patch_tr[idx] = med_filt(X_patch_tr[idx].unsqueeze(0)).squeeze()
            Y_patch_tr[idx] = tf.rotate(Y_patch_tr[idx], angle)


        X_patch_te = torch.zeros(X_te.shape[0], X_te.shape[1], params.kernel_size, params.kernel_size)
        Y_patch_te = torch.zeros(X_te.shape[0], X_te.shape[1], params.kernel_size, params.kernel_size)
        for idx in range(X_te.shape[0]):
            rand_x = np.random.choice(X_te.shape[2] - params.kernel_size)
            rand_y = np.random.choice(X_te.shape[3] - params.kernel_size)
            X_patch_te[idx] = X_te[idx, :, rand_x:rand_x+params.kernel_size, rand_y:rand_y+params.kernel_size]
            Y_patch_te[idx] = med_filt(X_patch_te[idx].unsqueeze(0)).squeeze()
            Y_patch_te[idx] = tf.rotate(Y_patch_te[idx], angle)
        X_tr = X_patch_tr
        Y_tr = Y_patch_tr
        X_te = X_patch_te
        Y_te = Y_patch_te
    return X_tr, Y_tr, X_te, Y_te

def make_loader(dataset, shuffle=True, batch_size=128, num_workers=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).mean(dim=-1)[0]
        return x
