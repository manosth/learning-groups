# systme imports
import os
from datetime import datetime

# pythom imports
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as ds

# file imports
from utils import *
from plot_utils import *
from config_enc import Params

import matplotlib.pyplot as plt
from matplotlib import rc, cm
import seaborn as sns
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
sns.set_theme()
sns.set_context('paper')
sns.set(font_scale=1.4)
# cmap = sns.cubehelix_palette(reverse=True, rot=-.2, as_cmap=True)
cmap = sns.color_palette("coolwarm", as_cmap=True)
color_plot = sns.cubehelix_palette(4, reverse=True, rot=-.2)

if __name__ == '__main__':
    params = Params()

    seed = 42
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    workers = max(4 * torch.cuda.device_count(), 4)

    name = "suwl_groups=4_kernel=6_stride=1_layers=4_step=0.01_lam=0_lamloss=0_lr=0.01_2023_04_13_T101656"
    folder_path = "results/cifarcolor_conv_" + name
    model_path = folder_path + "/" + name + ".pth"
    figs_path = folder_path + "/figs/"
    os.makedirs(figs_path, exist_ok=True)
    os.makedirs(figs_path + "encodings/", exist_ok=True)

    model = LearnGroupAction(params, device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    _, test_dl = gen_loaders(params, workers)

    model.eval()
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)

            out, enc = model(x)

            to_save = 5
            save_conv_encoding(enc[to_save, :, :, :], params, figs_path + "encodings/", to_save, cmap=cmap)
            save_img(x[to_save, :, :], params, figs_path + "encodings/", to_save)
            break
