# systme imports
import os
from datetime import datetime

# pythom imports
import numpy as np
import pickle

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as ds
import torchvision.transforms.functional as tf

# file imports
from utils import *
from plot_utils import *

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
    seed = 40
    # torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    workers = max(4 * torch.cuda.device_count(), 4)

    # name = "results/class/cifar10/svd_lgn_groups=4_kernel=6_layers=4_lam=0_lr=0.01_2023_05_16_T111412"
    # name = "results/class/mnist/lgn_groups=4_kernel=6_layers=4_lam=0_lr=0.01_2023_05_16_T233904"
    # name = "results/recon/cifar10/lgn_groups=4_kernel=6_layers=4_lam=0_lr=0.01_2023_05_17_T000621"
    name = "results/recon/mnist/lgn_groups=4_kernel=6_layers=4_lam=0_lr=0.01_2023_05_17_T002628"
    names = gen_names(None, name)

    with open(names.folder_path + "params.pckl", "rb") as file:
        params = pickle.load(file)

    os.makedirs(names.figs_path, exist_ok=True)
    os.makedirs(names.figs_path + "groups/", exist_ok=True)

    # model = LearnGroupAction_inv(params, device)
    model = LearnGroupAction_recon_inv(params, device)
    # model = LearnGroupAction(params, device)
    model.load_state_dict(torch.load(names.model_path))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for k in range(params.n_layers):
            A = model.A[k].clone().detach()
            for idx in range(params.n_groups):
                print(torch.linalg.det(A[idx]))
                print(torch.linalg.svdvals(A[idx]))
                for channel in range(params.n_channels):
                    plt.figure()
                    sns.heatmap(A[idx, channel, :, :].cpu().numpy(), cmap=cmap)#, vmin=-0.2, vmax=0.2)
                    # print("layer=" + str(layer) + ", mean_abs_diag: ", torch.diagonal(d).abs().mean())
                    # print(torch.diagonal(d))
                    plt.axis("off")
                    ax = plt.gca()
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect("equal")
                    plt.savefig(names.figs_path + "groups/GA_k=" + str(k) + "_group=" + str(idx) + "_channel=" + str(channel) + ".pdf", bbox_inches="tight")
                    plt.close()
