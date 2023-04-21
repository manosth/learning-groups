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
import torchvision.transforms.functional as tf

# file imports
from utils import *
from plot_utils import *
from config_plot import Params

import matplotlib.pyplot as plt
from matplotlib import rc, cm
import seaborn as sns
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
sns.set_theme()
sns.set_context('paper')
sns.set(font_scale=1.4)
cmap = sns.color_palette("coolwarm", as_cmap=True)

cmap_red = sns.cubehelix_palette(reverse=True, rot=.6, as_cmap=True)
cmap_green = sns.cubehelix_palette(reverse=True, rot=-.6, as_cmap=True)
cmap_blue = sns.cubehelix_palette(reverse=True, rot=-.2, as_cmap=True)

if __name__ == '__main__':
    params = Params()

    seed = 40
    # torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    workers = max(4 * torch.cuda.device_count(), 4)

    name = "final_suul_groups=4_kernel=6_stride=1_layers=4_step=0.01_lam=0_lamloss=0_lr=0.01_2023_04_13_T222729"
    folder_path = "results/cifarcolor_conv_" + name
    model_path = folder_path + "/" + name + ".pth"
    figs_path = folder_path + "/figs/"
    os.makedirs(figs_path, exist_ok=True)
    os.makedirs(figs_path + "actions/", exist_ok=True)

    model = LearnGroupAction(params, device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    names = gen_names(params)

    layer = 3
    group = 3
    model.eval()
    with torch.no_grad():
        for k in range(params.num_layers):
            gen_b = model.B[k].clone().detach()
            res = model.B[k].clone().detach()
            A = model.A[k].clone().detach()
            for idx in range(1, params.group_size):
                shp = model.B[k].shape
                res = A @ res.view(shp[0], shp[1], -1, 1)
                gen_b = torch.cat((gen_b, res.view(shp)), dim=0)
            gen_b = gen_b.data.cpu().numpy()
            save_conv_dictionary(gen_b, params, 0, k, figs_path + "actions/", names)
        A = model.A[layer].clone().detach()
        B = model.B[layer].clone().detach()

        A_group = A[group, :, :, :]
        B_group = B[group, :, :, :]

        # save inputs
        plt.figure()
        to_plot = B_group.cpu().numpy().transpose((1,2,0))
        to_plot -= np.min(to_plot)
        to_plot /= np.max(to_plot)
        plt.imshow(to_plot)
        plt.axis("off")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.savefig(figs_path + "actions/input_k=" + str(layer) + "_group=" + str(group) + ".pdf", bbox_inches="tight")
        plt.close()

        plt.figure()
        sns.heatmap(B_group[0, :, :].cpu().numpy(), cmap=cmap_red)
        plt.axis("off")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.savefig(figs_path + "actions/input_k=" + str(layer) + "_group=" + str(group) + "_channel=0R.pdf", bbox_inches="tight")
        plt.close()

        plt.figure()
        sns.heatmap(B_group[1, :, :].cpu().numpy(), cmap=cmap_green)
        plt.axis("off")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.savefig(figs_path + "actions/input_k=" + str(layer) + "_group=" + str(group) + "_channel=1G.pdf", bbox_inches="tight")
        plt.close()

        plt.figure()
        sns.heatmap(B_group[2, :, :].cpu().numpy(), cmap=cmap_blue)
        plt.axis("off")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.savefig(figs_path + "actions/input_k=" + str(layer) + "_group=" + str(group) + "_channel=2B.pdf", bbox_inches="tight")
        plt.close()

        # group action
        shp = B.shape
        res = (A @ B.view(shp[0], shp[1], -1, 1)).view(shp)

        # save outputs
        plt.figure()
        sns.heatmap(res[group, 0, :, :].cpu().numpy(), cmap=cmap_red)
        plt.axis("off")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.savefig(figs_path + "actions/output_k=" + str(layer) + "_group=" + str(group) + "_channel=0R.pdf", bbox_inches="tight")
        plt.close()

        plt.figure()
        sns.heatmap(res[group, 1, :, :].cpu().numpy(), cmap=cmap_green)
        plt.axis("off")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.savefig(figs_path + "actions/output_k=" + str(layer) + "_group=" + str(group) + "_channel=1G.pdf", bbox_inches="tight")
        plt.close()

        plt.figure()
        sns.heatmap(res[group, 2, :, :].cpu().numpy(), cmap=cmap_blue)
        plt.axis("off")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.savefig(figs_path + "actions/output_k=" + str(layer) + "_group=" + str(group) + "_channel=2B.pdf", bbox_inches="tight")
        plt.close()

        plt.figure()
        to_plot = res[group, :, :, :].cpu().numpy().transpose((1,2,0))
        to_plot -= np.min(to_plot)
        to_plot /= np.max(to_plot)
        plt.imshow(to_plot)
        plt.axis("off")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.savefig(figs_path + "actions/output_k=" + str(layer) + "_group=" + str(group) + ".pdf", bbox_inches="tight")
        plt.close()
