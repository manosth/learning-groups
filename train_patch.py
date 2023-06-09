# systme imports
import os
from datetime import datetime
import time

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

if __name__ == '__main__':
    params = parse_args()
    params.experiment = "rot_p"

    # create model and loaders
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    workers = max(4 * torch.cuda.device_count(), 4)

    names = gen_names(params)
    model = gen_model(params, device)
    train_dl, test_dl = gen_loaders(params, workers, patch=True)

    # housekeeping params
    times_per_epoch = 10
    report_period = len(train_dl) // times_per_epoch
    plot_period = 5

    train_log = names.folder_path + "trainlog.txt"
    test_log = names.folder_path + "testlog.txt"
    train_acc_log = names.folder_path + "trainclass.txt"
    test_acc_log = names.folder_path + "testclass.txt"

    os.makedirs(names.folder_path)
    with open(names.folder_path + 'params.txt', 'w') as file:
        file.write(str(vars(params)))

    with open(names.folder_path + 'params.pckl', 'wb') as file:
        pickle.dump(params, file)

    filters_path = names.figs_path + "filters/"
    os.makedirs(filters_path, exist_ok=True)

    if params.tensorboard:
        writer = SummaryWriter(names.folder_path)
        writer.add_text("params", str(vars(params)), global_step=0)

    # training params
    opt = optim.Adam(model.parameters(), lr=params.lr, eps=params.eps)
    schd = optim.lr_scheduler.MultiStepLR(opt, [int(1/2 * params.epochs), int(3/4 * params.epochs), int(7/8 * params.epochs)], gamma=1/2)
    loss_func = torch.nn.MSELoss()

    # training
    total_train = params.epochs * (len(train_dl) + len(test_dl))
    start = time.time()
    local = time.localtime()
    print(f"Starting iterations...\t(Start time: {local[3]:02d}:{local[4]:02d}:{local[5]:02d})")
    for epoch in range(1, params.epochs + 1):
        net_loss = 0.0
        n_total = 0

        model.train()
        for idx, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_func(out, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            # log_gradients_in_model(model, writer, epoch * len(train_dl) + idx)
            opt.step()

            net_loss += loss.item() * len(x)
            n_total += len(x)
            with torch.no_grad():
                model.normalize()

            if idx % report_period == 0:
                train_loss = net_loss / n_total
                curr_train = (epoch - 1) * (len(train_dl) + len(test_dl)) + idx
                # reset the buffer
                report_statistics(start, curr_train, total_train, val="")
                report_statistics(start, curr_train, total_train, val=np.round(train_loss, 4))
        train_loss = net_loss / n_total

        net_loss = 0.0
        n_total = 0
        model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dl):
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = loss_func(out, y)

                net_loss += loss.item() * len(x)
                n_total += len(x)

                if idx % report_period == 0:
                    test_loss = net_loss / n_total
                    curr_train = epoch * len(train_dl) + (epoch - 1) * len(test_dl) + idx
                    report_statistics(start, curr_train, total_train, val=np.round(test_loss, 4))
        test_loss = net_loss / n_total

        with open(train_log, "a") as file:
            file.write(str(train_loss) + "\n")

        with open(test_log, "a") as file:
            file.write(str(test_loss) + "\n")

        if params.tensorboard:
            writer.add_scalar("classifier_test_loss", test_loss, epoch + 1)
            writer.add_scalar("classifier_train_loss", train_loss, epoch + 1)
        schd.step()
    report_statistics(start, -1, total_train)
    if params.tensorboard:
        writer.close()

    # save model for visualization afterwards
    torch.save(model.state_dict(), names.model_path)
