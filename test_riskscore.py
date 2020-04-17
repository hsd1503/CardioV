"""
test on physionet data

The search stratagy is based on: 
Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). 
Designing Network Design Spaces. Retrieved from http://arxiv.org/abs/2003.13678

Shenda Hong, Apr 2020
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random

from util import read_data_with_train_val_test
from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary



def run_exp(base_filters, filter_list, m_blocks_list):
    dataset = MyDataset(X_train, Y_train)
    dataset_val = MyDataset(X_val, Y_val)
    dataset_test = MyDataset(X_test, Y_test)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)

    # make model
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = Net1D(
        # 1-->12
        in_channels=12,
        base_filters=base_filters,
        ratio=1.0,
        filter_list=filter_list,
        m_blocks_list=m_blocks_list,
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        n_classes=4)
    model.to(device)

    summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)

    # train and test
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = nn.BCELoss()

    n_epoch = 50
    step = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        for batch_idx, batch in enumerate(prog_iter):

            input_x, input_y = tuple(t.to(device) for t in batch)
            # print(input_x.data.numpy().shape)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('Loss/train', loss.item(), step)
            if is_debug:
                break

        scheduler.step(_)

        # val
        model.eval()
        prog_iter_val = tqdm(dataloader_val, desc="Validation", leave=False)
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_val):
                input_x, input_y = tuple(t.to(device) for t in batch)
                # print(input_x.data.numpy().shape)
                pred = model(input_x)
                pred = pred.cpu().data.numpy()
                # print(pred.shape)
                pred = (pred > 0.5).astype(int)
                # pred = np.sum(pred, axis=1)
                # print(pred.shape)
                all_pred_prob.append(pred)
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.sum(all_pred_prob, axis=1) - 1 # need check after exp
        # print(all_pred.shape)

        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                pred = pred.cpu().data.numpy()
                pred = (pred > 0.5).astype(int)
                all_pred_prob.append(pred)
        all_pred_prob = np.concatenate(all_pred_prob)
        all_pred = np.sum(all_pred_prob, axis=1) - 1 # need check after exp


if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)

    batch_size = 32

    is_debug = False
    if is_debug:
        # writer = SummaryWriter('/nethome/shong375/log/regnet/challenge2017/debug')
        writer = SummaryWriter('/home/tarena/heartvoice_cspc_incart_ptb/debug')
    else:
        # writer = SummaryWriter('/nethome/shong375/log/regnet/challenge2017/first')
        writer = SummaryWriter('/home/tarena/heartvoice_cspc_incart_ptb/first')

    # make data, (sample, channel, length)
    X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_val, pid_test = read_data_with_train_val_test()
    print(pid_val.shape, pid_test.shape)
    print(X_train.shape, Y_train.shape)

    base_filters = 64
    filter_list = [64, 160, 160, 400, 400, 1024, 1024]
    m_blocks_list = [2, 2, 2, 3, 3, 4, 4]

    run_exp(
        base_filters=base_filters,
        filter_list=filter_list,
        m_blocks_list=m_blocks_list)
