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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, r2_score
import random
import os
import pickle

from util import read_data_with_train_val_test
from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary

def my_eval(gt, pred):
    """
    gt: (n_samples, 4), values from (0, 1, 2, 3)
    pred: (n_samples, 4), values from (0, 1, 2, 3)
    """
    res = []
    res.append(accuracy_score(gt, pred))
    res.append(mean_absolute_error(gt, pred))
    res.append(r2_score(gt, pred))
    return np.array(res)

def get_pred_from_prob(prob, mode='max_prob'):
    """
    prob: (n,4)
    pred: (n,1)
    """
    pred = 0
    if mode == 'sum_up': # simple sum up
        pred = np.sum(prob, axis=1)
    elif mode == 'max_prob':
        pred = np.argmax(prob, axis=1)
    elif mode == 'all_0':
        pred = np.zeros(prob.shape[0])
    # print(pred)

    return pred

def run_exp(base_filters, filter_list, m_blocks_list):

    writer = SummaryWriter(save_path)
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
    # loss_func = nn.BCEWithLogitsLoss()
    res = []
    n_epoch = 50
    step = 0
    for _ in tqdm(range(n_epoch), desc="epoch", leave=False):

        # train
        model.train()
        prog_iter = tqdm(dataloader, desc="Training", leave=False)
        train_pred_prob = []
        tmp_res = []
        train_labels = []
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_func(pred, input_y)
            #
            pred = pred.cpu().data.numpy()
            pred = (pred > 0.5).astype(int)
            train_pred_prob.append(pred)
            train_labels.append(input_y.cpu().data.numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('Loss/train', loss.item(), step)
            if is_debug:
                break
        scheduler.step(_)
        #
        train_labels = np.concatenate(train_labels)
        train_pred_prob = np.concatenate(train_pred_prob)
        all_pred = get_pred_from_prob(train_pred_prob, mode=mode)
        all_gt = np.sum(train_labels, axis=1)
        tmp_res.extend(my_eval(all_gt, all_pred))

        # save model
        model_save_path = './model/epoch_{}_params_file.pkl'.format(_)
        torch.save(model.state_dict(), model_save_path)

        # val
        model.eval()
        prog_iter_val = tqdm(dataloader_val, desc="Validation", leave=False)
        val_pred_prob = []
        val_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_val):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                pred = pred.cpu().data.numpy()
                # pred = np.round(pred).astype(int)
                pred = (pred > 0.5).astype(int)
                val_pred_prob.append(pred)
                val_labels.append(input_y.cpu().data.numpy())
        val_pred_prob = np.concatenate(val_pred_prob)
        all_pred = get_pred_from_prob(val_pred_prob, mode=mode)
        all_gt = np.sum(val_labels, axis=1)
        tmp_res.extend(my_eval(all_gt, all_pred))

        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        test_pred_prob = []
        test_pred_value = []
        test_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                pred_prob = pred.cpu().data.numpy()
                pred_value = (pred_prob > 0.5).astype(int)
                test_pred_value.append(pred_value)
                test_pred_prob.append(pred_prob)
                test_labels.append(input_y.cpu().data.numpy())
        test_pred_prob = np.concatenate(test_pred_prob)
        test_labels = np.concatenate(test_labels)
        # save test prob
        prob_res = {'prob':test_pred_prob}
        with open('./prob/epoch_{}_prob.pkl'.format(_), 'wb') as f:
            pickle.dump(prob_res, f)

        test_pred_value = np.concatenate(test_pred_value)
        all_pred = get_pred_from_prob(test_pred_value, mode=mode)
        all_gt = np.sum(test_labels, axis=1)
        tmp_res.extend(my_eval(all_gt, all_pred))

        # save at each epoch
        res.append(tmp_res)
        np.savetxt(os.path.join(save_path, 'res.csv'), np.array(res), fmt='%.4f', delimiter=',')

    return np.array(res)


if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    mode = 'sum_up'# 'all_0', 'max_prob'
    batch_size = 128

    is_debug = True
    if is_debug:
        save_path = '/home/tarena/heartvoice_cspc_incart_ptb/debug'
    else:
        save_path = '/home/tarena/heartvoice_cspc_incart_ptb/first'

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
