
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, r2_score, f1_score
import random
import os
import pickle

from process_raw_data import read_dataset
from net1d import Net1D, MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary


def cal_pred_res(prob):
    test_pred = []
    for i,item in enumerate(prob):
        tmp_label = []
        tmp_label.append(1 - item[0])
        tmp_label.append(item[0] - item[1])
        tmp_label.append(item[1] - item[2])
        tmp_label.append(item[2])
        test_pred.append(tmp_label)
    test_pred = np.array(test_pred)
    #test_pred = np.argmax(test_pred, axis=1)
    return test_pred


def my_eval_r(gt, pred):
    res = []
    res.append(mean_absolute_error(gt, pred))
    return np.array(res)

def my_eval(gt, pred):
    res = []
    #res.append(accuracy_score(gt, pred))
    res.append(mean_absolute_error(gt, pred))
    #res.append(r2_score(gt, pred))
    #res.append(f1_score(gt, pred, average='micro'))
    return np.array(res)

def run_exp_c(base_filters, filter_list, m_blocks_list):

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
    loss_func1 = torch.nn.CrossEntropyLoss()
    #loss_func2 = nn.BCELoss()
    #loss_func = nn.BCEWithLogitsLoss()
    #loss_func = nn.MSELoss()
    res = []
    n_epoch = 10
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

            #print(pred.size(), input_y.size())
            pred = pred.view(pred.cpu().data.numpy().shape[0])
            #print(pred.size())

            loss = loss_func1(pred, input_y)

            #pred = pred.cpu().data.numpy()
            #pred = cal_pred_res(pred)
            #truth = input_y.cpu().data.numpy()
            #truth = np.sum(truth, axis=1)
            #pred = torch.from_numpy(pred)
            #truth = torch.from_numpy(truth).long()
            #pred.to(device)
            #truth.to(device)
            #loss1 = loss_func1(pred, truth).type(torch.cuda.FloatTensor)

            #loss = loss1 + loss2
            #print(loss1.dtype)
            #print(loss2.dtype)
            #print(loss.dtype)

            #pred = (pred > 0.5).astype(int)
            #train_pred_prob.append(pred)
            #train_labels.append(input_y.cpu().data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('Loss/train', loss.item(), step)
            #print(loss.item())
            if is_debug:
                break
        scheduler.step(_)

        # save model
        model_save_path = './res_reg/model_{}/epoch_{}_params_file.pkl'.format(i,_)
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
                #pred = pred.astype(int)
                #pred = (pred > 0.5).astype(int)
                #pred = np.argmax(pred, axis=1)
                # print(pred.shape)
                val_pred_prob.append(pred)
                val_labels.append(input_y.cpu().data.numpy())
        val_labels = np.concatenate(val_labels)
        val_pred_prob = np.concatenate(val_pred_prob)
        all_pred = np.argmax(val_pred_prob, axis=1)
        #all_pred = val_pred_prob
        all_gt = val_labels
        print(all_pred[:10], all_gt[:10])
        tmp_res.extend(my_eval(all_pred, all_gt))

        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        test_pred_prob = []
        test_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                pred_prob = pred.cpu().data.numpy()
                #pred_prob = cal_pred_res(pred_prob)
                # pred_prob = (pred_prob > 0.5).astype(int)
                test_pred_prob.append(pred_prob)
                test_labels.append(input_y.cpu().data.numpy())
        test_labels = np.concatenate(test_labels)
        test_pred_prob = np.concatenate(test_pred_prob)
        all_pred = np.argmax(test_pred_prob,axis=1)
        #all_pred = test_pred_prob
        all_gt = test_labels

        print(all_pred[:10], all_gt[:10])
        tmp_res.extend(my_eval(all_pred, all_gt))

        # save at each epoch
        res.append(tmp_res)
        np.savetxt(os.path.join(save_path, 'res.csv'),np.array(res), fmt='%.4f', delimiter=',')

    return np.array(res)

def run_exp_o(base_filters, filter_list, m_blocks_list):

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
        n_classes=3)
    model.to(device)

    summary(model, (X_train.shape[1], X_train.shape[2]), device=device_str)

    # train and test
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func1 = torch.nn.CrossEntropyLoss()
    loss_func2 = nn.BCELoss()
    #loss_func = nn.BCEWithLogitsLoss()
    res = []
    n_epoch = 12
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

            loss = loss_func2(pred, input_y)

            # pred = pred.cpu().data.numpy()
            # pred = cal_pred_res(pred)
            # truth = input_y.cpu().data.numpy()
            # truth = np.sum(truth, axis=1)
            # pred = torch.from_numpy(pred)
            #
            # truth = torch.from_numpy(truth).long()
            # pred.to(device)
            # truth.to(device)
            # loss1 = loss_func1(pred, truth).type(torch.cuda.FloatTensor)
            # loss1.requires_grad=True
            #
            # loss = 0.99*loss1 + 0.01*loss2
            #print(loss1.dtype)
            #print(loss2.dtype)
            #print(loss.dtype)

            #pred = (pred > 0.5).astype(int)
            #train_pred_prob.append(pred)
            #train_labels.append(input_y.cpu().data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('Loss/train', loss.item(), step)
            #print(loss.item())
            if is_debug:
                break
        scheduler.step(_)


        # save model
        model_save_path = './res_ordinal/model_{}/epoch_{}_params_file.pkl'.format(i,_)
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
                pred = cal_pred_res(pred)
                #pred = (pred > 0.5).astype(int)
                val_pred_prob.append(pred)
                val_labels.append(input_y.cpu().data.numpy())
        val_labels = np.concatenate(val_labels)
        val_pred_prob = np.concatenate(val_pred_prob)
        all_pred = np.argmax(val_pred_prob, axis=1)
        all_gt = np.sum(val_labels, axis=1)
        print(all_pred[:10], all_gt[:10])
        tmp_res.extend(my_eval(all_pred, all_gt))

        # test
        model.eval()
        prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
        test_pred_prob = []
        test_labels = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(prog_iter_test):
                input_x, input_y = tuple(t.to(device) for t in batch)
                pred = model(input_x)
                pred_prob = pred.cpu().data.numpy()
                pred_prob = cal_pred_res(pred_prob)
                #pred_prob = (pred_prob > 0.5).astype(int)
                test_pred_prob.append(pred_prob)
                test_labels.append(input_y.cpu().data.numpy())
        test_labels = np.concatenate(test_labels)
        test_pred_prob = np.concatenate(test_pred_prob)

        all_pred = np.argmax(test_pred_prob,axis=1)
        all_gt = np.sum(test_labels, axis=1)

        print(all_pred[:10], all_gt[:10])
        tmp_res.extend(my_eval(all_pred, all_gt))

        # save at each epoch
        res.append(tmp_res)
        np.savetxt(os.path.join(save_path, 'res.csv'),np.array(res), fmt='%.4f', delimiter=',')

    return np.array(res)



if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)
    batch_size = 128

    X_train, Y_train,X_val, Y_val, X_test, Y_test = read_dataset()

    base_filters = 64
    filter_list = [64, 160, 160, 400, 400, 1024, 1024]
    m_blocks_list = [2, 2, 2, 3, 3, 4, 4]

    is_debug = False
    for i in range(0,3):
        if is_debug:
            save_path = './test/debug'
        else:
            save_path = './ptb-xl/res_reg/res_{}'.format(i)

        run_exp_c(base_filters=base_filters,filter_list=filter_list,m_blocks_list=m_blocks_list)

    for i in range(0, 3):
        if is_debug:
            save_path = './test/debug'
        else:
            save_path = './ptb-xl/res_combineloss/res_{}'.format(i)

        #run_exp_o(base_filters=base_filters,filter_list=filter_list,m_blocks_list=m_blocks_list)





