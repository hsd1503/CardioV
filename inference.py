import os
import numpy as np
import pickle
import torch

from net1d import Net1D, MyDataset
from torch.utils.data import DataLoader
from process_raw_data import read_dataset
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def my_eval(gt, pred):
    res = []
    res.append(accuracy_score(gt, pred))
    res.append(mean_absolute_error(gt, pred))
    return np.array(res)

def get_f1(gt, pred):
    cm = confusion_matrix(gt, pred)
    print(cm)
    total = 0
    for i in range(np.array(cm).shape[0]):
        a = 2*cm[i][i] / (np.sum(cm[:,i])+np.sum(cm[i,:]))
        print(a, end=' ')
        total += a
    print(total / 4)
    print(my_eval(gt, pred))

def label_one_hot(label, n_classes):
    out_label = []
    for l in label:
        tmp_label = np.zeros(n_classes)

        tmp_label[int(l)] = 1
        out_label.append(tmp_label)
    return np.array(out_label)

def save_roc_curve(y_true, y_prob):
    fpr_0, tpr_0, treshold_0 = roc_curve(y_true[:, 0], y_prob[:, 0])
    fpr_1, tpr_1, treshold_1 = roc_curve(y_true[:, 1], y_prob[:, 1])
    fpr_2, tpr_2, treshold_2 = roc_curve(y_true[:, 2], y_prob[:, 2])
    fpr_3, tpr_3, treshold_3 = roc_curve(y_true[:, 3], y_prob[:, 3])
    roc_auc_0 = auc(fpr_0, tpr_0)
    roc_auc_1 = auc(fpr_1, tpr_1)
    roc_auc_2 = auc(fpr_2, tpr_2)
    roc_auc_3 = auc(fpr_3, tpr_3)
    plt.figure(figsize=(6, 4), dpi=180)
    plt.plot(fpr_0, tpr_0, color='tab:blue', lw=3, label='${ROC_0}$(AUC=%.4f)' % roc_auc_0)
    plt.plot(fpr_1, tpr_1, color='tab:orange', lw=3, label='${ROC_1}$(AUC=%.4f)' % roc_auc_1)
    plt.plot(fpr_2, tpr_2, color='tab:green', lw=3, label='${ROC_2}$(AUC=%.4f)' % roc_auc_2)
    plt.plot(fpr_3, tpr_3, color='tab:red', lw=3, label='${ROC_3}$(AUC=%.4f)' % roc_auc_3)
    plt.plot([0,1], [0,1], color='tab:purple', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
#    plt.title('ROC Curve of 4 Class', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('./roc_curve.pdf')


def get_auroc(y_true, y_pred_prob, n_classes):
    y_true = label_one_hot(y_true, n_classes)
    save_roc_curve(y_true, y_pred_prob)
    all_auroc = []
    for i in range(n_classes):

        all_auroc.append(roc_auc_score(y_true[:, i], y_pred_prob[:, i]))
    avg_auroc = np.mean(all_auroc)
    all_auroc.append(avg_auroc)
    # print(all_auroc)
    return all_auroc



def cal_pred_res(prob):
    test_pred = []
    for i, item in enumerate(prob):
        tmp_label = []
        tmp_label.append(1 - item[0])
        tmp_label.append(item[0] - item[1])
        tmp_label.append(item[1] - item[2])
        tmp_label.append(item[2])
        test_pred.append(tmp_label)
    #return np.argmax(np.array(test_pred), axis=1)
    return np.array(test_pred), np.argmax(np.array(test_pred), axis=1)

if __name__ == '__main__':
    base_filters = 64
    filter_list = [64, 160, 160, 400, 400, 1024, 1024]
    m_blocks_list = [2, 2, 2, 3, 3, 4, 4]
    batch_size = 128

    model = Net1D(in_channels=12,base_filters=base_filters,ratio=1.0,filter_list=filter_list,m_blocks_list=m_blocks_list,kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            n_classes=3)

    _,_,_,_,X_test,Y_test = read_dataset()

    dataset_test = MyDataset(X_test, Y_test)
    #dataset_test = MyDataset_o(X_val, Y_val)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)

    model_path_root = './ptb-xl'

    device = torch.device('cuda')
    model_list = os.listdir(os.path.join(model_path_root,'res_finetune_from_cls_filter/model_2'))
    #for i,item in enumerate(model_list):
   #model.load_state_dict(torch.load(os.path.join(model_path_root, 'res_finetune_from_cls_filter/model_2/{}'.format(item)),map_location=device))

    model.load_state_dict(torch.load(os.path.join(model_path_root, 'res_finetune_from_cls_filter/model_4/epoch_6_params_file.pkl'),map_location=device))

    model.to(device)

    model.eval()
    prog_iter_test = tqdm(dataloader_test, desc="Testing", leave=False)
    test_pred_prob = []
    test_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            pred_prob = pred.cpu().data.numpy()
            test_pred_prob.append(pred_prob)
            test_labels.append(input_y.cpu().data.numpy())
    test_labels = np.concatenate(test_labels)
    test_pred_prob = np.concatenate(test_pred_prob)
    # cls
    #all_gt = test_labels
    #all_pred = np.argmax(test_pred_prob, axis=1)
    # ordinal
    all_gt = np.sum(test_labels, axis=1)
    all_pred = cal_pred_res(test_pred_prob)[1]

    all_pred_prob = cal_pred_res(test_pred_prob)[0]
    get_f1(all_gt, all_pred)
    get_auroc(all_gt, all_pred_prob, 4)



