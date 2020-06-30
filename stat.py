import pickle
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, r2_score, roc_auc_score

def my_eval(gt, pred):
    """
    gt: (n_samples, 3), values from (0, 1, 2, 3)
    pred: (n_samples, 3), values from (0, 1, 2, 3)
    """
    res = []
    res.append(accuracy_score(gt, pred))
    res.append(mean_absolute_error(gt, pred))
#     res.append(r2_score(gt, pred))
    cm = confusion_matrix(gt, pred)
    F1_0 = 2*cm[0,0] / (np.sum(cm[0,:])+np.sum(cm[:,0]))
    F1_1 = 2*cm[1,1] / (np.sum(cm[1,:])+np.sum(cm[:,1]))
    F1_2 = 2*cm[2,2] / (np.sum(cm[2,:])+np.sum(cm[:,2]))
    F1_3 = 2*cm[3,3] / (np.sum(cm[3,:])+np.sum(cm[:,3]))
    F1 = np.mean([F1_0, F1_1, F1_2, F1_3])
    res.extend([F1_0, F1_1, F1_2, F1_3, F1])
    print(res, cm)
    return np.array(res), cm


def label_one_hot(label, n_classes):
    out_label = []
    for l in label:
        tmp_label = np.zeros(n_classes)

        tmp_label[int(l)] = 1
        out_label.append(tmp_label)
    return np.array(out_label)

def get_auroc(y_true, y_pred_prob, n_classes):
    y_true = label_one_hot(y_true, n_classes)
    #save_roc_curve(y_true, y_pred_prob)
    all_auroc = []
    for i in range(n_classes):

        all_auroc.append(roc_auc_score(y_true[:, i], y_pred_prob[:, i]))
    avg_auroc = np.mean(all_auroc)
    all_auroc.append(avg_auroc)
    print(all_auroc)
    return all_auroc


def cal_pred_res(prob):
    pred = []
    for i, item in enumerate(prob):
        tmp_label = []
        tmp_label.append(1 - item[0])
        tmp_label.append(item[0] - item[1])
        tmp_label.append(item[1] - item[2])
        tmp_label.append(item[2])
        pred.append(tmp_label)
    pred = np.array(pred)
    pred = np.argmax(pred, axis=1)
    return pred

def read_gt_pred():
    with open('val_test_res.pkl', 'rb') as fin:
        res = pickle.load(fin)
    test_prob = res['test_prob']
    test_truth = np.array(res['test_truth'], dtype=int)

    cum_truth = np.sum(test_truth, axis=1)
    cum_pred = cal_pred_res(test_prob)

    return cum_truth, cum_pred

def read_ecg():

    with open('./test_data_age_gender.pkl', 'rb') as fin:
        res = pickle.load(fin)
    print(res.keys())
    return res['test_data'], res['pid'], res['age'], res['gender']
    
def plot_ecg(data, fs=500):
    """
    data has shape: (n_lead, n_length)
    """
    
    names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    n_lead, n_length = data.shape[0], data.shape[1]
    x_margin = 200
    y_margin = 1
    gap = 2.9/2
    x = n_length + 2*x_margin
    y = (n_lead-1)*gap + 2*y_margin
    
    base_x = np.array([0,0.04,0.04,0.12,0.12,0.4]) * fs - 2*x_margin//3
    base_y = np.array([0,0,1.0,1.0,0,0])

    height = 10
    width = 10*(x/fs/0.2)/(y/0.5)
    
    fig = plt.figure(figsize=(width,height), dpi=400)
    ax = fig.add_subplot(1, 1, 1)
    for i in range(n_lead):
        ax.plot(data[i]+gap*(11-i), 'k', linewidth=1)
        ax.annotate(names[i], (-x_margin//3, gap*(11-i)+0.5))
        ax.plot(base_x, base_y+gap*(11-i), 'k', linewidth=1)
    
    major_x_ticks = np.arange(-x_margin, n_length+x_margin, fs*0.2)
    minor_x_ticks = np.arange(-x_margin, n_length+x_margin, fs*0.04)
    major_y_ticks = np.arange(-y_margin, n_lead*gap+y_margin, 0.5)
    minor_y_ticks = np.arange(-y_margin, n_lead*gap+y_margin, 0.1)
    ax.set_xticks(major_x_ticks)
#     ax.set_xticks(minor_x_ticks, minor=True)
    ax.set_yticks(major_y_ticks)
#     ax.set_yticks(minor_y_ticks, minor=True)
    ax.tick_params(colors='w')

    ax.grid(True, which='both', color='#FF8C00', linewidth=0.5) # '#CC5500'
    ax.set_xlim([-x_margin, n_length+x_margin//2])
    ax.set_ylim([-y_margin, n_lead*gap+y_margin//2])

    x_mesh = np.arange(-x_margin, n_length+x_margin//2, fs*0.04)
    y_mesh = np.arange(-y_margin, n_lead*gap+y_margin//2, 0.1)
    xv, yv = np.meshgrid(x_mesh, y_mesh)
    plt.scatter(xv, yv, s=0.1, color='#FF8C00')

    return fig
    
def get_confusion_matrix_image(cm, mode='recall', normalized=False, title='Normalized Confusion Matrix'):
    if mode == 'recall':
        cm = cm / np.sum(cm, axis=1)[:,None]
    elif mode == 'precision':
        cm = cm / np.sum(cm, axis=0)[None,:]
    elif mode == 'count':
        cm = cm
    fig = plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
#     plt.title(title)
    plt.xlabel('Prediction', fontsize=16)
    plt.ylabel('Reference', fontsize=16)
    plt.xticks([0,1,2,3], ['N','L','M','H'], fontsize=16)
    plt.yticks([0,1,2,3], ['N','L','M','H'], fontsize=16)
    plt.tight_layout()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalized:
                text = plt.text(j, i, '{:.2f}'.format(cm[i, j]), ha="center", va="center", color="k", fontsize=16)
            else:
                text = plt.text(j, i, '{:d}'.format(cm[i, j]), ha="center", va="center", color="k", fontsize=16)

    return fig    
    
if __name__ == '__main__':
    
    ### 1, confusion matrix
#     gt, pred = read_gt_pred()
#     res, cm = my_eval(gt, pred)
#     print(res)
#     get_confusion_matrix_image(cm, mode='precision')
#     plt.savefig('res/cm_precision.pdf')
#     get_confusion_matrix_image(cm, mode='recall')
#     plt.savefig('res/cm_recall.pdf')
#     get_confusion_matrix_image(cm, mode='count')
#     plt.savefig('res/cm_recall.pdf')


    ### 2, confusion matrix by age, gender
#     gt, pred = read_gt_pred()
#     res, cm = my_eval(gt, pred)
#     _, _, age, gender = read_ecg()
#     # by gender
#     g1 = (gender==0)
#     g2 = (gender==1)
#     print(gt[g1], pred[g1])
#     res, cm = my_eval(gt[g1], pred[g1])
#     get_auroc(gt[g1], pred[g1], 4)
#     res, cm = my_eval(gt[g2], pred[g2])
#      get_auroc(gt[g2], pred[g2], 4)
#     print(np.sum(g1), np.sum(g2))

#     # by age
#     g1 = (age < 65)
#     g2 = np.logical_and(age >= 18, age < 65)
#     g3 = (age >= 65)
#     res, cm = my_eval(gt[g1], pred[g1])
#     res, cm = my_eval(gt[g2], pred[g2])
#     res, cm = my_eval(gt[g3], pred[g3])
#     get_auroc(gt[g1], pred[g1], 4)
#     get_auroc(gt[g2], pred[g2], 4)
#     get_auroc(gt[g3], pred[g3], 4)
#     print(np.sum(g1), np.sum(g2), np.sum(g3))

    ### 3, case study
    gt, pred = read_gt_pred()
    #print(gt, pred)
    ecg_data, pid, _, _ = read_ecg()
    n_samples = ecg_data.shape[0]
    #yes_pid = set([])
    #pid_cnter = []
    for i in tqdm(range(n_samples)):
        #if gt[i] != pred[i]:
            #pid_cnter.append(pid[i])
        
        if np.abs(gt[i] - pred[i]) == 3:

            tmp_data = ecg_data[i]
            fig = plot_ecg(tmp_data)
            plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
            plt.tight_layout()
            plt.savefig('res/cases/3_{}_{}_{}_{}.pdf'.format(i, int(pid[i]), gt[i], pred[i]))

        if np.abs(gt[i] - pred[i]) == 2:
    #        if pid[i] not in yes_pid:
            tmp_data = ecg_data[i]
            fig = plot_ecg(tmp_data)
            plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
            plt.tight_layout()
            plt.savefig('res/cases/2_{}_{}_{}_{}.pdf'.format(i, int(pid[i]), gt[i], pred[i]))
#                 yes_pid.add(pid[i])

        if np.abs(gt[i] - pred[i]) == 1:
 #           if pid[i] not in yes_pid:
            tmp_data = ecg_data[i]
            fig = plot_ecg(tmp_data)
            plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
            plt.tight_layout()
            plt.savefig('res/cases/1_{}_{}_{}_{}.pdf'.format(i, int(pid[i]), gt[i], pred[i]))
#                 yes_pid.add(pid[i])

    # print(Counter(pid_cnter))