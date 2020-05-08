import pickle
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error, r2_score

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
    print(cm)
    return np.array(res), cm

def read_gt_pred():
    with open('pred_res.pkl', 'rb') as fin:
        res = pickle.load(fin)
    test_prob = res['test_prob']
    test_truth = np.array(res['test_truth'], dtype=int)

    thresh = [0.5, 0.5, 0.5]
    test_pred = np.zeros_like(test_truth)
    for i in range(3):
        test_pred[:,i] = test_prob[:,i] > thresh[i]
    cum_pred = np.sum(test_pred, axis=1)
    cum_truth = np.sum(test_truth, axis=1)
    
    return cum_truth, cum_pred

def read_ecg():
    path = '/home/weiguodong/test/val_test_data'
    with open(os.path.join(path, 'test_data.pkl'), 'rb') as fin:
        res = pickle.load(fin)
    print(res.keys())
    return res['test_pid'], res['test_data']
    
def plot_ecg(data, fs=500):
    """
    data has shape: (n_lead, n_length)
    """
    
    n_lead, n_length = data.shape[0], data.shape[1]
    x_margin = 500
    y_margin = 1
    gap = 3
    x = n_length + 2*x_margin
    y = (n_lead-1)*gap + 2*y_margin
    
    height = 10
    width = 10*(x/fs/0.2)/(y/0.5)
    
    fig = plt.figure(figsize=(width,height))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(n_lead):
        ax.plot(data[i]+gap*i, 'k')
    
    major_x_ticks = np.arange(-x_margin, n_length+x_margin, fs*0.2)
    minor_x_ticks = np.arange(-x_margin, n_length+x_margin, fs*0.04)
    major_y_ticks = np.arange(-y_margin, n_lead*gap+y_margin, 0.5)
    minor_y_ticks = np.arange(-y_margin, n_lead*gap+y_margin, 0.1)
    ax.set_xticks(major_x_ticks)
#     ax.set_xticks(minor_x_ticks, minor=True)
    ax.set_yticks(major_y_ticks)
#     ax.set_yticks(minor_y_ticks, minor=True)
    ax.tick_params(colors='w')

    ax.grid(True, which='both', color='tab:pink')
    ax.set_xlim([-x_margin, n_length+x_margin])
    ax.set_ylim([-y_margin, n_lead*gap+y_margin])
    return fig
    
def get_confusion_matrix_image(cm, mode='recall', normalized=True, title='Normalized Confusion Matrix'):
    if mode == 'recall':
        cm = cm / np.sum(cm, axis=1)[:,None]
    elif mode == 'precision':
        cm = cm / np.sum(cm, axis=0)[None,:]
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
    
    ### confusion matrix
    gt, pred = read_gt_pred()
    res, cm = my_eval(gt, pred)
    print(res)
    get_confusion_matrix_image(cm, mode='precision')
    plt.savefig('res/cm_precision.pdf')
    get_confusion_matrix_image(cm, mode='recall')
    plt.savefig('res/cm_recall.pdf')

    ### case study
#     pid, ecg_data = read_ecg()
#     n_samples = ecg_data.shape[0]
#     yes_pid = set([])
#     for i in tqdm(range(n_samples)):
        
#         if np.abs(gt[i] - pred[i]) == 3:
#             if pid[i] not in yes_pid:
#                 tmp_data = ecg_data[i]
#                 fig = plot_ecg(tmp_data)
#                 plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
#                 plt.tight_layout()
#                 plt.savefig('res/cases/3_{}_{}_{}_{}.png'.format(i, pid[i], gt[i], pred[i]))
#                 yes_pid.add(pid[i])
        
#         if np.abs(gt[i] - pred[i]) == 2:
#             if pid[i] not in yes_pid:
#                 tmp_data = ecg_data[i]
#                 fig = plot_ecg(tmp_data)
#                 plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
#                 plt.tight_layout()
#                 plt.savefig('res/cases/2_{}_{}_{}_{}.png'.format(i, pid[i], gt[i], pred[i]))
#                 yes_pid.add(pid[i])
        
#         if np.abs(gt[i] - pred[i]) == 1:
#             if pid[i] not in yes_pid:
#                 tmp_data = ecg_data[i]
#                 fig = plot_ecg(tmp_data)
#                 plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
#                 plt.tight_layout()
#                 plt.savefig('res/cases/1_{}_{}_{}_{}.png'.format(i, pid[i], gt[i], pred[i]))
#                 yes_pid.add(pid[i])