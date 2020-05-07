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
    res.append(r2_score(gt, pred))
    cm = confusion_matrix(gt, pred)
    print(cm)
    return np.array(res)

def read_gt_pred():
    with open('pred_res.pkl', 'rb') as fin:
        res = pickle.load(fin)
    test_prob = res['test_prob']
    test_truth = res['test_truth']

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
    return res['test_data']
    
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
    
if __name__ == '__main__':
    
    gt, pred = read_gt_pred()
    ecg_data = read_ecg()
    print(my_eval(gt, pred))
    print(gt.shape, pred.shape, ecg_data.shape)
    n_samples = ecg_data.shape[0]

    yes_samples = set([])
    for i in tqdm(range(n_samples)):
        
        if np.abs(gt[i] - pred[i]) == 3:
            yes_samples.add(i)
            if (i-1) not in yes_samples:
                tmp_data = ecg_data[i]
                fig = plot_ecg(tmp_data)
                plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
                plt.savefig('img/3_{}_{}_{}.png'.format(i, gt[i], pred[i]))
        
        if np.abs(gt[i] - pred[i]) == 2:
            yes_samples.add(i)
            if (i-1) not in yes_samples:
                tmp_data = ecg_data[i]
                fig = plot_ecg(tmp_data)
                plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
                plt.savefig('img/2_{}_{}_{}.png'.format(i, gt[i], pred[i]))
        
        if np.abs(gt[i] - pred[i]) == 1:
            yes_samples.add(i)
            if (i-1) not in yes_samples:
                tmp_data = ecg_data[i]
                fig = plot_ecg(tmp_data)
                plt.title('Groundtruth: {}, Prediction: {}'.format(gt[i], pred[i]))
                plt.savefig('img/1_{}_{}_{}.png'.format(i, gt[i], pred[i]))

        
        
        
        
        