
import numpy as np
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from net1d import Net1D, MyDataset
from torch.utils.data import DataLoader
from util import read_data_with_train_val_test

# model
base_filters = 64
filter_list = [64, 160, 160, 400, 400, 1024, 1024]
m_blocks_list = [2, 2, 2, 3, 3, 4, 4]
batch_size = 128
model = Net1D(
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

# get input data
_,_,X_test,_,_,Y_test,_,_ = read_data_with_train_val_test()
dataset_test = MyDataset(X_test, Y_test)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, drop_last=False)

device = torch.device('cuda')
model.load_state_dict(torch.load('./model/epoch_9_params_file.pkl', map_location=device))
model.to(device)

# predict
model.eval()
prog_iter_test = tqdm(dataloader_test, desc='Validation', leave=False)
test_pred_prob = []
test_labels = []
with torch.no_grad():
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x, input_y = tuple(t.to(device) for t in batch)
        pred = model(input_x)
        pred_prob = pred_prob.cpu().data.numpy()
        test_pred_prob.append(pred_prob)
        test_labels.append(input_y.cpu().data.numpy())
test_pred_prob = np.concatenate(test_pred_prob)
test_labels = np.concatenate(test_labels)

# save res
pred_res = {'test_prob': test_pred_prob, 'test_truth': test_labels}
with open('./pred_res.pkl', 'wb') as fin:
    pickle.dump(pred_res, fin)
# test_res = pd.DataFrame(test_pred_prob)
# test_res.to_csv('./test_prob.csv')
# test_truth = pd.DataFrame(test_labels)
# test_truth.to_csv('./test_truth.csv')
