import ast
import wfdb
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from mne.filter import filter_data
import pickle

def save_signal():
     Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id', encoding='gbk')
     Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
     X = load_raw_data(Y, sampling_rate, path)
     res = {'signal': X}
     with open('./signal_data.pkl', 'wb') as fin:
         pickle.dump(res, fin, protocol=4)

def load_raw_data(df, sampling_rate, path):
    data_list = []
    for f in tqdm(df.filename_hr,  desc='Reading signal...'):
        data = wfdb.rdsamp(path + f)
        # bandpass filter 
        data = filter_data(data[0].T, 500, 0.5, 50, verbose='ERROR')
        data_list.append(data)
    data_list = np.array(data_list)
    return data_list


def aggregate_diagnostic(y_dic):
    # get label of the max likihood
    label = max(y_dic, key=y_dic.get)
    return label

def read_dataset():
    path = './ptb-xl/'
    sampling_rate = 500

    # load and convert annotation data
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id', encoding='gbk')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    # print(Y.age)
    # print(type(Y.age))
    # print(np.array(Y.age))
    # print(len(np.array(Y.age)[np.array(Y.age) >= 65]))
    # X = load_raw_data(Y, sampling_rate, path)
    
    # Load raw signal data
    with open('./signal_data.pkl', 'rb') as fout:
        res = pickle.load(fout)
    X = res['signal']
    # print(X.shape)
       
    # Load scp_statements.csv for diagnostic c
    agg_df = pd.read_csv('./statements.csv', index_col=0)
    # print(list(agg_df.index))
    # agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['label'] = Y.scp_codes.apply(aggregate_diagnostic)
    print(Counter(np.array(Y['label'])))
    sum = 0
    dic = dict(Counter(np.array(Y['label'])))
    for i in dic:
        sum += dic[i]
    print(sum)

    # generate label dic
    l_dict = {}
    l_list = list(agg_df.riskscore)
    for i, item in tqdm(enumerate(list(agg_df.index)), desc='Create Label Dict...'):
        l_dict[item] = l_list[i]

    # get all labels
    y = np.array(Y['label'])
    L = np.zeros_like(y)
    for i, item in tqdm(enumerate(y), desc='Alter Label To Riskscore...'):
        L[i] = l_dict[y[i]] 
    print(Counter(L))

    # Split data into train, validate and test
    val_fold = 9
    test_fold = 10
    # Train
    x_train = X[np.where(np.array(Y.strat_fold) < val_fold)]
    y_train = L[np.where(np.array(Y.strat_fold) < val_fold)]
    # Validate
    x_val = X[np.where(np.array(Y.strat_fold) == val_fold)]
    y_val = L[np.where(np.array(Y.strat_fold) == val_fold)]
    # Test
    x_test = X[np.where(np.array(Y.strat_fold) == test_fold)]
    y_test = L[np.where(np.array(Y.strat_fold) == test_fold)]
    # print(x_train.shape, x_val.shape, x_test.shape)
    # print(y_train.shape, y_val.shape, y_test.shape)
     
    # save data
    # Age = np.array(Y.age)
    # Gender = np.array(Y.sex)
    # Patient_id = np.array(Y.patient_id)
    # age = Age[np.where(np.array(Y.strat_fold) == test_fold)]
    # gender = Gender[np.where(np.array(Y.strat_fold) == test_fold)]
    # patient_id = Patient_id[np.where(np.array(Y.strat_fold) == test_fold)]
    # res = {"test_data": x_test, "age":age, "gender":gender, "pid":patient_id}
    # ith open("./test_data_age_gender.pkl", "wb") as fin:
        #pickle.dump(res, fin, protocol=4)

    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == '__main__':
    path = './ptb-xl/'
    sampling_rate = 500
    #save_signal()
    x_train, y_train, x_val, y_val, x_test, y_test = read_dataset()
        
