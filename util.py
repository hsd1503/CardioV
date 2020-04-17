import numpy as np
import pandas as pd
import scipy.io
import dill, pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import wfdb

def cpsc_preprocess():
    # read label
    label_df = pd.read_csv('ecg_data/CPSC/REFERENCE.csv', header=0)
    label = label_df.iloc[:,1:].values

    # read data
    cpsc_data = []
    filenames = label_df.iloc[:,0].values
    for filename in tqdm(filenames):
        mat = scipy.io.loadmat('./ecg_data/CPSC/TrainningSet/{0}.mat'.format(filename))
        mat = np.array(mat['ECG'][0][0][-1].T)
        cpsc_data.append(mat)
    cpsc_data = np.array(cpsc_data)
    # cpsc_db1
    res_1 = {'data': cpsc_data[:2000], 'label': label[:2000]}
    with open('ecg_data/CPSC/cpsc_raw_1.pkl', 'wb') as fin:
        dill.dump(res_1, fin)
    # cpsc_db2
    res_2 = {'data': cpsc_data[2000:4470], 'label': label[2000:4470]}
    with open('ecg_data/CPSC/cpsc_raw_2.pkl', 'wb') as fin:
        dill.dump(res_2, fin)
    # cpsc_db3
    res_3 = {'data': cpsc_data[4470:], 'label': label[4470:]}
    with open('ecg_data/CPSC/cpsc_raw_3.pkl', 'wb') as fin:
        dill.dump(res_3, fin)

def ptb_preprocess():
    ptb_data = []
    comments = []

    with open('./ecg_data/PTBDB/ptbdb_raw.pkl', 'rb') as fout:
        data = dill.load(fout)
        print(data['data'].shape)

    for i in range(len(data['data'])):
    # for i in range(3):
        ptb_data.append(data['data'][i][:, :12])
        # print(data[i].shape)
        comments.append(data['comments'][i])
    ptb_data = np.array(ptb_data)
    disease_list = [['Myocardial infarction'], ['Healthy control'], ['Heart failure', 'Cardiomyopathy'],
                    ['Bundle branch block'], ['Dysrhythmia'], ['Hypertrophy'], ['Valvular heart disease'],
                    ['Myocarditis']]
    all_disease = []
    for c in comments:
        flag = False
        txt_disease = c[4]
        tmp_label = [0 for _ in range(len(disease_list))]
        for i in range(len(disease_list)):
            disease = disease_list[i]
            for d in disease:
                if d in txt_disease:
                    tmp_label[i] = 1
                    all_disease.append(tmp_label)
                    flag = True
                    break
        if flag == False:
            all_disease.append(tmp_label)
    all_disease = np.array(all_disease)
    print(type(all_disease))
    print(all_disease.shape)

    ptb_res_1 = {'data': ptb_data[:270], 'label': all_disease[:270]}
    with open('./ecg_data/PTBDB/ptb_raw_1.pkl', 'wb') as fin:
        dill.dump(ptb_res_1, fin)
    ptb_res_2 = {'data': ptb_data[270:], 'label': all_disease[270:]}
    with open('./ecg_data/PTBDB/ptb_raw_2.pkl', 'wb') as fin:
        dill.dump(ptb_res_2, fin)

def incart_preprocess():
    db_path = './ecg_data/incartdb/incartdb/'
    incart_data = []
    comments = []
    with open(db_path + 'RECORDS', 'r') as fin:
        all_record_name = fin.read().strip().split('\n')
    for record_name in all_record_name:
        tmp_data_res = wfdb.rdsamp(db_path + record_name)
        incart_data.append(tmp_data_res[0])
        comments.append(tmp_data_res[1]['comments'])
        print(tmp_data_res[0])
        print(tmp_data_res[0].shape)
    incart_data = np.array(incart_data)
    print(incart_data.shape)
    disease_list = [['Acute MI'], ['Transient ischemic attack'], ['Earlier MI'],
                    ['Coronary artery disease'], ['Sinus node dysfunction'], ['Supraventricular ectopy'],
                    ['Atrial fibrillation or SVTA'], ['WPW'], ['AV block'], ['Bundle branch block']]
    all_disease = []
    for c in comments:
        flag = False
        txt_disease = c[0]
        tmp_label = [0 for _ in range(len(disease_list))]
        for i in range(len(disease_list)):
            disease = disease_list[i]
            for d in disease:
                if d in txt_disease:
                    tmp_label[i] = 1
                    # print(tmp_label)
                    all_disease.append(tmp_label)
                    flag = True
                    break
        if flag == False:
            all_disease.append(tmp_label)
    all_disease = np.array(all_disease)
    print(type(all_disease))
    print(all_disease.shape)

    incart_res_1 = {'data': incart_data[:32], 'label': all_disease[:32]}
    with open('./ecg_data/incartdb/incartdb/incart_raw_1.pkl', 'wb') as fin:
        pickle.dump(incart_res_1, fin)
    incart_res_2 = {'data': incart_data[32:], 'label': all_disease[32:]}
    with open('./ecg_data/incartdb/incartdb/incart_raw_2.pkl', 'wb') as fin:
        pickle.dump(incart_res_2, fin)

# Add critical value label
def cpsc_alter_label():

    with open('./ecg_data/CPSC/cpsc_raw_1.pkl', 'rb') as fout:
        res = dill.load(fout)
    labels = res['label']
    # print(labels[:20])
    data = res['data']

    critical_value_label = []
    critical_value_data = []
    for i, label in enumerate(labels):
        # print(int(label[0]), type(label[0]))
        if int(label[0]) in [2, 7, 8]:
            critical_value_data.append(data[i])
            critical_value_label.append([0, 1, 0, 0])
        elif int(label[0]) in [3, 4, 5, 6]:
            critical_value_data.append(data[i])
            critical_value_label.append([0, 0, 1, 0])
        elif int(label[0]) == 9:
            critical_value_data.append(data[i])
            critical_value_label.append([1, 0, 0, 0])
        else: # normal
            critical_value_data.append(data[i])
            critical_value_label.append([0, 0, 0, 1])
    critical_value_data = np.array(critical_value_data)
    critical_value_label = np.array(critical_value_label)
    new_res = {'data':critical_value_data, 'label':critical_value_label}
    with open('./ecg_data/CPSC/cpsc_raw_1_1.pkl', 'wb') as fin:
        dill.dump(new_res, fin)

def ptb_alter_label():
    with open('./ecg_data/PTBDB/ptb_raw_2.pkl', 'rb') as fout:
        res = dill.load(fout)
    labels = res['label']
    # print(labels[:20])
    data = res['data']

    critical_value_label = []
    critical_value_data = []
    for i, label in enumerate(labels):
        if label[0]==1 or label[2]==1 or label[4]==1 or label[7]==1:
            critical_value_data.append(data[i])
            critical_value_label.append([0, 1, 0, 0])
        elif label[3]==1 or label[5]==1 or label[6]==1:
            critical_value_data.append(data[i])
            critical_value_label.append([0, 0, 1, 0])
        elif label[1]==1:
            critical_value_data.append(data[i])
            critical_value_label.append([0, 0, 0, 1])
        else: # Miscellaneous
            continue
    critical_value_data = np.array(critical_value_data)
    critical_value_label = np.array(critical_value_label)
    new_res = {'data':critical_value_data, 'label':critical_value_label}
    with open('./ecg_data/PTBDB/ptb_raw_2_1.pkl', 'wb') as fin:
        dill.dump(new_res, fin)

def incart_alter_label():
    with open('./ecg_data/incartdb/incartdb/incart_raw_2.pkl', 'rb') as fout:
        res = dill.load(fout)
    labels = res['label']
    # print(labels[:20])
    data = res['data']

    critical_value_label = []
    critical_value_data = []
    for i, label in enumerate(labels):
        if label[0]==1:# Acute MI
            critical_value_data.append(data[i])
            critical_value_label.append([1, 0, 0, 0])
        elif label[3]==1 or label[5]==1 or label[7]==1 or label[9]==1:
            critical_value_data.append(data[i])
            critical_value_label.append([0, 0, 1, 0])
        elif label[1]==1 or label[2]==1 or label[4]==1 or label[6]==1 or label[8]==1:
            critical_value_data.append(data[i])
            critical_value_label.append([0, 1, 0, 0])
        else:
            critical_value_data.append(data[i])
            critical_value_label.append([0, 0, 0, 1])
    critical_value_data = np.array(critical_value_data)
    critical_value_label = np.array(critical_value_label)
    new_res = {'data':critical_value_data, 'label':critical_value_label}
    with open('./ecg_data/incartdb/incartdb/incart_raw_2_1.pkl', 'wb') as fin:
        dill.dump(new_res, fin)

def merge_db():

    with open('./ecg_data/CPSC/cpsc_raw_1_1.pkl', 'rb') as fout_1:
        cpsc_res_1 = dill.load(fout_1)
    cpsc_data_1 = cpsc_res_1['data']
    cpsc_label_1 = cpsc_res_1['label']
    with open('./ecg_data/CPSC/cpsc_raw_2_1.pkl', 'rb') as fout_2:
        cpsc_res_2 = dill.load(fout_2)
    cpsc_data_2 = cpsc_res_2['data']
    cpsc_label_2 = cpsc_res_2['label']
    with open('./ecg_data/CPSC/cpsc_raw_3_1.pkl', 'rb') as fout_3:
        cpsc_res_3 = dill.load(fout_3)
    cpsc_data_3 = cpsc_res_3['data']
    cpsc_label_3 = cpsc_res_3['label']

    with open('./ecg_data/PTBDB/ptb_raw_1_1.pkl', 'rb') as fout_4:
        ptb_res_1 = dill.load(fout_4)
    ptb_data_1 = ptb_res_1['data']
    ptb_label_1 = ptb_res_1['label']
    with open('./ecg_data/CPSC/cpsc_raw_3_1.pkl', 'rb') as fout_5:
        ptb_res_2 = dill.load(fout_5)
    ptb_data_2 = ptb_res_2['data']
    ptb_label_2 = ptb_res_2['label']
    all_data = np.concatenate((cpsc_data_1, cpsc_data_2, cpsc_data_3, ptb_data_1, ptb_data_2))
    data_list = list(all_data)

    with open('./ecg_data/incartdb/incartdb/incart_raw_1_1.pkl', 'rb') as fout_6:
        incart_res_1 = dill.load(fout_6)
    incart_data_1 = incart_res_1['data']
    incart_label_1 = incart_res_1['label']
    with open('./ecg_data/incartdb/incartdb/incart_raw_2_1.pkl', 'rb') as fout_7:
        incart_res_2 = dill.load(fout_7)
    incart_data_2 = incart_res_2['data']
    incart_label_2 = incart_res_2['label']
    # all_label
    all_label = np.concatenate((cpsc_label_1, cpsc_label_2, cpsc_label_3, ptb_label_1, ptb_label_2, incart_label_1, incart_label_2))
    for i in range(incart_data_1.shape[0]):
        data_list.append(incart_data_1[i])
    for j in range(incart_data_2.shape[0]):
        data_list.append(incart_data_2[j])
    all_data = np.array(data_list)

    all_res = {'data': all_data, 'label': all_label}
    with open('./data.pkl', 'wb') as fin:
        dill.dump(all_res, fin)

def slide_and_cut(X, Y, window_size, stride, output_pid=False):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y[0] == 1:
            i_stride = stride//16
        elif tmp_Y[1] == 1:
            i_stride = stride
        elif tmp_Y[2] == 1:
            i_stride = stride
        elif tmp_Y[3] == 1:
            i_stride = stride//5
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            # x-->x.T
            out_X.append(tmp_ts[j:j+window_size].T)
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)
# stide: 500-->1000
def read_data_with_train_val_test(window_size=5000, stride=500):
    # read pkl
    with open('./ecg_data/CPSC/cpsc_raw_1_1.pkl', 'rb') as fin:
        res = pickle.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std
    all_label = res['label']

    X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=0)
    # get a part
    X_train, X_test, Y_train, Y_test = X_train[:80], X_test[:10], Y_train[:80], Y_test[:10]

    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)

    # slide and cut
    print('before: ')
    for i in range(4):
        print('Trian_{}:'.format(3-i), Counter(Y_train[:, i]),' Val_{}:'.format(3-i),Counter(Y_val[:, i]),' Test_{}:'.format(3-i), Counter(Y_test[:, i]))
    # print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_val, Y_val, pid_val = slide_and_cut(X_val, Y_val, window_size=window_size, stride=stride, output_pid=True)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    for i in range(4):
        print('Trian_{}:'.format(3-i), Counter(Y_train[:, i]),' Val_{}:'.format(3-i),Counter(Y_val[:, i]),' Test_{}:'.format(3-i), Counter(Y_test[:, i]))
    # print(Counter(Y_train), Counter(Y_val), Counter(Y_test))

    # # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    # X_train = np.expand_dims(X_train, 1)
    # X_val = np.expand_dims(X_val, 1)
    # X_test = np.expand_dims(X_test, 1)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, pid_val, pid_test

if __name__ == '__main__':

    # read_data_with_train_val_test()
    #merge_db()
    pass

