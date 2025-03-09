import numpy as np
import os
import os.path
import pandas as pd
import torch
import xlrd
import openpyxl
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from utils.loader import FaultDataset
from torch.autograd import Variable


def minmax(data) -> pd.DataFrame:
    label_column = data['label']

    normalized_data = data.drop(columns=['label'])

    scaler = preprocessing.MinMaxScaler()
    normalized_data = scaler.fit_transform(normalized_data)

    columns = data.columns[:-1]
    normalized_df = pd.DataFrame(normalized_data, columns=columns)

    normalized_df['label'] = label_column.values

    return normalized_df

def split_data(s, t) -> pd.DataFrame:

    use_cols = ['TWE_set',
                'TEI', 'TWEI', 'TEO', 'TWEO', 'TCI', 'TWCI', 'TCO', 'TWCO',
                'TSI', 'TSO', 'TBI', 'TBO', 'Cond Tons', 'Cooling Tons', 'Shared Cond Tons',
                'Cond Energy Balance', 'Evap Tons', 'Shared Evap Tons', 'Building Tons',
                'Evap Energy Balance', 'kW', 'COP', 'kW/Ton', 'FWC', 'FWE', 'TEA', 'TCA',
                'TRE', 'PRE', 'TRC', 'PRC', 'TRC_sub', 'T_suc', 'Tsh_suc', 'TR_dis', 'Tsh_dis',
                'P_lift', 'Amps', 'RLA%', 'Heat Balance%', 'Tolerance%',
                'Unit Status', 'Active Fault', 'TO_sump', 'TO_feed', 'PO_feed', 'PO_net',
                'TWCD', 'TWED', 'VSS', 'VSL', 'VH', 'VM', 'VC', 'VE', 'VW', 'TWI', 'TWO',
                'THI', 'THO', 'FWW', 'FWH', 'FWB'
                ]
    normal = pd.read_excel('data_chiller/Benchmark Tests/normal.xls', usecols=use_cols)
    fwc40 = pd.read_excel('data_chiller/Reduced condenser water flow/fwc40.xls', usecols=use_cols)
    fwe40 = pd.read_excel('data_chiller/Reduced evaporator water flow/fwe40.xls', usecols=use_cols)
    rl40 = pd.read_excel('data_chiller/Refrigerant leak/rl40.xls', usecols=use_cols)
    ro40 = pd.read_excel('data_chiller/Refrigerant overcharge/ro40.xls', usecols=use_cols)
    eo50 = pd.read_excel('data_chiller/Excess oil/eo50.xls', usecols=use_cols)
    cf45 = pd.read_excel('data_chiller/Condenser fouling/cf45.xls', usecols=use_cols)
    nc5 = pd.read_excel('data_chiller/Non-condensables in refrigerant/nc5.xls', usecols=use_cols)

    fault_data_all = [normal, fwc40, fwe40, rl40, ro40, eo50, cf45, nc5]

    for i, df in enumerate(fault_data_all):
        df['label'] = i

    fault_data = pd.concat(fault_data_all, ignore_index=True)

    # (fault_data['TWE_set'] == 40) | (fault_data['TWE_set'] == 39)
    # (fault_data['TWE_set'] == 50) | (fault_data['TWE_set'] == 49)

    s_fault_data = fault_data[(fault_data['TWE_set'] == s)]
    t_fault_data = fault_data[(fault_data['TWE_set'] == t)]

    print('Source data shape is {}'.format(s_fault_data.shape))
    print('Target data shape is {}'.format(t_fault_data.shape))

    s_fault_data = minmax(s_fault_data)
    t_fault_data = minmax(t_fault_data)

    train = pd.DataFrame(columns=t_fault_data.columns)
    val = pd.DataFrame(columns=t_fault_data.columns)
    test = pd.DataFrame(columns=t_fault_data.columns)

    for label in t_fault_data['label'].unique():
        label_data = t_fault_data[t_fault_data['label'] == label]
        np.random.seed(777)
        label_data = label_data.reset_index(drop=True)
        # print(label_data.index.tolist())
        sample_indices = np.arange(len(label_data))
        np.random.shuffle(sample_indices)

        # index_1 = shot+3
        train_indices = sample_indices[:1]
        val_indices = sample_indices[20:23]
        test_indices = sample_indices[23:]

        train = pd.concat([train, label_data.iloc[train_indices]])
        val = pd.concat([val, label_data.iloc[val_indices]])
        test = pd.concat([test, label_data.iloc[test_indices]])

    # target data was splited to three parts.
    labeled_target_3 = train.reset_index()
    unlabeled_target = test.reset_index()
    val_target_3 = val.reset_index()
    print('labeled_target shape is {}'.format(labeled_target_3.shape))
    print('unlabeled_target shape is {}'.format(unlabeled_target.shape))
    print('val_target_3 shape is {}'.format(val_target_3.shape))

    return s_fault_data, labeled_target_3, unlabeled_target, val_target_3

# def get_data(source, target, num):
#     # labeled_source_40
#     s_fault_data_path = 'data_chiller/data/labeled_source_%s.csv' % source
#     labeled_target_path = 'data_chiller/data/labeled_target_%s_%s.csv' % (target, num)
#     unlabeled_target_path = 'data_chiller/data/unlabeled_target_%s_%s.csv' % (target, num)
#     val_target_path = 'data_chiller/data/val_target_%s_3.csv' % (target)
#
#
#     s_fault_data = pd.read_csv(s_fault_data_path)
#     labeled_target = pd.read_csv(labeled_target_path)
#     unlabeled_target = pd.read_csv(unlabeled_target_path)
#     val_target = pd.read_csv(val_target_path)
#
#     return s_fault_data, labeled_target, unlabeled_target, val_target

def split_data_res(s, t):

    s_fault_data = pd.read_csv('data_chiller/data_res_feature/fault_data_{}.csv'.format(s))
    t_fault_data = pd.read_csv('data_chiller/data_res_feature/fault_data_{}.csv'.format(t))
    # t_fault_data = t_fault_data.sample(frac=0.5, random_state=1)


    print('Source data shape is {}'.format(s_fault_data.shape))
    print('Target data shape is {}'.format(t_fault_data.shape))

    train = pd.DataFrame(columns=t_fault_data.columns)
    val = pd.DataFrame(columns=t_fault_data.columns)
    test = pd.DataFrame(columns=t_fault_data.columns)

    for label in t_fault_data['label'].unique():
        label_data = t_fault_data[t_fault_data['label'] == label]
        np.random.seed(777)
        label_data = label_data.reset_index(drop=True)
        # print(label_data.index.tolist())
        sample_indices = np.arange(len(label_data))
        np.random.shuffle(sample_indices)

        # index_1 = shot+3
        train_indices = sample_indices[:1]
        val_indices = sample_indices[5:8]
        test_indices = sample_indices[8:]

        train = pd.concat([train, label_data.iloc[train_indices]])
        val = pd.concat([val, label_data.iloc[val_indices]])
        test = pd.concat([test, label_data.iloc[test_indices]])

    # target data was splited to three parts.
    labeled_target_3 = train.reset_index()
    val_target_3 = val.reset_index()
    unlabeled_target = test.reset_index(drop=True)

    print('labeled_target shape is {}'.format(labeled_target_3.shape))
    print('unlabeled_target shape is {}'.format(unlabeled_target.shape))
    print('val_target_3 shape is {}'.format(val_target_3.shape))

    return s_fault_data, labeled_target_3, unlabeled_target, val_target_3


def return_dataset(s, t, dataset=None):
    s_fault_data, labeled_target, unlabeled_target, val_target = split_data_res(s, t)
    # s_fault_data, labeled_target, unlabeled_target, val_target = split_data(s, t)
    # s_fault_data, labeled_target, unlabeled_target, val_target = get_data(50, 40, 1)
    source_dataset = FaultDataset(s_fault_data, dataset=dataset)
    target_dataset = FaultDataset(labeled_target, dataset=dataset)
    target_dataset_val = FaultDataset(val_target, dataset=dataset)
    target_dataset_unl = FaultDataset(unlabeled_target, dataset=dataset)
    target_dataset_test = FaultDataset(unlabeled_target, dataset=dataset)

    return source_dataset, target_dataset, target_dataset_val, target_dataset_unl, target_dataset_test

# if __name__ == '__main__':
#     s_fault_data, labeled_target_3, unlabeled_target, val_target_3 = split_data()
#     print(unlabeled_target.index.tolist())
