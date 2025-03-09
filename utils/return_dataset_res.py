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

def split_data(data: str = 'A', domain: int = None) -> pd.DataFrame:

    if data == 'A':
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

        fault_data_all = [normal,  fwe40, rl40, ro40, eo50, cf45, fwc40, nc5]

        for i, df in enumerate(fault_data_all):
            df['label'] = i

        fault_data = pd.concat(fault_data_all, ignore_index=True)

        # if domain == 45:
        #     fault_data = fault_data[(fault_data['TWE_set'] == domain) | (fault_data['TWE_set'] == 44)].reset_index(drop=True)
        # else:
        fault_data = fault_data[(fault_data['TWE_set'] == domain)].reset_index(drop=True)
    else:
        use_cols = ['冷却泵功率', '冷冻水供水流量', '冷冻泵功率', '冷却水供回水压差', '冷却塔风机功率', '新风风机功率',
                    '压缩机功率', '排风风机功率', '送风风机功率', '冷冻水供水压力', '冷冻水回水流量', '冷却水进水流量',
                    '冷却塔风速', '新风风压', '排风风速', '排风风压', '冷冻水回水压力', '冷却水出水流量', '新风流量',
                    '排风流量', '送风风压', '冷凝器制冷剂进口温度', '冷却塔进口温度', '蒸发器出口温度',
                    '冷凝器进口温度', '房间温度', '冷凝器制冷剂出口温度', '冷却塔出口温度', '蒸发器进口温度',
                    '冷凝器出口温度', '室外温度', '二氧化碳浓度', '新风风速', '送风风速', '新风温度', '表冷器进风温度',
                    '房间湿度', '排风温度', '表冷器出风口温度']

        normal_d1 = pd.read_excel('data_chiller/HY-31C/d1/正常.xlsx', usecols=use_cols)
        FWE_l2_d1 = pd.read_excel('data_chiller/HY-31C/d1/故障1 等级2.xlsx', usecols=use_cols)
        FWC_l2_d1 = pd.read_excel('data_chiller/HY-31C/d1/故障2 等级2.xlsx', usecols=use_cols)
        FWE_l1_d1 = pd.read_excel('data_chiller/HY-31C/d1/故障1 等级1.xlsx', usecols=use_cols)
        FWC_l1_d1 = pd.read_excel('data_chiller/HY-31C/d1/故障2 等级1.xlsx', usecols=use_cols)


        normal_d2 = pd.read_excel('data_chiller/HY-31C/d2/normal.xlsx', usecols=use_cols)
        FWC_l2_d2 = pd.read_excel('data_chiller/HY-31C/d2/condenser_l2.xlsx', usecols=use_cols)
        FWE_l2_d2 = pd.read_excel('data_chiller/HY-31C/d2/evaporator_l2.xlsx', usecols=use_cols)
        FWE_l1_d2 = pd.read_excel('data_chiller/HY-31C/d2/condenser_l1.xlsx', usecols=use_cols)
        FWC_l1_d2 = pd.read_excel('data_chiller/HY-31C/d2/condenser_l1.xlsx', usecols=use_cols)

        normal_d3 = pd.read_excel('data_chiller/HY-31C/d3/normal1.xlsx', usecols=use_cols)
        FWC_l2_d3 = pd.read_excel('data_chiller/HY-31C/d3/condenser_l2.xlsx', usecols=use_cols)
        FWE_l2_d3 = pd.read_excel('data_chiller/HY-31C/d3/evaporator_l2.xlsx', usecols=use_cols)
        FWE_l1_d3 = pd.read_excel('data_chiller/HY-31C/d3/condenser_l1.xlsx', usecols=use_cols)
        FWC_l1_d3 = pd.read_excel('data_chiller/HY-31C/d3/condenser_l1.xlsx', usecols=use_cols)

        fault_data_d1 = [normal_d1, FWE_l2_d1, FWC_l2_d1, FWE_l1_d1, FWC_l1_d1]
        fault_data_d2 = [normal_d2, FWE_l2_d2, FWC_l2_d2, FWE_l1_d2, FWC_l1_d2]
        fault_data_d3 = [normal_d3, FWE_l2_d3, FWC_l2_d3, FWE_l1_d3, FWC_l1_d3]

        for i, df in enumerate(fault_data_d1):
            df['label'] = i

        for i, df in enumerate(fault_data_d2):
            df['label'] = i

        for i, df in enumerate(fault_data_d3):
            df['label'] = i


        if domain == 1:
            fault_data = fault_data_d1
        elif domain == 2:
            fault_data = fault_data_d2
        elif domain == 3:
            fault_data = fault_data_d3

    fault_data = pd.concat(fault_data, ignore_index=True)

    print('Source data shape is {}'.format(fault_data.shape))
    fault_data = minmax(fault_data)

    fault_data_normal = fault_data[fault_data['label'] == 0]
    fault_data_fault = fault_data[fault_data['label'] != 0]
    # split_ratio = 0.7

    # train_data_normal = fault_data_normal.sample(frac=split_ratio, random_state=42)
    train_data_normal = fault_data_normal
    # test_data_normal = fault_data_normal.drop(train_data_normal.index)

    return train_data_normal,  fault_data_fault

def return_dataset_res(data, domain):

    train_data_normal,  fault_data_fault = split_data(data, domain)
    # s_fault_data, labeled_target, unlabeled_target, val_target = get_data(50, 40, 1)

    train_data_normal_i = FaultDataset(train_data_normal, dataset='H', test=False)
    # test_data_normal = FaultDataset(test_data_normal)
    fault_data_fault_i = FaultDataset(fault_data_fault, dataset='H', test=False)

    return train_data_normal_i, fault_data_fault_i, train_data_normal