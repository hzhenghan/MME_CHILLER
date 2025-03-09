import numpy as np
import os
import os.path
import pandas as pd
import torch.utils.data
import xlrd
import openpyxl
from sklearn import preprocessing


class FaultDataset(torch.utils.data.Dataset):
    def __init__(self,
                data,
                fea_cols=['TWE_set', 'TEI', 'TWEI', 'TEO', 'TWEO', 'TCI', 'TWCI', 'TCO', 'TWCO',
                'TSI', 'TSO', 'TBI', 'TBO', 'Cond Tons', 'Cooling Tons', 'Shared Cond Tons',
                'Cond Energy Balance', 'Evap Tons', 'Shared Evap Tons', 'Building Tons',
                'Evap Energy Balance', 'kW', 'COP', 'kW/Ton', 'FWC', 'FWE', 'TEA', 'TCA',
                'TRE', 'PRE', 'TRC', 'PRC', 'TRC_sub', 'T_suc', 'Tsh_suc', 'TR_dis', 'Tsh_dis',
                'P_lift', 'Amps', 'RLA%',  'Tolerance%','Heat Balance%',
                'Unit Status', 'Active Fault', 'TO_sump', 'TO_feed', 'PO_feed', 'PO_net',
                'TWCD', 'TWED', 'VSS', 'VSL', 'VH', 'VM', 'VC', 'VE', 'VW', 'TWI', 'TWO',
                'THI', 'THO', 'FWW', 'FWH', 'FWB'
                ],
                 dataset='A',
                 test=True,
                 return_id=True
                 ):
        if dataset != 'A':
            fea_cols=['冷却泵功率', '冷冻水供水流量', '冷冻泵功率', '冷却水供回水压差', '冷却塔风机功率', '新风风机功率',
                    '压缩机功率', '排风风机功率', '送风风机功率', '冷冻水供水压力', '冷冻水回水流量', '冷却水进水流量',
                    '冷却塔风速', '新风风压', '排风风速', '排风风压', '冷冻水回水压力', '冷却水出水流量', '新风流量',
                    '排风流量', '送风风压', '冷凝器制冷剂进口温度', '冷却塔进口温度', '蒸发器出口温度', '冷凝器进口温度', '房间温度', '冷凝器制冷剂出口温度', '冷却塔出口温度', '蒸发器进口温度',
                    '冷凝器出口温度', '室外温度', '二氧化碳浓度', '新风风速', '送风风速', '新风温度', '表冷器进风温度',
                    '房间湿度', '排风温度', '表冷器出风口温度']

        self.fea_data = data[fea_cols].values
        self.labels = data['label'].values
        self.test = test
        self.return_id = return_id

    def __getitem__(self, index):
        # path = os.path.join(self.root, self.imgs[index])
        # target = self.labels[index]

        data = self.fea_data[index]
        target = self.labels[index]

        if not self.test:
            return data, target
        elif self.return_id:
            return data, target, index
        else:
            return data, target, self.fea_data[index]

    def __len__(self):
        return len(self.fea_data)

    def process(self):
        pass
