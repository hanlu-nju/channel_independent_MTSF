import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', channel_independent=False, train_frac=1.0, transform='none',
                 p=0.5,
                 drop_channel='0', device=torch.device('cuda')):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.train_frac = train_frac
        self.transform = transform
        self.drop_channel = drop_channel
        self.channel_independent = channel_independent
        self.device = device
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        border1s, border2s, df_raw = self.get_borders(df_raw)
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
            self.mean = torch.from_numpy(self.scaler.scale_).float().to(self.device)
            self.var = torch.from_numpy(self.scaler.var_).float().to(self.device)
        else:
            data = df_data.values

        if self.set_type == 0:
            total_len = border2s[0] - border1s[0] - self.seq_len - self.pred_len + 1
            border1 = int(total_len * (1.0 - self.train_frac))

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.single_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        self.data_stamp = data_stamp
        self.channels = self.data_x.shape[-1]

    def get_borders(self, df_raw):
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        return border1s, border2s, df_raw

    def __getitem__(self, index):
        if self.channel_independent:
            s_begin = index // self.channels
            channel = index % self.channels
            data_x = self.data_x[:, channel:channel + 1]
            data_y = self.data_y[:, channel:channel + 1]
        else:
            s_begin = index
            data_x = self.data_x
            data_y = self.data_y

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.single_len * self.channels if self.channel_independent else self.single_len

    def inverse_transform(self, data):
        if self.scale:
            if self.scaler.with_std:
                data *= self.var
            if self.scaler.with_mean:
                data += self.mean
        return data


class Dataset_ETT_hour(Dataset_Custom):

    def get_borders(self, df_raw):
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        return border1s, border2s, df_raw


class Dataset_ETT_minute(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTm1.csv', target='OT', scale=True,
                 timeenc=0, freq='t', train_frac=1.0, transform='none', channel_independent=False, drop_channel='0',
                 device=torch.device('cuda')):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq, channel_independent,
                         train_frac, transform, p=0.5, drop_channel=drop_channel, device=device)

    def get_borders(self, df_raw):
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        return border1s, border2s, df_raw


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, channel_independent=False,
                 train_frac=1.0,
                 transform='none', p=0.5, drop_channel='0'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
            self.mean = torch.from_numpy(self.scaler.scale_).float()
            self.var = torch.from_numpy(self.scaler.var_).float()
        else:
            data = df_data.values
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        if self.scale:
            if self.scaler.with_std:
                data *= self.var.to(data.device)
            if self.scaler.with_mean:
                data += self.mean.to(data.device)
        return data

# class Dataset_ETT_hour(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
#         border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_ETT_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTm1.csv',
#                  target='OT', scale=True, timeenc=0, freq='t'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
#         border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.timeenc = timeenc
#         self.freq = freq
#
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         cols = list(df_raw.columns)
#         cols.remove(self.target)
#         cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         # print(cols)
#         num_train = int(len(df_raw) * 0.7)
#         num_test = int(len(df_raw) * 0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
#         border2s = [num_train, num_train + num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
#
#
# class Dataset_Pred(Dataset):
#     def __init__(self, root_path, flag='pred', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['pred']
#
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols = cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         if self.cols:
#             cols = self.cols.copy()
#             cols.remove(self.target)
#         else:
#             cols = list(df_raw.columns)
#             cols.remove(self.target)
#             cols.remove('date')
#         df_raw = df_raw[['date'] + cols + [self.target]]
#         border1 = len(df_raw) - self.seq_len
#         border2 = len(df_raw)
#
#         if self.features == 'M' or self.features == 'MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features == 'S':
#             df_data = df_raw[[self.target]]
#
#         if self.scale:
#             self.scaler.fit(df_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
#
#         tmp_stamp = df_raw[['date']][border1:border2]
#         tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
#         pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
#
#         df_stamp = pd.DataFrame(columns=['date'])
#         df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
#         if self.timeenc == 0:
#             df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
#             df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
#             df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
#             df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
#             df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
#             df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
#             data_stamp = df_stamp.drop(['date'], 1).values
#         elif self.timeenc == 1:
#             data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
#             data_stamp = data_stamp.transpose(1, 0)
#
#         self.data_x = data[border1:border2]
#         if self.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
#
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len
#
#         seq_x = self.data_x[s_begin:s_end]
#         if self.inverse:
#             seq_y = self.data_x[r_begin:r_begin + self.label_len]
#         else:
#             seq_y = self.data_y[r_begin:r_begin + self.label_len]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         return len(self.data_x) - self.seq_len + 1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)
