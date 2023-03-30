import os
import warnings

import numpy as np
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import GBRT
from utils.metrics import metric, MSE

warnings.filterwarnings('ignore')



def create_sub_series(series: np.ndarray, window_len: int, horizon: int, glob=False):
    subseries = sliding_window_view(series, window_len + horizon, axis=0)
    batch = subseries.shape[0]
    channel = subseries.shape[1]
    X, Y = subseries[:, :, :window_len], subseries[:, :, window_len:]
    if glob:
        X = X.reshape((batch * channel, -1))
        Y = Y.reshape((batch * channel, -1))
    else:
        X = X.reshape((batch, -1))
        Y = Y.reshape((batch, -1))
    return X, Y


class Exp_GBRT(Exp_Basic):
    def __init__(self, args):
        super(Exp_GBRT, self).__init__(args)
        if self.args.channel_independent:
            # if self.args.channel_independent:
            #     self.args.batch_size = int(np.ceil(self.args.batch_size / self.args.enc_in))
            self.args.c_out = self.args.enc_in = self.args.dec_in = 1
        self.model = GBRT.Model(self.args)

    def _build_model(self):
        return nn.Identity()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_X, train_Y = create_sub_series(train_data.data_x, self.args.seq_len, self.args.pred_len,
                                             self.args.channel_independent)
        val_X, val_Y = create_sub_series(vali_data.data_x, self.args.seq_len, self.args.pred_len,
                                         self.args.channel_independent)
        test_X, test_Y = create_sub_series(test_data.data_x, self.args.seq_len, self.args.pred_len,
                                           self.args.channel_independent)
        print("data loaded")
        self.model.fit(train_X, train_Y)
        print("model fitted")
        val_pred = self.model.predict(val_X)
        test_pred = self.model.predict(test_X)

        print(f'valid loss : {MSE(val_pred, val_Y)}, test loss:  {MSE(test_Y, test_pred)}')
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        test_X, test_Y = create_sub_series(test_data.data_x, self.args.seq_len, self.args.pred_len,
                                           self.args.channel_independent)

        preds = self.model.predict(test_X)
        trues = test_Y
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

        return
