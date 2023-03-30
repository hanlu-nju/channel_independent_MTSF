# @author : ThinkPad 
# @date : 2023/3/13
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        input_dim = configs.seq_len * configs.enc_in
        out_dim = configs.pred_len * configs.enc_in
        hidden = min((input_dim + out_dim) // 8, 512)
        self.networks = nn.Sequential(nn.Linear(input_dim, hidden),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(hidden, out_dim))

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # x: [Batch, Input length, Channel]
        bz = batch_x.size(0)
        o = self.networks(batch_x.view(bz, -1)).reshape(bz, self.pred_len, self.channels)
        return o  # [Batch, Output length, Channel]
