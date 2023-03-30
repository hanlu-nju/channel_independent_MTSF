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
        self.flat_input = configs.flat_input
        self.individual = configs.individual
        self.Linear = self.get_linear_layer(configs)

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def get_linear_layer(self, configs):
        if self.individual:
            return nn.ModuleList(nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels))
        else:
            if self.flat_input:
                input_dim = self.seq_len * self.channels
                output_dim = self.pred_len * self.channels
            else:
                input_dim = self.seq_len
                output_dim = self.pred_len
            if configs.low_rank:
                max_rank = min(input_dim, output_dim)
                rank = max(max_rank // configs.rank_ratio, 1)
                return nn.Sequential(nn.Linear(input_dim, rank, bias=False),
                                     nn.Linear(rank, output_dim))
            else:
                return nn.Linear(input_dim, output_dim)
                # if configs.low_rank:
                #     return nn.Sequential(nn.Linear(self.seq_len, configs.rank, bias=False),
                #                          nn.Linear(configs.rank, self.pred_len))
                # else:
                #     return nn.Linear(self.seq_len, self.pred_len)

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        # x: [Batch, Input length, Channel]
        if self.individual:
            o = torch.zeros([batch_x.size(0), self.pred_len, batch_x.size(2)],
                            dtype=batch_x.dtype).to(batch_x.device)
            for i in range(self.channels):
                o[:, :, i] = self.Linear[i](batch_x[:, :, i])
        else:
            if self.flat_input:
                bz = batch_x.size(0)
                o = self.Linear(batch_x.view(bz, -1)).reshape(bz, self.pred_len, self.channels)
            else:
                o = self.Linear(batch_x.permute(0, 2, 1)).permute(0, 2, 1)
        return o  # [Batch, Output length, Channel]
