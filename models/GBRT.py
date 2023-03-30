# @author : ThinkPad 
# @date : 2022/10/20
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb


class Model:
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
        model = xgb.XGBRegressor(learning_rate=0.2,
                                 n_estimators=50,
                                 max_depth=8,
                                 min_child_weight=1,
                                 gamma=0.0,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 scale_pos_weight=1,
                                 seed=42)

        self.model = MultiOutputRegressor(model)

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def fit(self, data_x, data_y):
        self.model.fit(data_x, data_y)

    def predict(self, data_x):
        return self.model.predict(data_x)
