# @author : ThinkPad 
# @date : 2022/10/21


# %% ../../nbs/models.nhits.ipynb 8
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ['linear', 'nearest']) or ('cubic' in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = theta[:, :self.backcast_size]
        knots = theta[:, self.backcast_size:]

        if self.interpolation_mode == 'nearest':
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == 'linear':
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif 'cubic' in self.interpolation_mode:
            batch_size = len(backcast)
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(knots[i * batch_size:(i + 1) * batch_size], size=self.forecast_size,
                                           mode='bicubic')
                forecast[i * batch_size:(i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        return backcast, forecast


# %% ../../nbs/models.nhits.ipynb 9
ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']

POOLING = ['MaxPool1d',
           'AvgPool1d']


# class NHITSBlock(nn.Module):
#     """
#     N-HiTS block which takes a basis function as an argument.
#     """
#
#     def __init__(self,
#                  input_size: int,
#                  h: int,
#                  n_theta: int,
#                  mlp_units: list,
#                  basis: nn.Module,
#                  futr_exog_size: int,
#                  hist_exog_size: int,
#                  stat_exog_size: int,
#                  n_pool_kernel_size: int,
#                  pooling_mode: str,
#                  dropout_prob: float,
#                  activation: str):
#         """
#         """
#         super().__init__()
#
#         pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
#         pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))
#
#         input_size = pooled_hist_size + \
#                      hist_exog_size * pooled_hist_size + \
#                      futr_exog_size * (pooled_futr_size) + stat_exog_size
#
#         self.dropout_prob = dropout_prob
#         self.futr_exog_size = futr_exog_size
#         self.hist_exog_size = hist_exog_size
#         self.stat_exog_size = stat_exog_size
#
#         assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
#         assert pooling_mode in POOLING, f'{pooling_mode} is not in {POOLING}'
#
#         activ = getattr(nn, activation)()
#
#         self.pooling_layer = getattr(nn, pooling_mode)(kernel_size=n_pool_kernel_size,
#                                                        stride=n_pool_kernel_size, ceil_mode=True)
#
#         # Block MLPs
#         hidden_layers = [nn.Linear(in_features=input_size,
#                                    out_features=mlp_units[0][0])]
#         for layer in mlp_units:
#             hidden_layers.append(nn.Linear(in_features=layer[0],
#                                            out_features=layer[1]))
#             hidden_layers.append(activ)
#
#             if self.dropout_prob > 0:
#                 raise NotImplementedError('dropout')
#                 # hidden_layers.append(nn.Dropout(p=self.dropout_prob))
#
#         output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
#         layers = hidden_layers + output_layer
#         self.layers = nn.Sequential(*layers)
#         self.basis = basis
#
#     def forward(self, insample_y: torch.Tensor, futr_exog: torch.Tensor,
#                 hist_exog: torch.Tensor, stat_exog: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#
#         # Pooling
#         # Pool1d needs 3D input, (B,C,L), adding C dimension
#         insample_y = insample_y.unsqueeze(1)
#         insample_y = self.pooling_layer(insample_y)
#         insample_y = insample_y.squeeze(1)
#
#         # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
#         # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
#         batch_size = len(insample_y)
#         if self.hist_exog_size > 0:
#             hist_exog = hist_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
#             hist_exog = self.pooling_layer(hist_exog)
#             hist_exog = hist_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
#             insample_y = torch.cat((insample_y, hist_exog.reshape(batch_size, -1)), dim=1)
#
#         if self.futr_exog_size > 0:
#             futr_exog = futr_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
#             futr_exog = self.pooling_layer(futr_exog)
#             futr_exog = futr_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
#             insample_y = torch.cat((insample_y, futr_exog.reshape(batch_size, -1)), dim=1)
#
#         if self.stat_exog_size > 0:
#             insample_y = torch.cat((insample_y, stat_exog.reshape(batch_size, -1)), dim=1)
#
#         # Compute local projection weights and projection
#         theta = self.layers(insample_y)
#         backcast, forecast = self.basis(theta)
#         return backcast, forecast


# %% ../../nbs/models.nhits.ipynb 10
class NHITSBlock(nn.Module):
    """
    N-HiTS block which takes a basis function as an argument.
    """

    def __init__(self,
                 input_size: int,
                 h: int,
                 n_theta: int,
                 mlp_units: list,
                 basis: nn.Module,
                 futr_exog_size: int,
                 hist_exog_size: int,
                 stat_exog_size: int,
                 n_pool_kernel_size: int,
                 pooling_mode: str,
                 dropout_prob: float,
                 activation: str):
        """
        """
        super().__init__()

        pooled_hist_size = int(np.ceil(input_size / n_pool_kernel_size))
        pooled_futr_size = int(np.ceil((input_size + h) / n_pool_kernel_size))

        input_size = pooled_hist_size + \
                     hist_exog_size * pooled_hist_size + \
                     futr_exog_size * (pooled_futr_size) + stat_exog_size

        self.dropout_prob = dropout_prob
        self.futr_exog_size = futr_exog_size
        self.hist_exog_size = hist_exog_size
        self.stat_exog_size = stat_exog_size

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        assert pooling_mode in POOLING, f'{pooling_mode} is not in {POOLING}'

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(kernel_size=n_pool_kernel_size,
                                                       stride=n_pool_kernel_size, ceil_mode=True)

        # Block MLPs
        hidden_layers = [nn.Linear(in_features=input_size,
                                   out_features=mlp_units[0][0])]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0],
                                           out_features=layer[1]))
            hidden_layers.append(activ)

            if self.dropout_prob > 0:
                raise NotImplementedError('dropout')
                # hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        layers = hidden_layers + output_layer
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: torch.Tensor, futr_exog: torch.Tensor,
                hist_exog: torch.Tensor, stat_exog: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Pooling
        # Pool1d needs 3D input, (B,C,L), adding C dimension
        # insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        # insample_y = insample_y.squeeze(1)

        # Flatten MLP inputs [B, L+H, C] -> [B, (L+H)*C]
        # Contatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batch_size = len(insample_y)
        if self.hist_exog_size > 0:
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            hist_exog = self.pooling_layer(hist_exog)
            hist_exog = hist_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat((insample_y, hist_exog.reshape(batch_size, -1)), dim=1)

        if self.futr_exog_size > 0:
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, L, C] -> [B, C, L]
            futr_exog = self.pooling_layer(futr_exog)
            futr_exog = futr_exog.permute(0, 2, 1)  # [B, C, L] -> [B, L, C]
            insample_y = torch.cat((insample_y, futr_exog.reshape(batch_size, -1)), dim=1)

        if self.stat_exog_size > 0:
            insample_y = torch.cat((insample_y, stat_exog.reshape(batch_size, -1)), dim=1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y.reshape(batch_size, -1))
        backcast, forecast = self.basis(theta)
        return backcast, forecast


# %% ../../nbs/models.nhits.ipynb 10
class Model(nn.Module):
    def __init__(self, configs,
                 stack_types: list = ['identity', 'identity', 'identity'],
                 n_blocks: list = [1, 1, 1],
                 mlp_units: list = 3 * [[512, 512]],
                 n_pool_kernel_size: list = [2, 2, 1],
                 n_freq_downsample: list = [4, 2, 1],
                 pooling_mode: str = 'MaxPool1d',
                 interpolation_mode: str = 'linear',
                 dropout_prob_theta=0.,
                 activation='ReLU',
                 futr_exog_list=None,
                 hist_exog_list=None,
                 stat_exog_list=None,
                 step_size: int = 1,
                 **trainer_kwargs):
        """
        N-HiTS Model.

        **Parameters:**<br>
        `input_size`: int, insample_size.<br>
        `h`: int, Forecast horizon. <br>
        `shared_weights`: bool, If True, all blocks within each stack will share parameters. <br>
        `activation`: str, Activation function. An item from ['ReLU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'PReLU', 'Sigmoid']. <br>
        `futr_exog_list`: List[str], List of future exogenous variables. <br>
        `hist_exog_list`: List[str], List of historic exogenous variables. <br>
        `stack_types`: List[str], List of stack types. Subset from ['seasonality', 'trend', 'identity'].<br>
        `n_blocks`: List[int], Number of blocks for each stack. Note that len(n_blocks) = len(stack_types).<br>
        `mlp_units`: List[List[int]], Structure of hidden layers for each stack type. Each internal list should contain the number of units of each hidden layer. Note that len(n_hidden) = len(stack_types).<br>
        `n_harmonics`: int, Number of harmonic terms for trend stack type. Note that len(n_harmonics) = len(stack_types). Note that it will only be used if a trend stack is used.<br>
        `n_polynomials`: int, Number of polynomial terms for seasonality stack type. Note that len(n_polynomials) = len(stack_types). Note that it will only be used if a seasonality stack is used.<br>
        `dropout_prob_theta`: float, Float between (0, 1). Dropout for N-BEATS basis.<br>
        `learning_rate`: float, Learning rate between (0, 1).<br>
        `loss`: Callable, Loss to optimize.<br>
        `random_seed`: int, random_seed for pseudo random pytorch initializer and numpy random generator.<br>
        """
        super().__init__()
        self.h = configs.pred_len
        self.input_size = configs.enc_in
        self.step_size = step_size
        self.futr_exog_list = futr_exog_list if futr_exog_list is not None else []
        self.hist_exog_list = hist_exog_list if hist_exog_list is not None else []
        self.stat_exog_list = stat_exog_list if stat_exog_list is not None else []

        self.futr_exog_size = len(self.futr_exog_list)
        self.hist_exog_size = len(self.hist_exog_list)
        self.stat_exog_size = len(self.stat_exog_list)

        blocks = self.create_stack(stack_types=stack_types,
                                   n_blocks=n_blocks,
                                   input_size=self.input_size,
                                   h=self.h,
                                   mlp_units=mlp_units,
                                   n_pool_kernel_size=n_pool_kernel_size,
                                   n_freq_downsample=n_freq_downsample,
                                   pooling_mode=pooling_mode,
                                   interpolation_mode=interpolation_mode,
                                   dropout_prob_theta=dropout_prob_theta,
                                   activation=activation,
                                   futr_exog_size=self.futr_exog_size,
                                   hist_exog_size=self.hist_exog_size,
                                   stat_exog_size=self.stat_exog_size)
        self.blocks = torch.nn.ModuleList(blocks)

        # Adapter with Loss dependent dimensions
        # if self.loss.outputsize_multiplier > 1:
        #     self.out = nn.Linear(in_features=self.h,
        #                          out_features=self.h*self.loss.outputsize_multiplier)

    def create_stack(self, stack_types,
                     n_blocks,
                     input_size,
                     h,
                     mlp_units,
                     n_pool_kernel_size,
                     n_freq_downsample,
                     pooling_mode,
                     interpolation_mode,
                     dropout_prob_theta,
                     activation,
                     futr_exog_size, hist_exog_size, stat_exog_size):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):
                assert stack_types[i] == 'identity', f'Block type {stack_types[i]} not found!'

                n_theta = (input_size + max(h // n_freq_downsample[i], 1))
                basis = _IdentityBasis(backcast_size=input_size,
                                       forecast_size=h,
                                       interpolation_mode=interpolation_mode)

                nbeats_block = NHITSBlock(input_size=input_size,
                                          h=h,
                                          n_theta=n_theta,
                                          mlp_units=mlp_units,
                                          n_pool_kernel_size=n_pool_kernel_size[i],
                                          pooling_mode=pooling_mode,
                                          basis=basis,
                                          dropout_prob=dropout_prob_theta,
                                          activation=activation,
                                          futr_exog_size=futr_exog_size,
                                          hist_exog_size=hist_exog_size,
                                          stat_exog_size=stat_exog_size)

                # Select type of evaluation and apply it to all layers of block
                block_list.append(nbeats_block)

        return block_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Parse windows_batch

        # insample
        residuals = x_enc.flip(dims=(1,))  # backcast init
        forecast = x_enc[:, -1:, :]  # Level with Naive1

        block_forecasts = [forecast.repeat(1, self.h, 1)]

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, futr_exog=None,
                                             hist_exog=None, stat_exog=None)
            residuals = (residuals - backcast)
            forecast = forecast + block_forecast
            if self.decompose_forecast:
                block_forecasts.append(block_forecast)

        if self.decompose_forecast:
            # (n_batch, n_blocks, h)
            block_forecasts = torch.stack(block_forecasts)
            block_forecasts = block_forecasts.permute(1, 0, 2)
            return block_forecasts
        else:

            # Last dimension Adapter
            # if self.loss.outputsize_multiplier > 1:
            #     forecast = forecast[:,:,None] + \
            #                self.loss.adapt_output(self.out(forecast))
            return forecast
