from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t
from torch.nn.parameter import Parameter


class ConvShared(nn.Conv1d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0, dilation: _size_1_t = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty(
            (self.out_channels, 1, *self.kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(self.out_channels, **factory_kwargs))

    def forward(self, input: Tensor) -> Tensor:
        repeat = [1] * len(self.weight.shape)
        repeat[1] = self.in_channels // self.groups
        return self._conv_forward(input, self.weight.repeat(*repeat), self.bias)


CONV_TYPE = 'normal'


def set_conv_type(args):
    global CONV_TYPE
    CONV_TYPE = args.conv_type


def conv1d(*args, **kwargs):
    global CONV_TYPE
    if CONV_TYPE == 'share':
        kwargs['groups'] = kwargs['in_channels']
        kwargs['out_channels'] = int(np.ceil(kwargs['out_channels'] / kwargs['in_channels'])) * kwargs['in_channels']
        return ConvShared(*args, **kwargs)
    elif CONV_TYPE == 'depth':
        kwargs['groups'] = kwargs['in_channels']
        kwargs['out_channels'] = int(np.ceil(kwargs['out_channels'] / kwargs['in_channels'])) * kwargs['in_channels']
        return nn.Conv1d(*args, **kwargs)
    else:
        return nn.Conv1d(*args, **kwargs)



def phi_(phi_c, x, lb=0, ub=1):
    mask = np.logical_or(x < lb, x > ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)




def train(model, train_loader, optimizer, epoch, device, verbose=0,
          lossFn=None, lr_schedule=None,
          post_proc=lambda args: args):
    if lossFn is None:
        lossFn = nn.MSELoss()

    model.train()

    total_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        bs = len(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        target = post_proc(target)
        output = post_proc(output)
        loss = lossFn(output.view(bs, -1), target.view(bs, -1))

        loss.backward()
        optimizer.step()
        total_loss += loss.sum().item()
    if lr_schedule is not None: lr_schedule.step()

    if verbose > 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))

    return total_loss / len(train_loader.dataset)


def test(model, test_loader, device, verbose=0, lossFn=None,
         post_proc=lambda args: args):
    model.eval()
    if lossFn is None:
        lossFn = nn.MSELoss()

    total_loss = 0.
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            bs = len(data)

            data, target = data.to(device), target.to(device)
            output = model(data)
            output = post_proc(output)

            loss = lossFn(output.view(bs, -1), target.view(bs, -1))
            total_loss += loss.sum().item()

    return total_loss / len(test_loader.dataset)


# Till EoF
# taken from FNO paper:
# https://github.com/zongyi-li/fourier_neural_operator

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
