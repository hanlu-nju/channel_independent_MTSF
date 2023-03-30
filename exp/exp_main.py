import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Transformer, DeepVAR, NHITS, DLinear, NLinear, Linear, TCN, MLP
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
# import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'DeepVAR': DeepVAR,
            'NHITS': NHITS,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'TCN': TCN,
            'MLP': MLP,
        }
        if self.args.channel_independent:
            self.args.c_out = self.args.enc_in = self.args.dec_in = 1

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.wd)
        return model_optim

    def _select_criterion(self, criterion_str):
        criterion_dict = {"mse": nn.MSELoss(),
                          "mae": nn.L1Loss(),
                          "huber": nn.SmoothL1Loss(beta=self.args.huber_beta)}
        criterion = criterion_dict[criterion_str]
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if self.args.pred_residual:
                    seq_last = batch_x[:, -1:, :].detach()
                    # batch_x = batch_x - seq_last
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # if self.args.channel_independent:`
                #     b, l, c = batch_x.shape
                #     batch_x = batch_x.permute(0, 2, 1).reshape(-1, batch_x.size(1), 1)
                #     # (b,)
                #     batch_x_mark = batch_x_mark.view(1, *batch_x_mark.shape) \
                #         .repeat(c, 1, 1, 1) \
                #         .view(-1, batch_x_mark.size(-2), batch_x_mark.size(-1))
                #     dec_inp = dec_inp.permute(0, 2, 1).reshape(-1, dec_inp.size(1), 1)
                #     batch_y_mark = batch_y_mark.view(1, *batch_y_mark.shape) \
                #         .repeat(c, 1, 1, 1) \
                #         .view(-1, batch_y_mark.size(-2), batch_y_mark.size(-1))
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # if self.args.channel_independent:
                #     outputs = outputs.reshape(b, c, -1).permute(0, 2, 1)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.pred_residual:
                    outputs = outputs + seq_last
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.train_loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader),
                                                                          total=min(len(train_loader),
                                                                                    self.args.max_iter)):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                if self.args.pred_residual:
                    seq_last = batch_x[:, -1:, :].detach()
                    # batch_x = batch_x - seq_last

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # mask = torch.ones((batch_x.size(0), self.args.label_len, batch_x.size(2)), device=self.device)
                # if self.args.mask_ratio > 0:
                #     select_idx = torch.from_numpy(
                #         np.random.choice(self.args.label_len, size=int(self.args.mask_ratio * self.args.label_len),
                #                          replace=False)).to(
                #         self.device)
                #     mask[:, select_idx, :] = 0.0
                # batch_x[:, -self.args.label_len:, :] = batch_x[:, -self.args.label_len:, :] * mask
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(
                    self.device)
                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # f_dim = -1 if self.args.features == 'MS' else 0
                        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # loss = criterion(outputs, batch_y)
                        # train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # if self.args.mask_ratio > 0:
                #     label_selected_idx = torch.cat([select_idx,
                #                                     torch.arange(self.args.label_len,
                #                                                  self.args.label_len + self.args.pred_len,
                #                                                  device=self.device)])
                #     batch_y = batch_y[:, label_selected_idx, f_dim:].to(self.device)
                #     outputs = outputs[:, label_selected_idx, f_dim:]
                # else:
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.inverse_scale:
                    outputs = train_data.inverse_transform(outputs)  # outputs.detach().cpu().numpy()  # .squeeze()
                    batch_y = train_data.inverse_transform(batch_y)  # batch_y.detach().cpu().numpy()  # .squeeze()
                if self.args.pred_residual:
                    outputs = outputs + seq_last
                loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())
                # wandb.log({'train loss': loss.item()})
                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                if i >= self.args.max_iter:
                    break

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            # wandb.log({'valid loss': vali_loss.item(), 'test loss': test_loss.item()})
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        import shutil
        shutil.rmtree(path)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader),
                                                                          total=len(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.pred_residual:
                    seq_last = batch_x[:, -1:, :].detach()
                    # batch_x = batch_x - seq_last
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # outputs = outputs.detach().cpu().numpy()
                # batch_y = batch_y.detach().cpu().numpy()
                if self.args.inverse_scale or self.args.eval_rescale:
                    outputs = test_data.inverse_transform(outputs)  # outputs.detach().cpu().numpy()  # .squeeze()
                    batch_y = test_data.inverse_transform(batch_y)  # batch_y.detach().cpu().numpy()  # .squeeze()
                if self.args.pred_residual:
                    outputs = outputs + seq_last
                pred = outputs.detach().cpu().numpy()  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y.detach().cpu().numpy()  # batch_y.detach().cpu().numpy()  # .squeeze()
                inputx.append(batch_x.detach().cpu().numpy())
                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        inputx = np.concatenate(inputx)
        if self.args.channel_independent:
            total = preds.shape[0]
            batch = total // test_data.channels
            preds = preds.reshape((batch, test_data.channels, -1)).transpose(0, 2, 1)
            trues = trues.reshape((batch, test_data.channels, -1)).transpose(0, 2, 1)
            inputx = inputx.reshape((batch, test_data.channels, -1)).transpose(0, 2, 1)
        print('test shape:', preds.shape, trues.shape)

        os.path.join('./checkpoints/' + setting, 'checkpoint.pth')

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(os.path.join(folder_path, "prediction.npy"), preds)
        # np.save(os.path.join(folder_path, "trues.npy"), trues)
        # np.save(os.path.join(folder_path, "inputx.npy"), inputx)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
