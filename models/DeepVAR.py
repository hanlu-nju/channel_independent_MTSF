# @author : ThinkPad 
# @date : 2022/10/20
import torch.nn as nn

from layers.Embed import DataEmbedding_wo_pos


class Model(nn.Module):
    """
    DeepVAR model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        # self.enc_embedding = TokenEmbedding(configs.enc_in, configs.d_model)
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.lstm = nn.LSTM(input_size=configs.d_model,
                            hidden_size=configs.d_model,
                            num_layers=configs.e_layers,
                            bias=True,
                            dropout=configs.dropout,
                            batch_first=True)
        self.adapterW = nn.Linear(configs.d_model, self.pred_len * configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # Parse windows_batch
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # Flatten inputs [B, W, C, L+H] -> [B, W, C*(L+H)]
        # Concatenate [ Y_t, | X_{t-L},..., X_{t} | F_{t-L},..., F_{t+H} | S ]
        batch_size, windows_size = enc_out.shape[:2]

        # LSTM forward
        dec, _ = self.lstm(enc_out)
        out = self.adapterW(dec[:, -1, :])

        return out.view(batch_size, self.pred_len, -1)
