import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.modified_mha import MultiheadAttention
from torch.nn import Parameter
from torch.nn.utils import weight_norm

class NAC(nn.Module):
    # based on https://github.com/arthurdouillard/nalu.pytorch/blob/master/nalu.py
    def __init__(self, in_dim, out_dim, init_fun=nn.init.xavier_uniform_):
        super().__init__()

        self._W_hat = nn.Parameter(torch.empty(in_dim, out_dim))
        self._M_hat = nn.Parameter(torch.empty(in_dim, out_dim))

        self.register_parameter('W_hat', self._W_hat)
        self.register_parameter('M_hat', self._M_hat)

        for param in self.parameters():
            init_fun(param)

    def forward(self, x):
        W = torch.tanh(self._W_hat) * torch.sigmoid(self._M_hat)
        return x.matmul(W)

class DenseLayer(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, output_dim=4096,
                 drop=0.0, hidden_act='relu', conv=False, cat_dim=1):
        super(DenseLayer, self).__init__()
        self.hidden_act = hidden_act
        self.conv = conv
        self.cat_dim = cat_dim

        if not self.conv:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(input_dim+hidden_dim, hidden_dim)
            self.fc_out = nn.Linear(input_dim+2*hidden_dim, output_dim)
        else:
            self.conv_1 = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
            self.conv_2 = nn.Conv1d(input_dim+hidden_dim, hidden_dim, 3, padding=1)
            self.conv_3 = nn.Conv1d(input_dim+2*hidden_dim, output_dim, 3, padding=1)

        if self.hidden_act == 'relu':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        self.drop1 = nn.Dropout(p=drop)
        self.drop2 = nn.Dropout(p=drop)

    def forward(self, x):
        # dense layer 1
        x1_in = x
        if not self.conv:
            x1 = self.fc1(x1_in)
        else:
            x1 = self.conv_1(x1_in.permute(0,2,1)).permute(0,2,1)
        x1 = self.act1(x1)
        x1 = self.drop1(x1)

        # dense layer 2
        x2_in = torch.cat((x, x1), dim=self.cat_dim)
        if not self.conv:
            x2 = self.fc2(x2_in)
        else:
            x2 = self.conv_2(x2_in.permute(0,2,1)).permute(0,2,1)
        x2 = self.act2(x2)
        x2 = self.drop2(x2)

        # dense layer 3
        x3_in = torch.cat((x, x1, x2), dim=self.cat_dim)
        if not self.conv:
            y = self.fc_out(x3_in)
        else:
            y = self.conv_3(x3_in.permute(0,2,1)).permute(0,2,1)
        return y


def _get_activation_fn(activation):
    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class PositionalEncoding(nn.Module):
    # based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class GatedTransformerEncoderLayer(nn.Module):
    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    def __init__(self, d_model, nhead, dim_feedforward=2048, gate=True, dropout=0.1, activation="relu"):
        super(GatedTransformerEncoderLayer, self).__init__()
        self.gate = gate

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, 
                                            out_dim_mult=1.0)
        self.linear_mha = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if self.gate:
            self.gate1 = nn.Linear(d_model*2, d_model)
            self.gate2 = nn.Linear(d_model*2, d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(GatedTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_ = self.norm1(src)
        src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear_mha(src2)
        if self.gate:
            g2 = self.gate1(torch.cat((src_, src2), dim=2))
            g2_shape = g2.shape
            g2 = g2.reshape((g2_shape[0], g2_shape[1], 128, 8))
            g2 = torch.softmax(g2, dim=-1)
            g2 = g2.reshape((g2_shape[0], g2_shape[1], g2_shape[2]))
            src2 = src2 * g2
        src = src + self.dropout1(src2)

        src_ = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_))))
        if self.gate:
            g2 = self.gate2(torch.cat((src_, src2), dim=2))
            g2_shape = g2.shape
            g2 = g2.reshape((g2_shape[0], g2_shape[1], 128, 8))
            g2 = torch.softmax(g2, dim=-1)
            g2 = g2.reshape((g2_shape[0], g2_shape[1], g2_shape[2]))
            src2 = src2 * g2
        src = src + self.dropout2(src2)

        return src

class MemoryTransformerEncoderLayer(nn.Module):
    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    def __init__(self, d_model, nhead, d_memory, dim_feedforward=2048, dropout=0.1, activation="relu",
                 mha_enabled=True, memory_position="", memory_gate=True, hidden_sentence_dropout=0.0):
        super(MemoryTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.mha_enabled = mha_enabled
        self.memory_position = memory_position
        self.memory_gate = memory_gate

        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.hsent_drop1 = nn.Dropout(hidden_sentence_dropout)
        self.linear1 = nn.Linear(d_model+(d_memory if 'ffn input' in memory_position else 0), dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.hsent_drop2 = nn.Dropout(hidden_sentence_dropout)
        self.linear2 = nn.Linear(dim_feedforward+(d_memory if 'ffn hidden' in memory_position else 0), d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        if self.mha_enabled:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, out_dim_mult=1.0)
            self.hsent_drop_mha = nn.Dropout(hidden_sentence_dropout)
            self.linear_mha = nn.Linear(d_model+(d_memory if 'mha hidden' in memory_position else 0), d_model)

            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)

        if self.memory_gate:
            self.mem_gate = nn.Linear(d_model+d_memory, d_memory)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MemoryTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        mem = src[:, :, self.d_model:]
        src = src[:, :, :self.d_model]

        if self.memory_gate:
            mem = mem*torch.tanh(torch.abs(self.mem_gate(torch.cat([src, mem], dim=2))))

        if self.mha_enabled:
            # src_ = self.norm1(src)
            src_ = src
            src2 = self.self_attn(src_, src_, src_, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            if 'mha hidden' in self.memory_position:
                src2 = torch.cat([self.hsent_drop_mha(src2), mem], dim=2)
            src2 = self.linear_mha(src2)
            src = src + self.dropout1(src2)
            src = self.norm1(src)

        # src_ = self.norm2(src)
        src_ = src
        if 'ffn input' in self.memory_position:
            src_ = torch.cat([self.hsent_drop1(src_), mem], dim=2)
        src2 = self.dropout(self.activation(self.linear1(src_)))
        if 'ffn hidden' in self.memory_position:
            src2 = torch.cat([self.hsent_drop2(src2), mem], dim=2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = torch.cat([src, mem], dim=2)
        return src

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        self.config = config

        self.s2v_dim = config['s2v_dim']
        self.word_edim = config['word_edim']

        # input
        self.mem_in_dr = nn.Dropout(config['sentence_encoder']['input_drop'])

        # gated transformer
        gtr_num_head = config['sentence_encoder']['transformer']['num_heads']
        gtr_num_layers = config['sentence_encoder']['transformer']['num_layers']
        gtr_dim_feedforward = config['sentence_encoder']['transformer']['ffn_dim']
        gtr_gate = config['sentence_encoder']['transformer']['gate']
        gtr_drop = config['sentence_encoder']['transformer']['dropout']

        if config['sentence_encoder']['transformer']['num_layers'] > 0:
            self.mem_norm = nn.LayerNorm(self.word_edim)

            mem_encoder_layer = GatedTransformerEncoderLayer(d_model=self.word_edim, nhead=gtr_num_head,
                                                        dim_feedforward=gtr_dim_feedforward,
                                                        gate=gtr_gate,
                                                        dropout=gtr_drop, activation="gelu")
            self.mem_gtr = nn.TransformerEncoder(mem_encoder_layer, num_layers=gtr_num_layers)

        # mha pool
        pool_mha_nhead = config['sentence_encoder']['pooling']['mha']['num_heads']
        pool_mha_drop = config['sentence_encoder']['pooling']['mha']['attention_dropout']
        pool_mha_out_mult = self.s2v_dim/self.word_edim

        self.mem_mha_pool = MultiheadAttention(self.word_edim, pool_mha_nhead, dropout=pool_mha_drop,
                                               out_dim_mult=pool_mha_out_mult)

        # mlm transformer
        self.mlm_in_dr = nn.Dropout(config['sentence_mlm']['input_drop'])
        mtr_num_head = config['sentence_mlm']['transformer']['num_heads']
        mtr_num_layers = config['sentence_mlm']['transformer']['num_layers']
        mtr_dim_feedforward = config['sentence_mlm']['transformer']['ffn_dim']
        mtr_drop = config['sentence_mlm']['transformer']['dropout']
        mtr_mha_en = config['sentence_mlm']['transformer']['mha']
        mtr_mem_pos = config['sentence_mlm']['transformer']['memory_position']
        mtr_mem_gate = config['sentence_mlm']['transformer']['memory_gate']
        mtr_hsent_drop = config['sentence_mlm']['transformer']['hidden_sentence_drop']

        mlm_encoder_layer = MemoryTransformerEncoderLayer(d_model=self.word_edim, nhead=mtr_num_head,
                                                          d_memory=self.s2v_dim,
                                                          dim_feedforward=mtr_dim_feedforward,
                                                          dropout=mtr_drop, activation="gelu",
                                                          mha_enabled=mtr_mha_en,
                                                          memory_position=mtr_mem_pos,
                                                          memory_gate=mtr_mem_gate,
                                                          hidden_sentence_dropout=mtr_hsent_drop)
        self.mlm_mtr = nn.TransformerEncoder(mlm_encoder_layer, num_layers=mtr_num_layers)

    def _emb_sent(self, sent, sent_mask=None):
        sent = self.mem_in_dr(sent)
        sent = sent.permute((1, 0, 2))

        if self.config['sentence_encoder']['transformer']['num_layers'] > 0:
            sent = self.mem_gtr(sent, src_key_padding_mask=sent_mask)
            sent = self.mem_norm(sent)
        sent, _ = self.mem_mha_pool(sent, sent, sent, key_padding_mask=sent_mask)

        sent = sent.permute((1, 0, 2))

        sent_mask_exp = torch.cat([sent_mask.unsqueeze(2)]*sent.shape[2], dim=2).type(torch.cuda.FloatTensor)
        if self.config['sentence_encoder']['pooling']['pooling_function'] == 'max':
            s2v, _ = torch.max(sent + sent_mask_exp*-1e3, axis=1)
        elif self.config['sentence_encoder']['pooling']['pooling_function'] == 'mean':
            s2v = torch.sum(sent*(1-sent_mask_exp), axis=1) / \
                  (torch.sum((1-sent_mask_exp), axis=1) + 1e-6)

        return s2v

    def _sent_mlm(self, sent, mem_s2v, sent_mask=None):
        sent = self.mlm_in_dr(sent)
        sent = sent.permute((1, 0, 2))

        mem_s2v = torch.cat([mem_s2v.unsqueeze(0)]*sent.shape[0], dim=0)
        sent = torch.cat([sent, mem_s2v], dim=2)
        sent = self.mlm_mtr(sent, src_key_padding_mask=sent_mask)
        sent = sent[:, :, 0:self.word_edim]

        sent = sent.permute((1, 0, 2))
        return sent

    def forward(self, sent, mem_sent, sent_mask=None, mem_sent_mask=None):
        mem_s2v = self._emb_sent(mem_sent, sent_mask=mem_sent_mask)

        sent = self._sent_mlm(sent, mem_s2v, sent_mask=sent_mask)

        sent = sent[:, :mem_sent.shape[1]]

        return {
            'mem_s2v': mem_s2v,
            'sent': sent
        }
