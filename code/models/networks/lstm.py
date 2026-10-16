import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=bidirectional)
        assert embd_method in ['maxpool', 'attention', 'last', 'dense']
        self.embd_method = embd_method
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh())
            self.softmax = nn.Softmax(dim=-1)
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential()
            self.bidirectional = bidirectional
            if bidirectional:
                self.dense_layer.add_module('linear', nn.Linear(2 * self.hidden_size, self.hidden_size))
            else:
                self.dense_layer.add_module('linear', nn.Linear(self.hidden_size, self.hidden_size))
            self.dense_layer.add_module('activate', nn.Tanh())
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        hidden_reps = self.attention_layer(r_out)
        atten_weight = hidden_reps @ self.attention_vector_weight
        atten_weight = self.softmax(atten_weight)
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        in_feat = r_out.transpose(1, 2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)

    def embd_last(self, r_out, h_n):
        return h_n.squeeze(0)

    def embd_dense(self, r_out, h_n):
        if self.bidirectional:
            h_n = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
            output = self.dense_layer(h_n)
        else:
            h_n = h_n[-1, :, :]
            output = self.dense_layer(h_n)
        return output

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_' + self.embd_method)(r_out, h_n)
        return embd
