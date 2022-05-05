import torch
from torch import nn


class lstm(nn.Module):
    def __init__(self, hidden_size=32, num_layers=1, num_classes=1000):
        super(lstm, self).__init__()

        self.rnn1 = nn.LSTM(          # if use nn.RNN(), it hardly learns
            input_size=1,
            hidden_size=hidden_size, # rnn hidden unit
            num_layers=num_layers,   # number of rnn layer
            batch_first=True,        # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        #self.rnn2 = nn.LSTM(          # if use nn.RNN(), it hardly learns
        #    input_size=hidden_size,
        #    hidden_size=hidden_size, # rnn hidden unit
        #    num_layers=num_layers,   # number of rnn layer
        #    batch_first=True,        # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        #)
        #self.relu1 = nn.ReLU()
        #self.relu2 = nn.ReLU()

        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x, features_only=False):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn1(x, None)   # None represents zero initial hidden state
        #r_out = self.relu1(r_out)
        #r_out, (h_n, h_c) = self.rnn2(r_out, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        x = r_out[:, -1, :]
        if features_only:
            return x
        out = self.out(x)
        return out
