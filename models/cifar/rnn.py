from torch import nn
from torch.nn.modules.rnn import LSTMCell
import torch

class lstm(nn.Module):
    def __init__(self,h_size=64,img_width=32*3, num_classes = 10):
        super(lstm, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns // LSTM
            input_size=img_width,
            hidden_size=h_size,         # rnn hidden unit
            num_layers=3,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(h_size, num_classes)

    def forward(self, x, features_only=False):
        # flatten
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(-1, 32, 32 * 3)
        print(x.shape)
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state # for lstm
        # r_out, h_n = self.rnn(x, None) # for rnn
        # choose r_out at the last time step
        if not features_only:
            output = self.out(r_out[:, -1, :])
            return output

        return r_out[:, -1, :]

class lstm_cell_level(nn.Module):
    def __init__(self, input_size=32*3, hidden_size=512, num_classes=10):
        super(lstm_cell_level, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)
        self.prune_timestep = []
        self.valid_timestep = torch.arange(32)

    def forward(self, inputs, features_only=False):
        # flatten
        inputs = inputs.permute(0, 2, 3, 1)
        inputs = inputs.contiguous().view(-1, 32, 32 * 3)
        inputs = inputs.permute(1,0,2)


        hx = torch.randn(inputs.shape[1], self.hidden_size) # (batch, hidden_size)
        cx = torch.randn(inputs.shape[1], self.hidden_size)
        if inputs.is_cuda:
            hx = hx.cuda()
            cx = cx.cuda()
            inputs = inputs.index_select(0, self.valid_timestep.cuda())
        else:
            inputs = inputs.index_select(0, self.valid_timestep)
        for i in range(inputs.shape[0]):
            hx, cx = self.lstm_cell(inputs[i], (hx, cx))
            # hx1, cx1 = self.lstm_cell(inputs[i], (hx, cx))
            
            # hx,cx = hx1,cx1

        if features_only:
            return hx
        
        # get the final step as output
        return self.out(hx)

    def setPruneTimeSteps(self, candidates):
        # for x in candidates:
        #     self.prune_timestep.add(x)
        # self.prune_timestep = candidates
        tmp = []
        for i in range(32):
            if not i in candidates:
                tmp.append(i)
        self.valid_timestep = torch.tensor(tmp)

