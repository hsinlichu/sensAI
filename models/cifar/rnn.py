from torch import nn

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
