import torch
from torch import nn

from torch.nn.modules.rnn import LSTMCell

class lstmCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=10):
        super(lstmCell, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        inputs = inputs.permute(1,0,2)

        hx = torch.randn(inputs.shape[1], self.hidden_size) # (batch, hidden_size)
        cx = torch.randn(inputs.shape[1], self.hidden_size)
        for i in range(inputs.shape[0]):
            hx, cx = self.lstm_cell(inputs[i], (hx, cx))
        
        # get the final step as output
        return self.out(hx)
        