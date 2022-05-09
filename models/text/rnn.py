import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.prune_timestep = []

    def forward(self, input, features_only=False):
        input = input.permute(1,0,2)
        hidden = self.initHidden(input.shape[1])
        
        for i in range(input.shape[0]):
            if not i in self.prune_timestep:
                combined = torch.cat((input[i], hidden), 1)
                hidden = self.i2h(combined)

        if features_only:
            return hidden
        
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output

    def initHidden(self,batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size)).cuda()

    def setPruneTimeSteps(self, candidates):
        self.prune_timestep = candidates
