import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn

'''
class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.LSTM = torch.nn.LSTM(D_in, H, 1, batch_first=True)
        self.linear1 = torch.nn.Linear(H, D_out)

    def forward(self, prev, z, hidden):
        inp = torch.cat((prev, z), dim=2)
        output, hn = self.LSTM(inp, hidden)
        return F.relu(self.linear1(output)), hn
'''

class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.LSTM = torch.nn.LSTM(D_in, H, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(H, D_out)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, z):
        inp = torch.cat((input, z), dim=2)
        output, hidden = self.LSTM(inp, hidden)
        output = self.linear(output)
        return output, hidden
