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

class UpperLevelDecoder(torch.nn.Module):
    def __init__(self, Z_in, H):
        super(UpperLevelDecoder, self).__init__()
        self.LSTM = torch.nn.LSTM(H, H, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(Z_in, H)

    def forward(self, z):
        output = self.linear(z)
        output, hidden = self.LSTM(output, None)
        return output, hidden
