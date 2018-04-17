import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn

'''
class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.LSTM = torch.nn.LSTM(D_in, H, 1, batch_first=True)
        self.linear1 = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        output, hn = self.LSTM(x, None)
        x = F.relu(self.linear1(output))
        return F.relu(self.linear2(x))
'''
class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.LSTM = torch.nn.LSTM(H, H, 1, batch_first=True)

    def forward(self, input):
        output = self.linear1(input)
        output, hidden = self.LSTM(output, None)
        return output, hidden
