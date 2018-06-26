import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn

class UpperLevelDecoder(torch.nn.Module):
    def __init__(self, Z_in, H):
        super(UpperLevelDecoder, self).__init__()
        self.LSTM = torch.nn.LSTM(H, H, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(Z_in, H)

    def forward(self, z):
        output = self.linear(z)
        output, hidden = self.LSTM(output, None)
        return output, hidden
