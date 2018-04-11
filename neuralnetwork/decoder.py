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
        self.linear1 = torch.nn.Linear(H,  H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, prev, z, hidden):
        #print("Decoder")
        #print(z)
        #exit()
        inp = torch.cat((prev, z), dim=2)
        #print(inp)
        #print(hidden)
        output, hn = self.LSTM(inp, hidden)
        #print(hn)
        #exit()
        #print(hn)
        #print(output)
        #exit()
        x = F.relu(self.linear1(output.squeeze(0)))
        return F.relu(self.linear2(x)), hn
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
