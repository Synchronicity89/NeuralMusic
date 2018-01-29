import torch,ipdb
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

i_size = 127 # 127 midi notes + 1 for rest.
num_layers=7
num_hyperparams=4
batch = 1
hidden_size = 20
rnn = nn.LSTM(input_size=num_hyperparams, hidden_size=hidden_size, num_layers=num_layers)

input = Variable(torch.randn(1, batch, num_hyperparams)) # (seq_len, batch, input_size)
h0 = Variable(torch.randn(num_layers, batch, hidden_size)) # (num_layers, batch, hidden_size)
c0 = Variable(torch.randn(num_layers, batch, hidden_size))
output, hn = rnn(input, (h0, c0))
affine1 = nn.Linear(hidden_size, num_hyperparams)

ipdb.set_trace()
print output.size()
print h0.size()
