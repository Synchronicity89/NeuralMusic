import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import mido
import time
import sys
import os
sys.path.insert(0, '../postprocess/')
sys.path.insert(0, '../preprocess/')
from encoder import Encoder
from decoder import Decoder
import recreate
import processmidi as pm
import progressbar
np.set_printoptions(threshold=np.nan)

class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, hidden_size, latent_dim, use_cuda):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(hidden_size, latent_dim)
        self._enc_log_sigma = torch.nn.Linear(hidden_size, latent_dim)
        self.use_cuda = use_cuda

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        var = Variable(std_z, requires_grad=False)
        var = var.cuda() if self.use_cuda else var

        return mu + sigma * var  # Reparameterization trick

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, state, criterion, batch_size, input_dim, latent_dim, z_size, song_length):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        #z = z.view(batch_size, latent_dim, z_size) # Change the first 1 if more that one batch
        output = Variable(torch.zeros(batch_size, 1, input_dim))
        output = output.cuda() if self.use_cuda else output
        song = None
        temp_loss = None
        for i in range(0, song_length):
            temp_z = z.squeeze(0)[i].view(1, 1, latent_dim)
            if i == 0:
                hidden = (temp_z, temp_z)
            output, hidden = self.decoder(output, temp_z, hidden)
            state_i = state.squeeze(0)[i].view(1, 1, input_dim)
            if song is None:
                song = output
                temp_loss = criterion(output, state_i)
            else:
                song = torch.cat((song, output), dim=0)
                temp_loss += criterion(output, state_i)
            output = state_i  # test teacher forcing
            output = output.view(batch_size, 1, input_dim)
        return temp_loss/song_length


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

def loss_function(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def loadData(directory, song_length):
    songdata = []
    i = 0
    with progressbar.ProgressBar(max_value=len(os.listdir(directory))) as bar:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                currentSong = np.loadtxt(directory+'/'+filename)
                if len(currentSong) >= song_length:
                    currentSong = currentSong[0:song_length] # Making song shorter
                    songdata.append(currentSong)
                i = i + 1
                bar.update(i)
            else:
                print("Error while loading data. Could not find .txt files")
                exit()
        return songdata


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    t0 = time.time()
    '''Input data config'''
    song_length = 256
    directory = 'onesong'
    songdata = loadData(directory, song_length)

    input_dim = 130 # 128 = 8 bars, 130 is midi notes, hold and pause
    batch_size = 1

    dataloader = torch.utils.data.DataLoader(songdata, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    hidden_size = 100
    z_size = 1
    latent_dim = 30

    print('Number of samples: ', len(songdata))
    encoder = Encoder(input_dim, hidden_size, hidden_size)
    decoder = Decoder(input_dim + latent_dim, latent_dim, input_dim)
    vae = VAE(encoder, decoder, hidden_size, latent_dim, use_cuda)
    criterion = nn.MSELoss()
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
        vae.cuda()
        criterion.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=0.0003)
    #optimizer = optim.Adam(vae.parameters(), lr=0.01)
    l = None
    for epoch in range(10):
        with progressbar.ProgressBar(max_value=len(dataloader)) as bar:
            for i, data in enumerate(dataloader, 0):
                #song_length = len(data[0])
                if use_cuda:
                    data = data.cuda()
                inputs = Variable(data).float()
                inputs = inputs.cuda() if use_cuda else inputs
                #inputs = Variable(inputs.resize_(batch_size, input_dim))
                optimizer.zero_grad()
                loss = vae(inputs, criterion, batch_size, input_dim, latent_dim, z_size, song_length)
                ll = latent_loss(vae.z_mean, vae.z_sigma)
                loss = loss + ll
                loss.backward()
                optimizer.step()
                l = loss.data[0]
                bar.update(i)
        print(epoch, l)
    t1 = time.time()

    #exit()

    sample = Variable(torch.randn(batch_size, 1, latent_dim))
    sample = sample.cuda() if use_cuda else sample
    #s = vae.encoder(Variable(torch.randn(1, 128, 130)))
    #sample = vae._sample_latent(s)
    #sample = sample.view(1, 1, 128)
    output = Variable(torch.zeros(batch_size, 1, 130))
    output = output.cuda() if use_cuda else output

    song = None
    hidden = (sample, sample)
    for i in range(0, song_length):
        output, hidden = vae.decoder(output, sample, hidden)
        if song is None:
            song = output
        else:
            print(output)
            print(song)
            song = torch.cat((song, output), dim=1)
        output = output.view(batch_size, 1, 130)
    #print(song)
    #output = song.view(1, 128, 130)
    #output = song
    #song = None
    song = song.squeeze(0)
    for data in song:
        #value, index = torch.max(data, 0) This is highest probability
        #index = index.data[0]
        b = np.zeros(shape=(1, 130))
        values, indices = data.max(0)
        print(indices)

        '''Get highest probability'''
        '''
        if (index == 129):
            topKValue, topKIndex = torch.topk(data, 2)
            b.itemset((0, topKIndex.data[0]), 1)
            b.itemset((0, topKIndex.data[1]), 1)
        elif (index == 128):
            topKValue, topKIndex = torch.topk(data, 2)
            b.itemset((0, topKIndex.data[1]), 1)
        else:
            b.itemset((0, index), 1)
            print(index)
            #exit()
            '''

        ''' Multinnomial Sampling'''

        '''
        value = data.multinomial(1)
        value = value.data[0][0]
        if (value == 129):
            topKValue, topKIndex = torch.topk(data, 2)
            topKIndex = topKIndex.data[0]
            if topKIndex[0] == 129:
                valueToBeSet = topKIndex[1]
            else:
                valueToBeSet = topKIndex[1]
            b.itemset((0, value), 1)
            b.itemset((0, valueToBeSet), 1)
        else:
            b.itemset((0, value), 1)
        if (song is None):
            song = b
        else:
            song = np.concatenate([song, b])
    rec = recreate.RecreateMIDI()
    #print(song)
    track = rec.recreateMIDI(song, 30)
    rec.createMIDITest(track, 'VAERecreated')
    print('Runtime: ' + str(t1-t0) + " seconds")
    '''
