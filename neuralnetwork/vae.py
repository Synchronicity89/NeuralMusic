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
from upperleveldecoder import UpperLevelDecoder
import recreate
import processmidi as pm
import progressbar
from compare import Compare
import math
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
    def __init__(self, encoder, decoder, upperleveldecoder, hidden_size, latent_dim, use_cuda):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.upperleveldecoder = upperleveldecoder
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
        o_enc, h_enc = self.encoder(state)
        z = self._sample_latent(o_enc)
        #z = z.view(batch_size, latent_dim, z_size) # Change the first 1 if more that one batch
        output = Variable(torch.zeros(batch_size, 1, input_dim))
        output = output.cuda() if self.use_cuda else output
        song = None
        temp_loss = None
        #temp_hid = h_enc.squeeze(0)[-1].view(1, 1, input_dim)
            #hidden = (temp_z, temp_z)
        z_out, hidden = self.upperleveldecoder(z)
        for i in range(0, song_length):
            zi = z.squeeze(0)[i].view(1, 1, latent_dim)
            output, hidden = self.decoder(output, hidden, zi)
            #output, hidden = self.decoder(output, hidden)
            state_i = state.squeeze(0)[i].view(1, 1, input_dim)
            if song is None:
                song = output
                temp_loss = criterion(output, state_i)
            else:
                song = torch.cat((song, output), dim=0)
                temp_loss += criterion(output, state_i)
            output = state_i  # test teacher forcing
            output = output.view(batch_size, 1, input_dim)
        return temp_loss/song_length, z


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

def loadData(directory):
    songdata = []
    i = 0
    with progressbar.ProgressBar(max_value=len(os.listdir(directory))) as bar:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                currentSong = np.loadtxt(directory+'/'+filename)
                newSong = []
                if (len(currentSong > 0)):
                    for timestep in currentSong:
                        if (len(np.nonzero(timestep)) > 0):
                            if (np.nonzero(timestep)[0][0] != 128):
                                newSong.append(timestep)
                    #currentSong = currentSong[0:song_length] # Making song shorter
                    newSong = np.array(newSong)
                    songdata.append(newSong)
                    i = i + 1
                    bar.update(i)
        return songdata



if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    #torch.cuda.set_device(1)
    t0 = time.time()
    '''Input data config'''
    directory = 'bluessolodata'
    songdata = loadData(directory)
    input_dim = 130 # 128 = 8 bars, 130 is midi notes, hold and pause
    batch_size = 1

    dataloader = torch.utils.data.DataLoader(songdata, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    hidden_size = 64 #usually 100
    z_size = 1
    latent_dim = 30

    print('Number of samples: ', len(songdata))
    encoder = Encoder(input_dim, hidden_size, hidden_size)
    upperleveldecoder = UpperLevelDecoder(latent_dim, hidden_size)
    decoder = Decoder(input_dim + latent_dim, hidden_size, input_dim) #It was (input_dim + latent_dim, latent_dim, input_dim)
    vae = VAE(encoder, decoder, upperleveldecoder, hidden_size, latent_dim, use_cuda)
    criterion = nn.MSELoss()
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
        vae.cuda()
        criterion.cuda()

    #optimizer = optim.Adam(vae.parameters(), lr=0.0005)
    #optimizer = optim.Adam(vae.parameters(), lr=0.01)
    learningRate = 0.0095
    optimizer = optim.Adam(vae.parameters(), lr=learningRate)
    l = None
    songToProcess = None
    lastZ = None
    for epoch in range(200):
        with progressbar.ProgressBar(max_value=len(dataloader)) as bar:
            for i, data in enumerate(dataloader, 0):
                song_length = len(data[0])
                if use_cuda:
                    data = data.cuda()
                inputs = Variable(data).float()
                inputs = inputs.cuda() if use_cuda else inputs
                #inputs = Variable(inputs.resize_(batch_size, input_dim))
                optimizer.zero_grad()
                loss, z = vae(inputs, criterion, batch_size, input_dim, latent_dim, z_size, song_length)
                ll = latent_loss(vae.z_mean, vae.z_sigma)
                loss = loss  + ll
                loss.backward()
                optimizer.step()
                l = loss.data[0]
                bar.update(i)
                songToProcess = data
        print(epoch, l)
        learningRate = learningRate * math.exp(-0.01)
        print('Updating learning rate: ' + str(learningRate))
    t1 = time.time()

    '''Generate from encoder/decoder directly'''
    '''
    song = None
    o_enc, h_enc = vae.encoder(Variable(songToProcess).float())
    output = Variable(torch.zeros(batch_size, 1, 130))
    hidden = h_enc
    for i in range(0, song_length):
        output, hidden = vae.decoder(output, hidden)
        if song is None:
            song = output
        else:
            song = torch.cat((song, output), dim=1)
        output = output.view(batch_size, 1, 130)'''

    samples = []
    song_length = 256
    s = Variable(torch.randn(batch_size, song_length, latent_dim))
    s = s.cuda() if use_cuda else s
    s1 = Variable(torch.zeros(batch_size, song_length, latent_dim))
    s1 = s1.cuda() if use_cuda else s1
    s2 = Variable(torch.ones(batch_size, song_length, latent_dim))
    s2 = s2.cuda() if use_cuda else s2
    samples.append(s)
    samples.append(s1)
    samples.append(s2)
    #s = vae.encoder(Variable(torch.randn(1, 128, 130)))
    #sample = vae._sample_latent(s)
    #sample = sample.view(1, 1, 128)


    for number, sample in enumerate(samples):
        song = None
        output = Variable(torch.zeros(batch_size, 1, 130))
        output = output.cuda() if use_cuda else output
        z_out, hidden = vae.upperleveldecoder(sample)
        for i in range(0, song_length):
            samplei = sample.squeeze(0)[i].view(1, 1, latent_dim)
            output, hidden = vae.decoder(output, hidden, samplei)
            if song is None:
                song = output
            else:
                song = torch.cat((song, output), dim=1)
            output = output.view(batch_size, 1, 130)
        #print(song)
        #output = song.view(1, 128, 130)

        output = song.squeeze(0)
        song = None
        prevNote = None
        limit = 16
        for data in output:
            #value, index = torch.max(data, 0) This is highest probability
            #index = index.data[0]
            '''Highest probability'''
            b = np.zeros(shape=(1, 130))
            values, indices = data.max(0)
            if (indices.data[0] == 129):
                topKValue, topKIndex = torch.topk(data, 2)
                if (prevNote is None or prevNote != topKIndex.data[1] or limit == 0):
                    b.itemset((0, topKIndex.data[1]), 1)
                    prevNote = topKIndex.data[1]
                    limit = 16
                else:
                    b.itemset((0, topKIndex.data[0]), 1)
                    b.itemset((0, topKIndex.data[1]), 1)
                    limit = limit - 1
                    prevNote = topKIndex.data[1]
            else:
                if (len(indices.data) == 1):
                    b.itemset((0, indices.data[0]), 1)
                    prevNote = indices.data[0]
                    limit = 16

            ''' Multinnomial Sampling'''
            '''value = data.multinomial(1)
            if (value.data[0] == 129):
                topKValue, topKIndex = torch.topk(data, 2)
                if topKIndex.data[0] == 129:
                    valueToBeSet = topKIndex.data[1]
                else:
                    valueToBeSet = topKIndex.data[1]
                b.itemset((0, value.data[0]), 1)
                b.itemset((0, valueToBeSet), 1)
            else:
                b.itemset((0, value.data[0]), 1)'''
            if (song is None):
                song = b
            else:
                song = np.concatenate([song, b])
        print(song)
        rec = recreate.RecreateMIDI()
        #print(song)
        track = rec.recreateMIDI(song, 30)
        rec.createMIDITest(track, 'VAERecreated'+str(number))
    print('Runtime: ' + str(t1-t0) + " seconds")

    '''Compare solo with test data'''
    comparator = Compare()
    comparator.compareData(songdata, track)
