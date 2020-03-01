import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import numpy as np
import math

def Pendulum(t, b, kappa, A0=1, m=1, delta0=0):
    
    omega = torch.sqrt(kappa/m)*torch.sqrt(1-b**2/(4*m*kappa))
    Pendulum_position = A0*torch.exp(-b*t/(2*m))*torch.cos(omega*t + delta0)
    return Pendulum_position


class Net(nn.Module):
    def __init__(self, input_size, latent_size, encoder_num_units,
                       decoder_num_units, output_size, question_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units
        self.output_size = output_size
        self.question_size = question_size

        self.encode_hidden_1 = nn.Linear(self.input_size, self.encoder_num_units[0])
        self.encode_hidden_2 = nn.Linear(self.encoder_num_units[0], self.encoder_num_units[1])
        self.encode_latent = nn.Linear(self.encoder_num_units[1], self.latent_size)
        self.decode_latent = nn.Linear(self.latent_size + self.question_size, self.decoder_num_units[0])
        self.decode_hidden_1 = nn.Linear(self.decoder_num_units[0], self.decoder_num_units[1])
        self.decode_hidden_2 = nn.Linear(self.decoder_num_units[1], self.output_size)

        self.elu = nn.ELU()

    def forward(self, x, q):
        # encoder part (linear layers with ELU activation functions)
        x = self.elu(self.encode_hidden_1(x))
        x = self.elu(self.encode_hidden_2(x))
        # no activation function for latent layer to keep it unbounded
        x1 = self.encode_latent(x)
        # concat latent layer with input question
        x = torch.cat((x1, q), dim=1)
        # decoder part
        x = self.elu(self.decode_latent(x))
        x = self.elu(self.decode_hidden_1(x))
        # output layer has no activation function for now because it is unbounded (for now)
        x = self.decode_hidden_2(x)
        return x

class Dataset(data.Dataset):
  def __init__(self, num_samples, input_size, output_size, question_size):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.question_size = question_size

  def __len__(self):
        return self.num_samples

  def __getitem__(self, index):
        # Randomly sample pendulum for now...
        tvec = torch.linspace(0, 5, 50)
        b = (1.2 - 0.4) * torch.rand(1) + 0.4
        k = (12 - 4) * torch.rand(1) + 4
        X = Pendulum(tvec, b, k)
        q = torch.rand(self.question_size)*10.
        y = Pendulum(q, b, k)

        return X, y, q


def train_pmns(nepoch, batch_size=512, learning_rate=0.001,
               input_size=50, latent_size=3, encoder_num_units=[500, 100],
               decoder_num_units=[100, 100], output_size=1, question_size=1):

    net = Net(input_size, latent_size, encoder_num_units, decoder_num_units, output_size, question_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # load data
    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 4}
    training_set = Dataset(95000, input_size, output_size, question_size)
    validation_set = Dataset(5000, input_size, output_size, question_size)
    trainloader = data.DataLoader(training_set, **params)
    valloader = data.DataLoader(validation_set, **params)

    for epoch in range(nepoch):  # loop over the dataset multiple times
        running_tloss = 0.0
        running_vloss = 0.0
        for i, batch in enumerate(trainloader):
            net.train()
     
            # get the inputs
            inputs, labels, questions = batch
            labels = labels.view(-1,output_size)

            # zero the parameter gradients
            optimizer.zero_grad()
         
            # forward + backward + optimize
            outputs = net(inputs, questions)
            loss = criterion(outputs, labels.type(torch.float))
            loss.backward()
            optimizer.step()
         
            # print statistics
            running_tloss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] train loss: %.5f' %
                      (epoch + 1, i + 1, running_tloss / 50))
                running_tloss = 0.0

        for i, batch in enumerate(valloader):
            net.eval()
     
            # get the inputs
            inputs, labels, questions = batch
            labels = labels.view(-1,output_size)

            # forward + backward + optimize
            outputs = net(inputs, questions)
            vloss = criterion(outputs, labels.type(torch.float))
         
            # print statistics
            running_vloss += vloss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('[%d, %5d] val loss: %.5f' %
                      (epoch + 1, i + 1, running_vloss / 10))
                running_vloss = 0.0


    print('Finished Training')

    return net


