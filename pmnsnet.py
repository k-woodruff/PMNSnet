import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import Oscillations_PMNS

import numpy as np
import math

def Pendulum(t, b, kappa, A0=1, m=1, delta0=0):
    
    omega = torch.sqrt(kappa/m)*torch.sqrt(1-b**2/(4*m*kappa))
    Pendulum_position = A0*torch.exp(-b*t/(2*m))*torch.cos(omega*t + delta0)
    return Pendulum_position


def generate_data(low_L=0, high_L=20e3, data_size=90):
    ## making the torch data set for 0->20000 L/E
    ## with random points

    Oscillation = Oscillations_PMNS.Oscillations()
    
    prob_e_to_mu = []
    prob_e_to_tau = []
    E = 1
    L = np.random.uniform(low=low_L, high=high_L, size=data_size)
    L = torch.from_numpy(L)
    
    Oscillation.setE(E * Oscillations_PMNS.units.GeV)
    
    for x in range(0,len(L)):
        Oscillation.setL(L[x] * Oscillations_PMNS.units.km)
        prob_e_to_mu.append(Oscillation.p(Oscillations_PMNS.nu_e, Oscillations_PMNS.nu_mu))
        prob_e_to_tau.append(Oscillation.p(Oscillations_PMNS.nu_e, Oscillations_PMNS.nu_tau))
        
    prob_e_to_mu   = torch.FloatTensor(prob_e_to_mu)
    prob_e_to_tau  = torch.FloatTensor(prob_e_to_tau)

    return L, prob_e_to_mu, prob_e_to_tau


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
        self.encode_latent = nn.Linear(self.encoder_num_units[1], 2*self.latent_size)
        self.decode_latent = nn.Linear(self.latent_size + self.question_size, self.decoder_num_units[0])
        self.decode_hidden_1 = nn.Linear(self.decoder_num_units[0], self.decoder_num_units[1])
        self.decode_hidden_2 = nn.Linear(self.decoder_num_units[1], self.output_size)

        self.elu = nn.ELU()

    def forward(self, x, q, ep):
        # encoder part (linear layers with ELU activation functions)
        x = self.elu(self.encode_hidden_1(x))
        x = self.elu(self.encode_hidden_2(x))
        # no activation function for latent layer to keep it unbounded
        x1 = self.encode_latent(x)
        sigmas = x1[:,:self.latent_size]
        mus = x1[:,self.latent_size:]
        x1_sample = mus + sigmas*ep
        # concat latent layer with input question
        x = torch.cat((x1_sample, q), dim=1)
        # decoder part
        x = self.elu(self.decode_latent(x))
        x = self.elu(self.decode_hidden_1(x))
        # output layer has no activation function for now because it is unbounded (for now)
        x = self.decode_hidden_2(x)
        return x, sigmas, mus

class DatasetPendulum(data.Dataset):
  def __init__(self, num_samples, input_size, output_size, question_size, latent_size):
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.question_size = question_size
        self.latent_size = latent_size

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
        eps = torch.randn(self.latent_size)

        return X, y, q, eps, b.item(), k.item()

class DatasetPMNS(data.Dataset):
  def __init__(self, baselines, mu_probs, tau_probs, latent_size):
        self.baselines = baselines
        self.mu_probs = mu_probs
        self.tau_probs = tau_probs
        self.latent_size = latent_size

  def __len__(self):
        return len(self.baselines)

  def __getitem__(self, index):
        L = self.baselines[index].view(1)
        q = torch.randint(0,2,(1,))
        if q.item() == 0:
            P = self.mu_probs[index]
        else:
            P = self.tau_probs[index]

        # Randomly sample pendulum for now...
        eps = torch.randn(self.latent_size)

        return L, P, q, eps


def train_pmns(nepoch, batch_size=512, learning_rate=0.001, beta=1e-3,
               input_size=1, latent_size=3, encoder_num_units=[500, 100],
               decoder_num_units=[100, 100], output_size=1, question_size=1):

    net = Net(input_size, latent_size, encoder_num_units, decoder_num_units, output_size, question_size)
    latent_array = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # generate PMNS data
    L, prob_e_to_mu, prob_e_to_tau = generate_data(data_size=10)
    L_val, prob_e_to_mu_val, prob_e_to_tau_val = generate_data(data_size=5)
    # load data
    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 4}
    training_set = DatasetPMNS(L, prob_e_to_mu, prob_e_to_tau, latent_size)
    validation_set = DatasetPMNS(L_val, prob_e_to_mu_val, prob_e_to_tau_val, latent_size)
    trainloader = data.DataLoader(training_set, **params)
    valloader = data.DataLoader(validation_set, **params)

    for epoch in range(nepoch):  # loop over the dataset multiple times
        running_tloss = 0.0
        running_vloss = 0.0
        for i, batch in enumerate(trainloader):
            net.train()
     
            # get the inputs
            inputs, labels, questions, epsilons = batch
            labels = labels.view(-1,output_size)

            # zero the parameter gradients
            optimizer.zero_grad()
         
            # forward + backward + optimize
            outputs, lat_sigs, lat_mus = net(inputs.float(), questions.float(), epsilons.float())
            loss = criterion(outputs, labels.type(torch.float)) \
             - beta/2.*((torch.log(lat_sigs**2) - lat_sigs**2 - lat_mus**2).sum(dim=1)-latent_size).mean()
            loss.backward()
            optimizer.step()
         
            # print statistics
            running_tloss += loss.item()
            if i == len(trainloader) - 1:    # print every 2000 mini-batches
                print('[%d, %5d] train loss: %.5f' %
                      (epoch + 1, i + 1, running_tloss / len(trainloader)))
                running_tloss = 0.0

        for i, batch in enumerate(valloader):
            net.eval()
     
            # get the inputs
            inputs, labels, questions, epsilons = batch
            labels = labels.view(-1,output_size)

            # forward + backward + optimize
            outputs, lat_sigs, lat_mus = net(inputs.float(), questions.float(), epsilons.float())
            vloss = criterion(outputs, labels.type(torch.float)) \
             - beta/2.*((torch.log(lat_sigs**2) - lat_sigs**2 - lat_mus**2).sum(dim=1)-latent_size).mean()
         
            # print statistics
            running_vloss += vloss.item()
            if i == len(valloader) - 1:    # print every 2000 mini-batches
                print('[%d, %5d] val loss: %.5f' %
                      (epoch + 1, i + 1, running_vloss / len(valloader)))
                running_vloss = 0.0

            if epoch == (nepoch - 1):
                latent_array.append(epsilons.numpy()*lat_sigs.detach().numpy()+lat_mus.detach().numpy())
            


    print('Finished Training')

    return net, np.array(latent_array)


