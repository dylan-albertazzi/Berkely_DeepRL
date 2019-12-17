#!/usr/bin/env python
import os
import pickle
import tensorflow as tf
import torch
import numpy as np
import tf_util
import gym
import load_policy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.loads(f.read())

class Net(nn.Module):
    def __init__(self,input_placeholder, output_size):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_placeholder, 255)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(255, 255)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(255, output_size)


    def forward(self, x): #x is the batch size
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert dataset')
    expert_data = load_data(args.expert_policy_file)

    # observations = torch.LongTensor(expert_data['observations'])
    # actions = torch.LongTensor(expert_data['actions'])
    observations = torch.from_numpy(expert_data['observations'])
    observations = observations.type(torch.FloatTensor)
    actions = torch.from_numpy(expert_data['actions'])
    actions = actions.type(torch.FloatTensor)
    actions = np.squeeze(actions, axis=1)
    print('loaded and two numpy arrays of observations and action built')
    
    data_size = np.shape(observations)[0]
    obs_size = np.shape(observations)[1]
    actions_size = np.shape(actions)[1]

    # observations = torch.from_numpy(observations)
    # observations = observations.LongTensor()
    # # observations = observations.(dtype=torch.LongTensor)
    # # observations = torch.tensor(observations, dtype=torch.double)
    # actions = torch.from_numpy(actions)
    # actions = actions.LongTensor()

    # actions = torch.tensor(actions, dtype=torch.double)


    
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    net = Net(obs_size, actions_size) #create a nn

    #define loss function
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    #Train the network
    for epoch in range(2):
        running_loss = 0.0
        k = 0
        for i in range(data_size):
            optimizer.zero_grad()

            outputs = net(observations[k,:])
            loss = criterion(outputs, actions[k,:])
            loss.backward()
            optimizer.step()
            k = k + 1
         # print statistics
            running_loss += loss.item()

            if i % 10 == 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
    print('Finished Training')
    torch.save(net, 'myFirstNet.pt')

if __name__ == '__main__':
    main()