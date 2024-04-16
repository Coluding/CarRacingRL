import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import Sequence
import os
from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def save_checkpoint(self):
        if not hasattr(self, 'checkpoint_file'):
            raise AttributeError("Attribute 'checkpoint_file' not found")
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, path=None):
        if not hasattr(self, 'checkpoint_file'):
            raise AttributeError("Attribute 'checkpoint_file' not found")

        print('... loading checkpoint ...')
        if path is not None:
            self.load_state_dict(torch.load(os.path.join(path, self.name + '_td3')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))

    def compute_conv_dims(self, shape):
        shape = torch.zeros(1, *shape)
        return int(np.prod(self.conv_layer(shape).size()))


class DQN(nn.Module, AbstractModel):
    def __init__(self, alpha: float, n_actions: int, input_dims: Sequence[int],
                 fc1_dims: int, fc2_dims: int,
                 channel_dims: Sequence[int] = (128, 256, 512, 1024),
                 name: str = "DQN", chkpt_dir: str = "tmp/dq3_backup"
                 ):
        super(DQN, self).__init__()

        convs = [nn.Conv2d(input_dims[0], channel_dims[0], 3), nn.LeakyReLU()]
        for i in range(len(channel_dims) - 1):
            convs.append(nn.Conv2d(channel_dims[i], channel_dims[i + 1], 3, 2))
            convs.append(nn.LeakyReLU())

        self.conv_layer = nn.Sequential(*convs)
        self.fc_input_dims = self.compute_conv_dims(input_dims)

        self.fc_layer = nn.Sequential(
            nn.Linear(self.fc_input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_dql')

    def forward(self, state: torch.Tensor):

        conv_out = self.conv_layer(state)
        conv_out = conv_out.view(conv_out.size()[0], -1)
        return self.fc_layer(conv_out)


class CriticNetwork(nn.Module, AbstractModel):
    def __init__(self, alpha: float, input_dims: Sequence[int],
                 fc1_dims: int, fc2_dims: int,
                 channel_dims: Sequence[int] = (128, 256, 512, 1024),
                 name: str = "critic", chkpt_dir: str = "tmp/critic_backup"
                 ):
        super(CriticNetwork, self).__init__()

        convs = []
        for i in range(len(channel_dims) - 1):
            convs.append(nn.Conv2d(channel_dims[i], channel_dims[i + 1], 3))
            convs.append(nn.LeakyReLU())
            convs.append(nn.MaxPool2d(2))

        self.conv_layer = nn.Sequential(*convs)
        self.fc_input_dims = self.compute_conv_dims(input_dims)

        self.fc_layer = nn.Sequential(
            nn.Linear(self.fc_input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_critic')

    def forward(self, state: torch.Tensor):
        conv_out = self.conv_layer(state)
        conv_out = conv_out.view(conv_out.size()[0], -1)
        return self.fc_layer(conv_out)


class ActorNetwork(nn.Module, AbstractModel):
    def __init__(self, alpha: float, input_dims: Sequence[int],
                 fc1_dims: int, fc2_dims: int, n_actions: int,
                 channel_dims: Sequence[int] = (128, 256, 512, 1024),
                 name: str = "actor", chkpt_dir: str = "tmp/actor_backup"
                 ):
        super(ActorNetwork, self).__init__()

        convs = []
        for i in range(len(channel_dims) - 1):
            convs.append(nn.Conv2d(channel_dims[i], channel_dims[i + 1], 3))
            convs.append(nn.MaxPool2d(2))

        self.conv_layer = nn.Sequential(*convs)
        self.fc_input_dims = self.compute_conv_dims(input_dims)

        self.fc_layer = nn.Sequential(
            nn.Linear(self.fc_input_dims, fc1_dims),
            nn.LeakyReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LeakyReLU(),
            nn.Linear(fc2_dims, n_actions)
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_actor')

    def forward(self, state: torch.Tensor) -> torch.distributions.categorical.Categorical:
        conv_out = self.conv_layer(state)
        conv_out = conv_out.view(conv_out.size()[0], -1)
        actions = self.fc_layer(conv_out)
        pa = torch.nn.Softmax()(actions)
        dist = Categorical(probs=pa)

        return dist

