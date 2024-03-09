"""
EECS 445 - Introduction to Machine Learning
Fall 2023 - Project 2
Target CNN
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import config


class Target(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        ## TODO: define each layer
        p = 5 // 2
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2,padding=p)
        self.pool =  torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,stride=2,padding=p)
        self.conv3 = torch.nn.Conv2d(in_channels=64,out_channels=8,kernel_size=5,stride=2,padding=p)
        self.fc_1 = torch.nn.Linear(32,2)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        ## TODO: initialize the parameters for [self.fc_1]
        nn.init.normal_(self.fc_1.weight, 0.0, 1 / sqrt(32))
        nn.init.constant_(self.fc_1.bias, 0.0)        

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """
        N, C, H, W = x.shape

        ## TODO: forward pass
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.nn.functional.relu(self.conv3(x))

        x = x.view(-1,32)
        x = self.fc_1(x)

        return x
