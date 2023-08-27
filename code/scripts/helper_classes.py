import torch.nn as nn 
from dataclasses import dataclass


@dataclass
class ProbabilisticVAEArchitecture:
    '''
    Helper class for defining the architecture of a probabilistic VAE.
    '''
    encoder:nn.Module
    latent_mu:nn.Module
    latent_sigma:nn.Module
    decoder:nn.Module
    recon_mu:nn.Module
    recon_sigma:nn.Module




@dataclass
class LogTrainPreds:
    '''
    Helper class for logging training progress.
    '''
    loss: [float]
    kl: [float]
    recon_loss: [float]


from enum import Enum
# Enum for model selection
class ModelToTrain(Enum):
    CNN_MNIST = 1,
    FULLY_TABULAR = 2



class LabelsMNIST(int, Enum):
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9
