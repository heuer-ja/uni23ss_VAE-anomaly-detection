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