import torch.nn as nn 
from dataclasses import dataclass




class VAEArchitecture:
    '''
    Helper class for defining the architecture of a VAE.
    '''
    def __init__(
        self,
        encoder:nn.Module,
        latent_mu:nn.Module,
        latent_sigma:nn.Module,
        decoder:nn.Module
    ):
        self.encoder = encoder
        self.latent_mu = latent_mu
        self.latent_sigma = latent_sigma
        self.decoder = decoder
        pass

class pVAEArchitecture(VAEArchitecture):
    '''
    Helper class for defining the architecture of a probabilistic VAE.
    '''
    def __init__(
        self,
        encoder:nn.Module,
        latent_mu:nn.Module,
        latent_sigma:nn.Module,
        decoder:nn.Module,
        recon_mu:nn.Module,
        recon_sigma:nn.Module
    ):
        super().__init__(encoder, latent_mu, latent_sigma, decoder)
        self.recon_mu = recon_mu
        self.recon_sigma = recon_sigma
        pass



class VAELogTrain():
    '''
    Helper class for logging training progress of normal VAE
    '''
    def __init__(self, loss:[float]) -> None:
        self.loss = loss
        pass

class pVAELogTrain(VAELogTrain):
    '''
    Helper class for logging training progress of probabilistic VAE
    '''
    def __init__(self, 
        loss: [float],
        kl: [float],
        recon_loss: [float]      
    ):
        super().__init__(loss)
        self.kl = kl
        self.recon_loss = recon_loss


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




@dataclass
class StrIntMapping:
    label:str
    encoded:int

class LabelsKDD1999(Enum):
    Normal = StrIntMapping('normal', 0)
    Probe = StrIntMapping('probe', 1)
    DoS = StrIntMapping('dos', 2)
    U2R = StrIntMapping('u2r', 3)
    R2L = StrIntMapping('r2l', 4)

# Iterate over LabelsKDD1999 and print all int values using list comprehension
print([class_label.value.encoded for class_label in LabelsKDD1999])
