import os
import sys

sys.dont_write_bytecode = True

import numpy as np 

import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset , DataLoader
from torch.optim import Adam
from torch.nn.functional import softplus
from torch.distributions import Normal, kl_divergence

# own classes
from dataset import DatasetMNIST
from helper_vae_architecture import ProbabilisticVAEArchitecture


class VAE_CNN(nn.Module):
    def __init__(
                self,
                io_size:int=784,
                latent_size:int=10
                ) -> None:
        super().__init__()
        self.io_size = io_size
        self.latent_size = latent_size

        self.L = 10 #  Number of samples in the latent space to detect the anomaly.
        self.prior =  Normal(0,1)
        
        architecture:ProbabilisticVAEArchitecture= self.get_architecture()
        self.encoder:nn.Module =  architecture.encoder
        self.latent_mu:nn.Module = architecture.latent_mu
        self.latent_sigma:nn.Module = architecture.latent_sigma
        self.decoder:nn.Module = architecture.decoder
        self.recon_mu:nn.Module = architecture.recon_mu
        self.recon_sigma:nn.Module = architecture.recon_sigma

    def get_architecture(self)  -> ProbabilisticVAEArchitecture:
        architecture:ProbabilisticVAEArchitecture = ProbabilisticVAEArchitecture(
                # ENCODER
                encoder = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten(),
                ),
                # LATENT SPACE
                latent_mu=nn.Linear(64 * 7 * 7, self.latent_size),
                latent_sigma=nn.Linear(64 * 7 * 7, self.latent_size),
                # DECODER
                decoder = nn.Sequential(
                    nn.Linear(self.latent_size, 64 * 7 * 7),
                    nn.Unflatten(1, (64, 7, 7)),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
                    nn.Sigmoid(),
                ),
                # RECONSTRUCTION
                 recon_mu= nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28*28, self.io_size)
                ),
                recon_sigma=  nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28*28, self.io_size)
                ),
        )
        return architecture
    

    #=================[FORWARD PASS]==============
    def forward(self, x: torch.Tensor) -> dict:
        x = x.float()  # Convert input to torch.float32

        pred_result = self.predict(x)
        x = x.unsqueeze(0)  # unsqueeze to broadcast input across sample dimension (L)

        # LOSS
        log_lik = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x).mean(
            dim=0)  # average over sample dimension
        log_lik = log_lik.mean(dim=0).sum()
        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        loss = kl - log_lik

        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)
     
    def predict(self, x) -> dict:
        batch_size = len(x)

        # ENCODING
        x = self.encoder(x)

        # LATENT SPACE - softplus to ensure values are positive
        latent_mu = self.latent_mu(x) 
        latent_sigma = softplus(self.latent_sigma(x)) 

        dist = Normal(latent_mu, latent_sigma)

        z = dist.rsample([self.L])  
        z = z.view(self.L * batch_size, self.latent_size) 

        # DECODER 
        decoded = self.decoder(z)

        recon_mu = self.recon_mu(decoded)
        recon_mu = recon_mu.view(self.L, -1, 1, 28, 28)

        recon_sigma = softplus(self.recon_sigma(decoded))
        recon_sigma = recon_sigma.view(self.L, -1, 1, 28, 28)

        return dict(
            z=z, latent_dist=dist, latent_mu=latent_mu,latent_sigma=latent_sigma, 
            recon_mu=recon_mu, recon_sigma=recon_sigma)
