from abc import abstractmethod, ABC

import numpy as np 

import torch 
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import Normal, kl_divergence

from pvae_architecture import ProbVAEArchitecture


class VAE(nn.Module, ABC):
    #=================[ARCHITECTURE]==============
    def __init__(
            self,
            io_size:int,
            latent_size:int
    ) -> None:
        super().__init__()
        self.io_size = io_size
        self.latent_size = latent_size

        self.L = 10 #  Number of samples in the latent space to detect the anomaly.
        self.prior =  Normal(0,1)
        
        architecture:ProbVAEArchitecture= self.get_architecture()
        self.encoder:nn.Module =  architecture.encoder
        self.latent_mu:nn.Module = architecture.latent_mu
        self.latent_sigma:nn.Module = architecture.latent_sigma
        self.decoder:nn.Module = architecture.decoder
        self.recon_mu:nn.Module = architecture.recon_mu
        self.recon_sigma:nn.Module = architecture.recon_sigma
        return 
    
    @abstractmethod
    def get_architecture(self)  -> ProbVAEArchitecture:
        pass

    #=================[FORWARD PASS]==============
    def forward(self, x: torch.Tensor) -> dict:
        pred_result = self.predict(x)
        x = x.unsqueeze(0)  # unsqueeze to broadcast input across sample dimension (L)
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
        
        # LATENT SPACE
        latent_mu = self.latent_mu(x)
        latent_sigma = softplus(self.latent_sigma(x)) # softplus to ensure values are positive

        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample([self.L])  # shape: [L, batch_size, latent_size]
        z = z.view(self.L * batch_size, self.latent_size)

        # DECODER 
        decoded = self.decoder(z)
        recon_mu = self.recon_mu(decoded)
        recon_mu = recon_mu.view(self.L, batch_size, self.io_size // 1)
        recon_sigma = softplus(self.recon_sigma(decoded))
        recon_sigma = recon_sigma.view(self.L, batch_size, self.io_size // 1)
                
        return dict(
            z=z, latent_dist=dist, latent_mu=latent_mu,latent_sigma=latent_sigma, 
            recon_mu=recon_mu, recon_sigma=recon_sigma)
    

class VAE_Tabular(VAE):
    def __init__(
            self,
            io_size:int=121,
            latent_size:int=10
            ) -> None:
        super().__init__(io_size=io_size, latent_size=latent_size)

    def get_architecture(self)  -> ProbVAEArchitecture:
        architecture:ProbVAEArchitecture = ProbVAEArchitecture(
            # ENCODER
            encoder = nn.Sequential(
                    nn.Linear(self.io_size // 1, self.io_size // 2, dtype=torch.float32),
                    nn.ReLU(),
                    nn.Linear(self.io_size // 2, self.io_size // 4, dtype=torch.float32),
                    nn.ReLU()
            ),
            # LATENT SPACE
            latent_mu     = torch.nn.Linear(self.io_size // 4, self.latent_size, dtype=torch.float32),
            latent_sigma  = torch.nn.Linear(self.io_size // 4, self.latent_size, dtype=torch.float32),
            # DECODER
            decoder = nn.Sequential(
                nn.Linear(self.latent_size, self.io_size // 4, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(self.io_size // 4, self.io_size // 2, dtype=torch.float32),
                nn.ReLU(),
            ),
            # RECONSTRUCTION
            recon_mu     = nn.Linear(self.io_size // 2, self.io_size // 1, dtype=torch.float32),
            recon_sigma  = nn.Linear(self.io_size // 2, self.io_size // 1, dtype=torch.float32)
        )
        return architecture