import sys
sys.dont_write_bytecode = True

# libs
import numpy as np
import torch 
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import Normal, kl_divergence
from abc import ABC, abstractmethod

# own classes
from helper_classes import ProbabilisticVAEArchitecture

class IVAE(nn.Module, ABC):
    def __init__(self, 
                 io_size:int,
                latent_size:int) -> None:
        super().__init__()
        self.io_size = io_size
        self.latent_size = latent_size
        self.L = 10 #  Number of samples in the latent space to detect the anomaly.
        self.prior =  Normal(0,1)
        
        # architecture
        architecture:ProbabilisticVAEArchitecture= self.get_architecture()
        self.encoder:nn.Module =  architecture.encoder
        self.latent_mu:nn.Module = architecture.latent_mu
        self.latent_sigma:nn.Module = architecture.latent_sigma
        self.decoder:nn.Module = architecture.decoder
        self.recon_mu:nn.Module = architecture.recon_mu
        self.recon_sigma:nn.Module = architecture.recon_sigma
    
    @abstractmethod
    def get_architecture(self)  -> ProbabilisticVAEArchitecture:
        pass 

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
        # INPUT
        batch_size = len(x)
        shape:[int] = list(x.shape)

        # ENCODING
        x_encoded = self.encoder(x)

        # LATENT SPACE - softplus to ensure values are positive
        latent_mu = self.latent_mu(x_encoded) 
        latent_sigma = softplus(self.latent_sigma(x_encoded)) 
        # try catch for ValueError in case of nan values
        try:
            dist = Normal(latent_mu, latent_sigma)
        except ValueError:
            print("Error")
            print("x", x)
            print("x_encoded", x_encoded)
            raise ValueError('latent_mu or latent_sigma contain nan values')

        z = dist.rsample([self.L])  
        z = z.view(self.L * batch_size, self.latent_size) 

        # DECODER 
        decoded = self.decoder(z)
        recon_mu = self.recon_mu(decoded)
        recon_sigma = softplus(self.recon_sigma(decoded))

        # reshape to [L, batch_size, io_size]
        #   - KDD1999 (tabular) [10, 128, 121]       
        #   - MNIST   (image)   [10, 128, 1, 28, 28] 
        view_shape:[int] = [self.L, *shape]
        recon_mu = recon_mu.view(*view_shape)       
        recon_sigma = recon_sigma.view(*view_shape)
        
        return dict(
            z=z, latent_dist=dist, latent_mu=latent_mu,latent_sigma=latent_sigma, 
            recon_mu=recon_mu, recon_sigma=recon_sigma)

    #=================[ANOMALY DETECTION]==============
    def is_anomaly(self, x: torch.Tensor, alpha: float = 0.05):
        p = self.reconstruction_probability(x)
        is_ano = p < alpha
        return is_ano, p
    
    def reconstruction_probability(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        with torch.no_grad():
            pred = self.predict(x)

        recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])
        x = x.unsqueeze(0)

        # calc probability, so that shape is [batch_size]
        x_len_shape = len(x.shape)
        p:torch.Tensor = recon_dist.log_prob(x).exp().mean(dim=0)
        for _ in range(x_len_shape - 2):
            p = p.mean(dim=-1)
    
        return p
    
    def reconstruct(self, x: torch.Tensor, device) -> torch.Tensor:
        '''
        batch of input data x

        returns batch of decoded data (reconstructed)
        '''
        x = x.float().to(device)

        with torch.no_grad():
            x_encoded = self.encoder(x)
            latent_mu = self.latent_mu(x_encoded)
            decoded  = self.decoder(latent_mu)
        
        return decoded 


    

class VAE_CNN(IVAE):
    def __init__(
                self,
                io_size:int=784,
                latent_size:int=300
                ) -> None:
        super().__init__(io_size=io_size, latent_size=latent_size)
        return 

    def get_architecture(self)  -> ProbabilisticVAEArchitecture:
        architecture:ProbabilisticVAEArchitecture = ProbabilisticVAEArchitecture(
                # ENCODER
                encoder = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2),  # Leaky ReLU activation
                    nn.BatchNorm2d(32),  # Batch normalization
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2),  # Leaky ReLU activation
                    nn.BatchNorm2d(64),  # Batch normalization
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

class VAE_Tabular(IVAE):
    def __init__(
            self,
            io_size:int=121,
            latent_size:int=10
            ) -> None:
        super().__init__(io_size=io_size, latent_size=latent_size)
        return 

    def get_architecture(self)  -> ProbabilisticVAEArchitecture:
        architecture:ProbabilisticVAEArchitecture = ProbabilisticVAEArchitecture(
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
