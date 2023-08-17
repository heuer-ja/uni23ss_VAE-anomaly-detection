import numpy as np 

import torch 
import torch.nn as nn
from torch.nn.functional import softplus
from torch.distributions import Normal, kl_divergence

from helper_vae_architecture import ProbabilisticVAEArchitecture




class VAE_CNN(nn.Module):
    def __init__(
                self,
                io_size:int=28*28,
                latent_size:int=10
                ) -> None:
        super().__init__(io_size=io_size, latent_size=latent_size)

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
        pred_result = self.predict(x)
        print('forward 1')
        x = x.unsqueeze(0)  # unsqueeze to broadcast input across sample dimension (L)
        print('forward 2')
        log_lik = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x).mean(
            dim=0)  # average over sample dimension
        print('forward 3')

        log_lik = log_lik.mean(dim=0).sum()
        print('forward 4')

        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        print('forward 5')

        loss = kl - log_lik
        print('forward 6')

        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)
     
    def predict(self, x) -> dict:
        batch_size = len(x)
        # ENCODING
        
        print(np.shape(x))
        print('predict 1')
        x = self.encoder(x)
        print(np.shape(x))
        print('predict 2')

        
        # LATENT SPACE
        latent_mu = self.latent_mu(x)
        print(np.shape(latent_mu))
        print('predict 3')
        latent_sigma = softplus(self.latent_sigma(x)) # softplus to ensure values are positive
        print('predict 4')
        print(np.shape(latent_sigma))


        dist = Normal(latent_mu, latent_sigma)
        print('predict 5')

        z = dist.rsample([self.L])  # shape: [L, batch_size, latent_size]
        z = z.view(self.L * batch_size, self.latent_size)
        print('predict 6')

        # DECODER 
        decoded = self.decoder(z)
        print('predict 7')
        print(np.shape(decoded))

        recon_mu = self.recon_mu(decoded)
        print('predict 8')
        recon_mu = recon_mu.view(self.L, batch_size, self.io_size // 1)
        print('predict 9')
        recon_sigma = softplus(self.recon_sigma(decoded))
        print('predict 10')
        recon_sigma = recon_sigma.view(self.L, batch_size, self.io_size // 1)
        print('predict 11')

        return dict(
            z=z, latent_dist=dist, latent_mu=latent_mu,latent_sigma=latent_sigma, 
            recon_mu=recon_mu, recon_sigma=recon_sigma)



