import torch.nn as nn 

class ProbVAEArchitecture:
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
        recon_sigma:nn.Module,
        ) -> None:
        
        self.encoder:nn.Module = encoder 
        self.latent_mu:nn.Module = latent_mu,
        self.latent_sigma:nn.Module = latent_sigma,
        self.decoder:nn.Module = decoder,
        self.recon_mu:nn.Module = recon_mu,
        self.recon_sigma:nn.Module = recon_sigma,
        pass