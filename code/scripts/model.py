import sys
sys.dont_write_bytecode = True

# libs
import numpy as np
import torch 
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softplus
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import mse_loss
from abc import ABC, abstractmethod

# own classes
from helper_classes import VAEArchitecture

class IVAE(nn.Module, ABC):
    def __init__(self, 
                 io_size:int,
                latent_size:int,
                is_probabilistic:bool,
                ) -> None:
        super().__init__()
        self.is_probabilistic = is_probabilistic # determines whether VAE is probabilistic or not
        self.io_size = io_size
        self.latent_size = latent_size
        self.prior =  Normal(0,1)
        
        # architecture
        architecture:VAEArchitecture= self.get_architecture()
        self.encoder:nn.Module =  architecture.encoder
        self.latent_mu:nn.Module = architecture.latent_mu
        self.latent_sigma:nn.Module = architecture.latent_sigma
        self.decoder:nn.Module = architecture.decoder
        
        # probabilistic architecture
        self.recon_mu:nn.Module = architecture.recon_mu
        self.recon_sigma:nn.Module = architecture.recon_sigma

    @abstractmethod
    def get_architecture(self)  -> VAEArchitecture:
        pass 

    #=================[FORWARD PASS]==============
    def forward(self, x: Tensor) -> dict:
        x = x.float()  # Convert input to torch.float32

        # PREDICTION
        pred_result = self.predict(x)
        
        # LOSS 
        loss_dict:dict = self.get_loss(x, pred_result)

        return dict(**loss_dict, **pred_result)

    def predict(self, x) -> dict:
        # INPUT
        x = x.float()  # Convert input to torch.float32
        batch_size = len(x)
        shape:[int] = list(x.shape)

        # ENCODING
        x_encoded = self.encoder(x)

        # LATENT SPACE - softplus to ensure values are positive
        latent_mu = self.latent_mu(x_encoded) 
        latent_sigma = softplus(self.latent_sigma(x_encoded)) 

        dist = Normal(latent_mu, latent_sigma)

        # SAMPLE FROM LATENT SPACE with repameterization trick 
        z = dist.rsample()
        z = z.view(batch_size, self.latent_size)
    
        # DECODER 
        decoded = self.decoder(z)

        # PROBABILISTIC
        recon_mu = None
        recon_sigma = None

        if self.is_probabilistic:
            recon_mu = self.recon_mu(decoded)
            recon_sigma = softplus(self.recon_sigma(decoded))

        return dict(
            latent_sigma=latent_sigma, 
            latent_dist = dist,
            z = z,
            decoded=decoded,       
            recon_mu=recon_mu, 
            recon_sigma=recon_sigma
        )
    
    def get_reconstruction_loss(self, x:Tensor, pred_result:dict) -> Tensor:
        ''' calculates reconstruction loss for each instance in batch
           - pVAE: log  likelihood
            - dVAE: mse loss

        no KL divergence is calculated


        returns 
        [batch_size] Tensor, 
        each entry is the reconstruction loss for the corresponding instance in the batch
        '''
        recon_loss = None

        # probablistic VAE
        if self.is_probabilistic:
            # Reconstructed Distribution
            dist_recon:Normal = Normal(pred_result['recon_mu'], pred_result['recon_sigma'])
                        
            # reshape x to match dist_recon.scale.shape
            x = x.view(dist_recon.scale.shape) # [batch_size, 784] | [batch_size, 121]

            # .log_prob(x) [batch_size, 784] | [batch_size, 121] -> mean() [batch_size] 
            log_lik = dist_recon.log_prob(x).mean(dim=1) 
 
            recon_loss =  log_lik

        # normal/deterministic VAE
        else:
            recon_x:Tensor = pred_result['decoded']	
            mse_loss_list = []
            
            for (xi, recon_xi) in zip(x, recon_x):
                mse = mse_loss(recon_xi, xi, reduction='sum')
                mse_loss_list.append(mse.item())  

            mse_loss_batch = torch.Tensor(mse_loss_list) 
            recon_loss = mse_loss_batch
        return recon_loss

    def get_loss(self, x:Tensor, pred_result:dict) -> dict:
        loss_dict:dict = dict()

        # probablistic VAE
        if self.is_probabilistic:

            # Distributions
            dist_latent:Normal = pred_result['latent_dist']
            dist_recon:Normal = Normal(pred_result['recon_mu'], pred_result['recon_sigma'])
                        
            # reshape x to match dist_recon.scale.shape
            x = x.view(dist_recon.scale.shape) # [batch_size, 784] | [batch_size, 121]

            # .log_prob(x) [batch_size, 784] | [batch_size, 121] -> mean() [784] | [121] -> sum()/mean single value
            log_lik = dist_recon.log_prob(x).mean(dim=1).sum() # mean or sum or ?
            
            kl = kl_divergence(dist_latent, self.prior).mean(dim=0).sum() # single value

            loss = kl - log_lik
            loss_dict['kl'] = kl
            loss_dict['recon_loss'] = log_lik
            loss_dict['loss'] = loss

        # normal/deterministic VAE
        else:
            recon_x:Tensor = pred_result['decoded']	
            posterior:Normal = pred_result['latent_dist']

            # LOSS
            recon_loss = mse_loss(recon_x, x, reduction='sum')
            kl = kl_divergence(posterior, self.prior).sum()
            loss = kl + recon_loss

            loss_dict['kl'] = kl
            loss_dict['recon_loss'] = recon_loss
            loss_dict['loss'] = loss

        return loss_dict
    
    #=================[RECONSTRUCTION]==============
    def reconstruct(self, x: Tensor) -> Tensor:
        '''
        batch of input data x

        returns batch of decoded data (reconstructed)
        '''
        with torch.no_grad():
            pred = self.predict(x)
            reconstructions = pred['decoded']

        return reconstructions


    #=================[ANOMALY DETECTION]==============
    def is_anomaly(self, x: Tensor, alpha: float = 0.05):
        '''
            case 1: pVAE anomaly if recon_prob  < alpha
            case 2: dVAE anomaly if recon_error > alpha
        '''
        # predict
        with torch.no_grad():
            pred:dict = self.predict(x)

        # Probabilistic VAE (pVAE)
        if self.is_probabilistic:
            # Compute the reconstruction probability for each instance in the batch
            recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])
            x = x.unsqueeze(0)  # Add an extra dimension to match the batch size
            x_len_shape = len(x.shape)
            
            # Calculate the log probabilities and take the exponential
            log_probs = recon_dist.log_prob(x).exp()
            
            # Compute the mean along dimensions other than the batch dimension
            for _ in range(x_len_shape - 2):
                log_probs = log_probs.mean(dim=-1)

            # Sum over the remaining dimension to get the individual instance probabilities
            recon_prob = log_probs.sum(dim=-1)

            # Determine if each instance is an anomaly based on the threshold (alpha)
            is_ano = recon_prob < alpha
            return is_ano, recon_prob

        # Deterministic VAE (dVAE)
        else:
            # Calculate the reconstruction error for each instance in the batch using MSE
            recon_x = pred['decoded']
            
            # Compute the MSE error for each instance
            mse_error = mse_loss(recon_x, x, reduction='sum')
            
            # Compute the mean along dimensions other than the batch dimension
            for _ in range(len(mse_error.shape) - 1):
                mse_error = mse_error.mean(dim=-1)

            # Determine if each instance is an anomaly based on the threshold (alpha)
            is_ano = mse_error > alpha
            return is_ano, mse_error
       

class VAE_CNN(IVAE):
    def __init__(
                self,
                is_probabilistic:bool,
                io_size:int=784,
                latent_size:int=15
                ) -> None:
        super().__init__(
            is_probabilistic=is_probabilistic,
            io_size=io_size, 
            latent_size=latent_size
        )
        return
    
    def get_architecture(self)  -> VAEArchitecture:
        architecture_3layer = VAEArchitecture(
            # ENCODER
            encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 3x3 kernel
                nn.ReLU(),
                nn.BatchNorm2d(32),  
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 3x3 kernel
                nn.ReLU(),
                nn.BatchNorm2d(64),  
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 3x3 kernel, stride 2 for downsampling
                nn.ReLU(),
                nn.BatchNorm2d(128), 
                nn.Flatten(),
            ),
            # LATENT SPACE
            latent_mu=nn.Linear(128 * 14 * 14, self.latent_size),
            latent_sigma=nn.Linear(128 * 14 * 14, self.latent_size),
            # DECODER
            decoder = nn.Sequential(
                nn.Linear(self.latent_size, 128 * 14 * 14),
                nn.Unflatten(1, (128, 14, 14)),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x4 kernel
                nn.ReLU(),
                nn.BatchNorm2d(64), 
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),  # 3x3 kernel
                nn.ReLU(),
                nn.BatchNorm2d(32),  
                nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),  # 3x3 kernel
                nn.Sigmoid(),
            ),
            # RECONSTRUCTION
            recon_mu= None if not self.is_probabilistic else nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, self.io_size)
            ),
            recon_sigma= None if not self.is_probabilistic else nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28, self.io_size)
            ) 
        )
        
        return architecture_3layer
    

class VAE_Tabular(IVAE):
    def __init__(
            self,
            is_probabilistic:bool,
            io_size:int=121,
            latent_size:int=10
            ) -> None:
        super().__init__(
            is_probabilistic=is_probabilistic,
            io_size=io_size, 
            latent_size=latent_size,
        )
        return 

    def get_architecture(self)  -> VAEArchitecture:
        architecture:VAEArchitecture = VAEArchitecture(
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
                nn.Linear(self.io_size // 2, self.io_size, dtype=torch.float32),
                nn.ReLU(),
            ),

            # RECONSTRUCTION
            recon_mu     = None if not self.is_probabilistic else nn.Linear(
                self.io_size // 1, self.io_size // 1, dtype=torch.float32
            ),
            recon_sigma  = None if not self.is_probabilistic else nn.Linear(
                self.io_size // 1, self.io_size // 1, dtype=torch.float32
            )
        )
        return architecture
