import torch 
import torch.nn as nn



class VAE_Tabular(nn.Module):
    def __init__(
            self
    ) -> None:
        super().__init__()

        self.io_size = 121
        self.latent_size = 10

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Linear(self.io_size // 1, self.io_size // 2, dtype=float),
            nn.ReLU(),
            nn.Linear(self.io_size // 2, self.io_size // 4, dtype=float),
        )    
            
        self.z_mean     =   torch.nn.Linear(self.io_size // 4, self.latent_size, dtype=float)
        self.z_log_var  = torch.nn.Linear(self.io_size // 4, self.latent_size, dtype=float)

        # DECODER
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, self.io_size // 4, dtype=float),
            nn.ReLU(),
            nn.Linear(self.io_size // 4, self.io_size // 2, dtype=float),
            nn.ReLU(),
            nn.Linear(self.io_size // 2, self.io_size // 1, dtype=float),
        )

        # TODO: map to z and log var again
        return 
    
     
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
    