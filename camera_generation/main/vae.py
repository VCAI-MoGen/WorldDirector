import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraEncoder(nn.Module):
    def __init__(self, in_channels=9, hidden_dim=64, latent_dim=32):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),

            # downsample 1: N → 2M+1
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=2),
            nn.ReLU(),

            # downsample 2: 2M+1 → M+1
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, stride=2),
        )

        self.to_mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        """
        x: (B, T, 9)
        return:
            mu, logvar: (B, D, T')
        """
        x = x.transpose(1, 2)   # (B, 9, T)
        h = self.conv(x)        # (B, 128, T')
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

class CameraDecoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, out_channels=9):
        super().__init__()

        self.deconv = nn.Sequential(
            # upsample 1: M+1 → 2M+1
            nn.ConvTranspose1d(
                latent_dim, hidden_dim,
                kernel_size=1,
                stride=2
            ),
            nn.ReLU(),

            # upsample 2: 2M+1 → 4M+1
            nn.ConvTranspose1d(
                hidden_dim, hidden_dim,
                kernel_size=1,
                stride=2
            ),
            nn.ReLU(),

            # projection
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, z):
        """
        z: (B, D, T')
        """
        x_hat = self.deconv(z)    # (B, 9, T)
        return x_hat.transpose(1, 2)

class CameraVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = CameraEncoder(latent_dim=latent_dim)
        self.decoder = CameraDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)        # (B, D, T')
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)             # (B, T, 9)
        return x_hat, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        mu = mu.transpose(1, 2)
        return mu

    def decode(self, z):
        z = z.transpose(1, 2)
        x = self.decoder(z)
        return x
