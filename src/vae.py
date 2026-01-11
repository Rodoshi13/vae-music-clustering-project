"""
vae.py - Variational Autoencoder Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicVAE(nn.Module):
    """Simple VAE for audio features"""
    
    def __init__(self, input_dim=256, latent_dim=20, hidden_dim=256):
        super(BasicVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode to latent space"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Sample from latent space"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z


class VAELoss(nn.Module):
    """VAE Loss Function"""
    
    def __init__(self, beta=1.0):
        super(VAELoss, self).__init__()
        self.beta = beta
    
    def forward(self, recon_x, x, mu, logvar):
        """
        Reconstruction + KL divergence loss
        """
        # Reconstruction loss
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = BCE + self.beta * KLD
        return total_loss, BCE, KLD
