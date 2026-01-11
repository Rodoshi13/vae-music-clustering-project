"""
vae_conditional.py - Conditional VAE with genre information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    """VAE conditioned on genre label"""
    
    def __init__(self, input_dim=556, latent_dim=40, hidden_dim=256, n_classes=10):
        super(ConditionalVAE, self).__init__()
        
        self.n_classes = n_classes
        
        # Genre embedding
        self.genre_embedding = nn.Embedding(n_classes, 50)
        
        # Encoder
        encoder_input = input_dim + 50  # features + genre embedding
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        decoder_input = latent_dim + 50  # latent + genre embedding
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x, genre):
        """Encode with genre conditioning"""
        genre_emb = self.genre_embedding(genre)
        h_input = torch.cat([x, genre_emb], dim=1)
        h = self.encoder(h_input)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Sample from latent space"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, genre):
        """Decode with genre conditioning"""
        genre_emb = self.genre_embedding(genre)
        z_input = torch.cat([z, genre_emb], dim=1)
        return self.decoder(z_input)
    
    def forward(self, x, genre):
        """Forward pass"""
        mu, logvar = self.encode(x, genre)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, genre)
        return recon_x, mu, logvar, z
