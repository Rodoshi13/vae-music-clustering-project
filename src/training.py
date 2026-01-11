"""
training.py - Training utilities for VAE models
"""

import os
import torch
from tqdm import tqdm
import numpy as np


def train_vae(model, loss_fn, train_loader, test_loader, 
              epochs=50, lr=1e-3, device='cpu', save_path=None):
    """
    Train VAE model with support for both 2-value and 3-value batch formats
    
    Args:
        model: VAE model to train
        loss_fn: Loss function (VAELoss)
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        epochs: Number of training epochs (default: 50)
        lr: Learning rate (default: 1e-3)
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save trained model (optional)
    
    Returns:
        tuple: (trained_model, history_dict)
            - trained_model: The trained VAE model
            - history_dict: Dictionary with 'train_loss' and 'test_loss' lists
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'test_loss': []
    }
    
    for epoch in range(epochs):
        # ========== TRAINING PHASE ==========
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Handle both 2-value and 3-value batch formats
            # Easy task: (features, idx) - 2 values
            # Medium/Hard task: (features, genre, idx) - 3 values
            if len(batch) == 3:
                batch_features, _, _ = batch  # Unpack 3 values
            else:
                batch_features, _ = batch      # Unpack 2 values
            
            batch_features = batch_features.to(device)
            
            # Clamp features to valid range [0, 1] for reconstruction
            batch_features = torch.clamp(batch_features, 0, 1)
            
            # Forward pass - Model returns 4 values: (recon, mu, logvar, z)
            recon_batch, mu, logvar, z = model(batch_features)
            loss, recon_loss, kl_loss = loss_fn(recon_batch, batch_features, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * batch_features.size(0)
        
        # Average training loss
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # ========== VALIDATION PHASE ==========
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle both 2-value and 3-value batch formats
                if len(batch) == 3:
                    batch_features, _, _ = batch
                else:
                    batch_features, _ = batch
                
                batch_features = batch_features.to(device)
                batch_features = torch.clamp(batch_features, 0, 1)
                
                # Forward pass - Model returns 4 values: (recon, mu, logvar, z)
                recon_batch, mu, logvar, z = model(batch_features)
                loss, _, _ = loss_fn(recon_batch, batch_features, mu, logvar)
                
                # Accumulate loss
                test_loss += loss.item() * batch_features.size(0)
        
        # Average test loss
        test_loss /= len(test_loader.dataset)
        history['test_loss'].append(test_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"✓ Model saved to {save_path}")
    
    return model, history


def extract_latent_features(model, data_loader, device='cpu'):
    """
    Extract latent features from trained VAE
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader for the data
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        tuple: (latent_features_array, track_ids_array)
            - latent_features_array: numpy array of shape (n_samples, latent_dim)
            - track_ids_array: numpy array of track IDs
    """
    model.eval()
    latent_features = []
    track_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle both 2-value and 3-value batch formats
            if len(batch) == 3:
                batch_features, _, ids = batch
            else:
                batch_features, ids = batch
            
            batch_features = batch_features.to(device)
            batch_features = torch.clamp(batch_features, 0, 1)
            
            # Extract latent features - Model returns 4 values
            _, _, _, z = model(batch_features)
            
            latent_features.append(z.cpu().numpy())
            track_ids.extend(ids if isinstance(ids, list) else ids.cpu().numpy())
    
    # Stack all batches
    latent_features = np.vstack(latent_features)
    track_ids = np.array(track_ids)
    
    return latent_features, track_ids


def get_reconstruction_error(model, data_loader, device='cpu'):
    """
    Calculate reconstruction error for each sample
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader for the data
        device: Device to use
    
    Returns:
        numpy array of reconstruction errors
    """
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle both 2-value and 3-value batch formats
            if len(batch) == 3:
                batch_features, _, _ = batch
            else:
                batch_features, _ = batch
            
            batch_features = batch_features.to(device)
            batch_features = torch.clamp(batch_features, 0, 1)
            
            # Get reconstruction - Model returns 4 values
            recon_batch, _, _, _ = model(batch_features)
            
            # Calculate MSE reconstruction error
            mse = torch.mean((recon_batch - batch_features) ** 2, dim=1)
            reconstruction_errors.append(mse.cpu().numpy())
    
    return np.concatenate(reconstruction_errors)