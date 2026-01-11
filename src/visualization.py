"""
visualization.py - Visualization utilities for VAE results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Visualizer:
    """Visualization utilities for VAE results"""
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """Plot training curves - FIXED VERSION"""
        
        # Handle both dict and list formats
        if isinstance(history, dict):
            train_loss = history.get('train_loss', [])
            test_loss = history.get('test_loss', [])
        else:
            # If history is a list/array of losses
            train_loss = history
            test_loss = []
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Training Loss
        if len(train_loss) > 0:
            axes[0].plot(train_loss, label='Train', marker='o', markersize=3)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Train vs Test
        if len(test_loss) > 0:
            axes[1].plot(train_loss, label='Train', marker='o', markersize=3)
            axes[1].plot(test_loss, label='Test', marker='s', markersize=3)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Total Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].plot(train_loss, label='Loss', marker='o', markersize=3)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Total Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_tsne(features, labels, title="t-SNE Visualization", save_path=None):
        """Plot t-SNE visualization"""
        
        print(f"Computing t-SNE (this may take a minute)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot with colors
        scatter = ax.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=30
        )
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(title)
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_reconstruction(original, reconstructed, n_samples=10, save_path=None):
        """Plot original vs reconstructed samples"""
        
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
        
        for i in range(n_samples):
            # Original
            axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=12)
            
            # Reconstructed
            axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Reconstructed', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_latent_space(z, labels=None, title="Latent Space", save_path=None):
        """Plot 2D latent space representation"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab20', alpha=0.6, s=30)
            plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(z[:, 0], z[:, 1], alpha=0.6, s=30)
        
        ax.set_xlabel('Z1')
        ax.set_ylabel('Z2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig
    
    @staticmethod
    def plot_clusters(features, labels, title="Cluster Visualization", save_path=None):
        """Plot clusters using t-SNE"""
        
        print("Computing t-SNE for cluster visualization...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_2d = tsne.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot each cluster with different color
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[color],
                label=f'Cluster {label}',
                alpha=0.7,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        return fig
