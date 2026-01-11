"""
clustering.py - Clustering functions
"""

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """Clustering helper"""
    
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def cluster_kmeans(self, features):
        """K-Means clustering"""
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(features)
        logger.info(f"K-Means: {np.unique(labels)}")
        return labels, kmeans
    
    def baseline_pca_kmeans(self, features, n_components=10):
        """PCA + K-Means baseline"""
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features)
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(pca_features)
        
        logger.info(f"PCA variance: {pca.explained_variance_ratio_.sum():.4f}")
        return labels, pca_features, pca
