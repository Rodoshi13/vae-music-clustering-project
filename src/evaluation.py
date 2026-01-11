"""
evaluation.py - Evaluation metrics
"""

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import pandas as pd


class MetricsEvaluator:
    """Calculate clustering metrics"""
    
    @staticmethod
    def evaluate_all(features, labels):
        """Compute all metrics"""
        return {
            'Silhouette Score': silhouette_score(features, labels),
            'Calinski-Harabasz Index': calinski_harabasz_score(features, labels),
            'Davies-Bouldin Index': davies_bouldin_score(features, labels)
        }
    
    @staticmethod
    def compare_methods(methods_dict, features):
        """Compare multiple methods"""
        results = {}
        for method_name, labels in methods_dict.items():
            results[method_name] = MetricsEvaluator.evaluate_all(features, labels)
        return pd.DataFrame(results).T
