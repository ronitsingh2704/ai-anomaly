"""
Anomaly Detection Models Module
Implements various machine learning algorithms for anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional, List
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Main anomaly detector class that supports multiple algorithms."""
    
    def __init__(self, algorithm: str, **kwargs):
        """
        Initialize anomaly detector.
        
        Args:
            algorithm: Algorithm name ('isolation_forest', 'pca_based', 'autoencoder', 'lstm')
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm.lower()
        self.kwargs = kwargs
        self.detector = None
        self.is_trained = False
        self.feature_names = None
        
        # Initialize specific detector
        if self.algorithm == 'isolation_forest':
            self.detector = IsolationForestDetector(**kwargs)
        elif self.algorithm == 'pca_based':
            self.detector = PCABasedDetector(**kwargs)
        elif self.algorithm == 'autoencoder':
            self.detector = AutoencoderDetector(**kwargs)
        elif self.algorithm == 'lstm':
            self.detector = LSTMDetector(**kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Train the anomaly detector.
        
        Args:
            X: Training data
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        self.detector.fit(X)
        self.is_trained = True
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Input data
            
        Returns:
            np.ndarray: Anomaly scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.detector.predict(X)
    
    def get_feature_importance(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get feature importance for anomaly detection.
        
        Args:
            X: Input data
            
        Returns:
            np.ndarray: Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if hasattr(self.detector, 'get_feature_importance'):
            return self.detector.get_feature_importance(X)
        else:
            # Fallback: return uniform importance
            return np.ones(X.shape[-1]) / X.shape[-1]

class IsolationForestDetector:
    """Isolation Forest anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, **kwargs):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray) -> None:
        """Train the Isolation Forest model."""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model.fit(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores (negative values for anomalies)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to 0-100 scale (higher = more anomalous)
        # Isolation Forest returns negative scores for anomalies
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score != min_score:
            normalized_scores = (max_score - scores) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(scores)
        
        return normalized_scores * 100
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get feature importance based on feature contributions to anomaly scores."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Calculate feature-wise contribution using permutation importance
        base_scores = self.predict(X)
        feature_importance = np.zeros(X.shape[-1])
        
        for i in range(X.shape[-1]):
            X_permuted = X.copy()
            # Permute the i-th feature
            if len(X) > 1:
                np.random.shuffle(X_permuted[:, i])
            else:
                X_permuted[:, i] = 0  # For single sample, set to neutral value
            
            permuted_scores = self.predict(X_permuted)
            feature_importance[i] = np.mean(np.abs(base_scores - permuted_scores))
        
        # Normalize
        if feature_importance.sum() > 0:
            feature_importance = feature_importance / feature_importance.sum()
        
        return feature_importance

class PCABasedDetector:
    """PCA-based anomaly detector using reconstruction error."""
    
    def __init__(self, n_components: int = 5, **kwargs):
        """
        Initialize PCA-based detector.
        
        Args:
            n_components: Number of principal components
        """
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.threshold = None
    
    def fit(self, X: np.ndarray) -> None:
        """Train the PCA model."""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        # Calculate reconstruction errors for training data to set threshold
        X_reconstructed = self._reconstruct(X_scaled)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        self.threshold = np.percentile(errors, 95)  # 95th percentile as threshold
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores based on reconstruction error."""
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Reconstruct data
        X_reconstructed = self._reconstruct(X_scaled)
        
        # Calculate reconstruction errors
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Convert to 0-100 scale
        if self.threshold > 0:
            scores = (errors / self.threshold) * 50  # Scale relative to training threshold
        else:
            scores = errors * 10  # Fallback scaling
        
        # Cap at 100
        scores = np.minimum(scores, 100)
        
        return scores
    
    def _reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct data using PCA components."""
        X_transformed = self.pca.transform(X)
        return self.pca.inverse_transform(X_transformed)
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get feature importance based on PCA component loadings."""
        if self.pca is None:
            return np.ones(X.shape[-1]) / X.shape[-1]
        
        # Use absolute values of component loadings weighted by explained variance
        loadings = np.abs(self.pca.components_)
        explained_variance_ratio = self.pca.explained_variance_ratio_
        
        # Weight loadings by explained variance
        weighted_loadings = loadings * explained_variance_ratio.reshape(-1, 1)
        
        # Sum across components to get feature importance
        feature_importance = np.sum(weighted_loadings, axis=0)
        
        # Normalize
        if feature_importance.sum() > 0:
            feature_importance = feature_importance / feature_importance.sum()
        
        return feature_importance

class AutoencoderDetector:
    """Neural network autoencoder for anomaly detection."""
    
    def __init__(self, encoding_dim: int = 10, epochs: int = 50, **kwargs):
        """
        Initialize autoencoder detector.
        
        Args:
            encoding_dim: Dimension of encoding layer
            epochs: Number of training epochs
        """
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.input_dim = None
    
    def fit(self, X: np.ndarray) -> None:
        """Train the autoencoder model."""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        self.input_dim = X_scaled.shape[1]
        
        # Build autoencoder
        self._build_model()
        
        # Train the model
        self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate reconstruction errors for training data to set threshold
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        self.threshold = np.percentile(errors, 95)
    
    def _build_model(self) -> None:
        """Build the autoencoder model."""
        # Encoder
        input_layer = keras.Input(shape=(self.input_dim,))
        
        # Hidden layers for encoder
        encoded = layers.Dense(
            max(self.encoding_dim * 2, 16), 
            activation='relu'
        )(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(
            max(self.encoding_dim * 2, 16), 
            activation='relu'
        )(encoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores based on reconstruction error."""
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Get reconstructions
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction errors
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        # Convert to 0-100 scale
        if self.threshold and self.threshold > 0:
            scores = (errors / self.threshold) * 50
        else:
            scores = errors * 10  # Fallback scaling
        
        scores = np.minimum(scores, 100)
        
        return scores
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get feature importance based on reconstruction error contribution."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        
        # Calculate feature-wise reconstruction errors
        feature_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=0)
        
        # Normalize to get importance
        if feature_errors.sum() > 0:
            feature_importance = feature_errors / feature_errors.sum()
        else:
            feature_importance = np.ones(len(feature_errors)) / len(feature_errors)
        
        return feature_importance

class LSTMDetector:
    """LSTM autoencoder for time series anomaly detection."""
    
    def __init__(self, sequence_length: int = 10, hidden_units: int = 50, 
                 num_layers: int = 2, epochs: int = 50, dropout: float = 0.2, **kwargs):
        """
        Initialize LSTM detector.
        
        Args:
            sequence_length: Length of input sequences
            hidden_units: Number of LSTM units
            num_layers: Number of LSTM layers
            epochs: Number of training epochs
            dropout: Dropout rate
        """
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.n_features = None
    
    def fit(self, X: np.ndarray) -> None:
        """Train the LSTM autoencoder model."""
        if len(X.shape) != 3:
            raise ValueError("LSTM requires 3D input (samples, sequence_length, features)")
        
        self.n_features = X.shape[2]
        
        # Scale the data (reshape for scaling, then back to 3D)
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Build model
        self._build_model()
        
        # Train the model
        self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate threshold
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=(1, 2))
        self.threshold = np.percentile(errors, 95)
    
    def _build_model(self) -> None:
        """Build the LSTM autoencoder model."""
        # Encoder
        input_layer = keras.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers for encoder
        x = input_layer
        for i in range(self.num_layers - 1):
            x = layers.LSTM(
                self.hidden_units, 
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.dropout
            )(x)
        
        # Last encoder layer (return sequences=False to get encoding)
        encoded = layers.LSTM(
            self.hidden_units, 
            return_sequences=False,
            dropout=self.dropout,
            recurrent_dropout=self.dropout
        )(x)
        
        # Repeat the encoding for decoder
        x = layers.RepeatVector(self.sequence_length)(encoded)
        
        # LSTM layers for decoder
        for i in range(self.num_layers):
            x = layers.LSTM(
                self.hidden_units, 
                return_sequences=True,
                dropout=self.dropout,
                recurrent_dropout=self.dropout
            )(x)
        
        # Output layer
        decoded = layers.TimeDistributed(
            layers.Dense(self.n_features, activation='linear')
        )(x)
        
        # Create and compile model
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores for sequences."""
        if len(X.shape) != 3:
            raise ValueError("LSTM requires 3D input (samples, sequence_length, features)")
        
        # Scale the data
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Get reconstructions
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        
        # Calculate reconstruction errors (mean over sequence and features)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=(1, 2))
        
        # Convert to 0-100 scale
        if self.threshold and self.threshold > 0:
            scores = (errors / self.threshold) * 50
        else:
            scores = errors * 10
        
        scores = np.minimum(scores, 100)
        
        return scores
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get feature importance for LSTM model."""
        if len(X.shape) != 3:
            raise ValueError("LSTM requires 3D input (samples, sequence_length, features)")
        
        # Scale the data
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(original_shape)
        
        # Get reconstructions
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        
        # Calculate feature-wise reconstruction errors across all sequences and samples
        feature_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=(0, 1))
        
        # Normalize to get importance
        if feature_errors.sum() > 0:
            feature_importance = feature_errors / feature_errors.sum()
        else:
            feature_importance = np.ones(len(feature_errors)) / len(feature_errors)
        
        return feature_importance
