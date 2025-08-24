"""
Enhanced Streamlit-based Anomaly Detection System for Multivariate Time Series Data
Combined from all project modules: app.py, anomaly_models.py, data_processor.py, 
feature_attribution.py, and utils.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import io
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
from datetime import datetime, timedelta
import time
import json
import re
import warnings
from collections import Counter

# Machine Learning imports
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings('ignore')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
load_dotenv()
def validate_training_period(start_time: Union[datetime, int], 
                           end_time: Union[datetime, int],
                           timestamp_column: Optional[str] = None) -> Tuple[bool, str]:
    """Validate that the training period meets minimum requirements."""
    try:
        if timestamp_column:
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)
                
            duration = end_time - start_time
            
            if duration.total_seconds() < 0:
                return False, "End time must be after start time"
            
            if duration.total_seconds() < 72 * 3600:
                return False, "Training period should be at least 72 hours for reliable results"
        else:
            if end_time <= start_time:
                return False, "End index must be greater than start index"
            
            if end_time - start_time < 100:
                return False, "Training period should contain at least 100 data points"
                
        return True, ""
        
    except Exception as e:
        return False, f"Error validating training period: {str(e)}"

def format_anomaly_score(score: float) -> int:
    """Format anomaly score to integer between 0-100."""
    if pd.isna(score) or np.isinf(score):
        return 0
    
    formatted_score = max(0, min(100, round(float(score))))
    return formatted_score

def get_severity_label(score: int) -> str:
    """Get severity label for anomaly score."""
    if score <= 10:
        return "Normal"
    elif score <= 30:
        return "Slight"
    elif score <= 60:
        return "Moderate"
    elif score <= 90:
        return "Significant"
    else:
        return "Severe"

def get_severity_color(score: int) -> str:
    """Get color code for anomaly score visualization."""
    if score <= 10:
        return "#2ECC71"
    elif score <= 30:
        return "#F1C40F"
    elif score <= 60:
        return "#E67E22"
    elif score <= 90:
        return "#E74C3C"
    else:
        return "#8E44AD"

def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str, Dict]:
    """Validate CSV structure for anomaly detection."""
    validation_info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': 0,
        'timestamp_columns': 0,
        'missing_data_percentage': 0,
        'constant_columns': 0
    }
    
    try:
        if df.empty:
            return False, "CSV file is empty", validation_info
        
        if len(df.columns) == 0:
            return False, "No columns found in CSV", validation_info
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        validation_info['numeric_columns'] = len(numeric_cols)
        
        timestamp_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'time' in col.lower() or 'date' in col.lower():
                timestamp_cols.append(col)
        validation_info['timestamp_columns'] = len(timestamp_cols)
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        validation_info['missing_data_percentage'] = (missing_cells / total_cells) * 100
        
        constant_cols = []
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        validation_info['constant_columns'] = len(constant_cols)
        
        if validation_info['numeric_columns'] == 0:
            return False, "No numeric columns found for anomaly detection", validation_info
        
        if validation_info['missing_data_percentage'] > 80:
            return False, "Too much missing data (>80%)", validation_info
        
        if len(df) < 50:
            return False, "Insufficient data: At least 50 rows required", validation_info
        
        if validation_info['numeric_columns'] - validation_info['constant_columns'] == 0:
            return False, "No valid features (all numeric columns are constant)", validation_info
        
        return True, "", validation_info
        
    except Exception as e:
        return False, f"Error validating CSV: {str(e)}", validation_info

def export_results_summary(results_df: pd.DataFrame, 
                          anomaly_scores: np.ndarray,
                          feature_names: List[str]) -> str:
    """Create a text summary of anomaly detection results."""
    summary_lines = []
    summary_lines.append("=" * 50)
    summary_lines.append("ANOMALY DETECTION RESULTS SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    
    summary_lines.append(f"Total data points: {len(results_df)}")
    summary_lines.append(f"Features analyzed: {len(feature_names)}")
    summary_lines.append("")
    
    severity_counts = {
        "Normal (0-10)": len([s for s in anomaly_scores if s <= 10]),
        "Slight (11-30)": len([s for s in anomaly_scores if 11 <= s <= 30]),
        "Moderate (31-60)": len([s for s in anomaly_scores if 31 <= s <= 60]),
        "Significant (61-90)": len([s for s in anomaly_scores if 61 <= s <= 90]),
        "Severe (91-100)": len([s for s in anomaly_scores if s > 90])
    }
    
    summary_lines.append("SEVERITY DISTRIBUTION:")
    for severity, count in severity_counts.items():
        percentage = (count / len(anomaly_scores)) * 100
        summary_lines.append(f"  {severity}: {count} ({percentage:.1f}%)")
    
    summary_lines.append("")
    
    top_anomaly_indices = np.argsort(anomaly_scores)[-10:][::-1]
    summary_lines.append("TOP 10 ANOMALIES:")
    for i, idx in enumerate(top_anomaly_indices, 1):
        score = anomaly_scores[idx]
        severity = get_severity_label(score)
        summary_lines.append(f"  {i:2d}. Row {idx:4d}: Score {score:3.0f} ({severity})")
    
    summary_lines.append("")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 50)
    
    return "\n".join(summary_lines)

# ============================================================================
# DATA PROCESSOR CLASS
# ============================================================================

class DataProcessor:
    """Handles all data processing operations for the anomaly detection system."""
    
    def __init__(self):
        self.feature_columns: Optional[List[str]] = None
        self.timestamp_column: Optional[str] = None
        self.scaler = None
        
    def load_data(self, uploaded_file: io.BytesIO) -> pd.DataFrame:
        """Load and perform initial validation of CSV data."""
        try:
            encodings = ['utf-8', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                raise ValueError("Unable to decode the CSV file. Please check the file encoding.")
            
            if df.empty:
                raise ValueError("The uploaded file is empty.")
                
            if len(df.columns) == 0:
                raise ValueError("No columns found in the CSV file.")
                
            if len(df) < 100:
                st.warning(" Dataset contains fewer than 100 rows. Consider using more data for better results.")
                
            if len(df) > 10000:
                st.warning(" Dataset contains more than 10,000 rows. Processing may take longer.")
                
            df = df.dropna(how='all')
            df = self._infer_data_types(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    def _infer_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer and convert appropriate data types."""
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                if self._is_timestamp_column(df_copy[col]):
                    try:
                        df_copy[col] = pd.to_datetime(df_copy[col], infer_datetime_format=True)
                    except:
                        pass
                else:
                    numeric_converted = pd.to_numeric(df_copy[col], errors='coerce')
                    if numeric_converted.notna().sum() / len(df_copy) > 0.5:
                        df_copy[col] = numeric_converted
                        
        return df_copy
    
    def _is_timestamp_column(self, series: pd.Series) -> bool:
        """Check if a series likely contains timestamp data."""
        if series.dtype != 'object':
            return False
            
        sample_values = series.dropna().head(10)
        
        if len(sample_values) == 0:
            return False
            
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{4}/\d{2}/\d{2}',
            r'\d{2}-\d{2}-\d{4}',
        ]
        
        matches = 0
        for value in sample_values:
            value_str = str(value)
            for pattern in timestamp_patterns:
                if re.search(pattern, value_str):
                    matches += 1
                    break
                    
        return matches / len(sample_values) > 0.7
    
    def identify_timestamp_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify potential timestamp columns in the dataframe."""
        timestamp_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['time', 'date', 'timestamp', 'ts', 'datetime']):
                timestamp_cols.append(col)
                continue
                
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                timestamp_cols.append(col)
                continue
                
            if self._is_timestamp_column(df[col]):
                timestamp_cols.append(col)
                
        return timestamp_cols
    
    def identify_numeric_columns(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> List[str]:
        """Identify numeric columns suitable for anomaly detection."""
        if exclude_columns is None:
            exclude_columns = []
            
        numeric_cols = []
        
        for col in df.columns:
            if col in exclude_columns:
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() > 1 and df[col].std() > 0:
                    numeric_cols.append(col)
                    
        return numeric_cols
    
    def preprocess_data(self, 
                       df: pd.DataFrame, 
                       feature_columns: List[str],
                       timestamp_column: Optional[str] = None) -> pd.DataFrame:
        """Preprocess data for anomaly detection."""
        try:
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found in data: {missing_cols}")
            
            feature_data = df[feature_columns].copy()
            feature_data = self._handle_missing_values(feature_data)
            feature_data = self._remove_constant_features(feature_data)
            
            if feature_data.shape[1] == 0:
                raise ValueError("No valid features remaining after preprocessing.")
            
            self.feature_columns = list(feature_data.columns)
            self.timestamp_column = timestamp_column
            
            self._validate_data_quality(feature_data)
            
            return feature_data
            
        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_processed = df.copy()
        
        missing_percentages = (df_processed.isnull().sum() / len(df_processed)) * 100
        
        cols_to_drop = missing_percentages[missing_percentages > 50].index.tolist()
        if cols_to_drop:
            st.warning(f" Dropping columns with >50% missing values: {cols_to_drop}")
            df_processed = df_processed.drop(columns=cols_to_drop)
        
        for col in df_processed.columns:
            missing_pct = missing_percentages[col]
            
            if missing_pct > 0:
                if missing_pct <= 5:
                    df_processed[col] = df_processed[col].ffill().bfill()
                elif missing_pct <= 20:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    df_processed[col] = df_processed[col].interpolate(method='linear').bfill()
        
        rows_before = len(df_processed)
        df_processed = df_processed.dropna()
        rows_after = len(df_processed)
        
        if rows_before != rows_after:
            st.info(f" Removed {rows_before - rows_after} rows with remaining missing values.")
        
        return df_processed
    
    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with zero or near-zero variance."""
        df_processed = df.copy()
        
        variances = df_processed.var()
        constant_cols = variances[variances < 1e-10].index.tolist()
        
        if constant_cols:
            st.info(f" Removing constant features: {constant_cols}")
            df_processed = df_processed.drop(columns=constant_cols)
        
        return df_processed
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """Validate data quality for anomaly detection."""
        if len(df) < 50:
            raise ValueError("Insufficient data: At least 50 rows required for anomaly detection.")
        
        if df.shape[1] == 0:
            raise ValueError("No valid features available for anomaly detection.")
        
        if np.isinf(df.values).any():
            raise ValueError("Dataset contains infinite values. Please clean your data.")
        
        max_values = df.max()
        if (max_values > 1e10).any():
            st.warning(" Some features have very large values. Consider normalizing your data.")
        
        total_variance = df.var().sum()
        if total_variance < 1e-6:
            st.warning(" Data has very low variance. Anomaly detection may not be meaningful.")
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive statistics for features."""
        stats = {
            'n_features': df.shape[1],
            'n_samples': df.shape[0],
            'feature_names': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'statistics': df.describe().to_dict()
        }
        
        return stats

# ============================================================================
# ANOMALY DETECTION MODELS
# ============================================================================

class AnomalyDetector:
    """Main anomaly detector class that supports multiple algorithms."""
    
    def __init__(self, algorithm: str, **kwargs):
        self.algorithm = algorithm.lower()
        self.kwargs = kwargs
        self.detector = None
        self.is_trained = False
        self.feature_names = None
        

        if self.algorithm == 'pca_based':
            self.detector = PCABasedDetector(**kwargs)
        elif self.algorithm == 'isolation_forest':
            self.detector = IsolationForestDetector(**kwargs)
        elif self.algorithm == 'autoencoder':
            self.detector = AutoencoderDetector(**kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """Train the anomaly detector."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        
        self.detector.fit(X)
        self.is_trained = True
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.detector.predict(X)
    
    def get_feature_importance(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get feature importance for anomaly detection."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if hasattr(self.detector, 'get_feature_importance'):
            return self.detector.get_feature_importance(X)
        else:
            return np.ones(X.shape[-1]) / X.shape[-1]

class IsolationForestDetector:
    """Isolation Forest anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, **kwargs):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray) -> None:
        """Train the Isolation Forest model."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model.fit(X_scaled)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        
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
        
        base_scores = self.predict(X)
        feature_importance = np.zeros(X.shape[-1])
        
        for i in range(X.shape[-1]):
            X_permuted = X.copy()
            if len(X) > 1:
                np.random.shuffle(X_permuted[:, i])
            else:
                X_permuted[:, i] = 0
            
            permuted_scores = self.predict(X_permuted)
            feature_importance[i] = np.mean(np.abs(base_scores - permuted_scores))
        
        if feature_importance.sum() > 0:
            feature_importance = feature_importance / feature_importance.sum()
        
        return feature_importance

class PCABasedDetector:
    """PCA-based anomaly detector using reconstruction error."""
    
    def __init__(self, n_components: int = 5, **kwargs):
        self.n_components = n_components
        self.pca = None
        self.scaler = StandardScaler()
        self.threshold = None
    
    def fit(self, X: np.ndarray) -> None:
        """Train the PCA model."""
        X_scaled = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        X_reconstructed = self._reconstruct(X_scaled)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        self.threshold = np.percentile(errors, 95)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores based on reconstruction error."""
        X_scaled = self.scaler.transform(X)
        X_reconstructed = self._reconstruct(X_scaled)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        if self.threshold > 0:
            scores = (errors / self.threshold) * 50
        else:
            scores = errors * 10
        
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
        
        loadings = np.abs(self.pca.components_)
        explained_variance_ratio = self.pca.explained_variance_ratio_
        
        weighted_loadings = loadings * explained_variance_ratio.reshape(-1, 1)
        feature_importance = np.sum(weighted_loadings, axis=0)
        
        if feature_importance.sum() > 0:
            feature_importance = feature_importance / feature_importance.sum()
        
        return feature_importance

class AutoencoderDetector:
    """Neural network autoencoder for anomaly detection."""
    
    def __init__(self, encoding_dim: int = 10, epochs: int = 50, **kwargs):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.input_dim = None
    
    def fit(self, X: np.ndarray) -> None:
        """Train the autoencoder model."""
        X_scaled = self.scaler.fit_transform(X)
        self.input_dim = X_scaled.shape[1]
        
        self._build_model()
        
        self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        self.threshold = np.percentile(errors, 95)
    
    def _build_model(self) -> None:
        """Build the autoencoder model."""
        input_layer = keras.Input(shape=(self.input_dim,))
        
        encoded = layers.Dense(
            max(self.encoding_dim * 2, 16), 
            activation='relu'
        )(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        decoded = layers.Dense(
            max(self.encoding_dim * 2, 16), 
            activation='relu'
        )(encoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores based on reconstruction error."""
        X_scaled = self.scaler.transform(X)
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        
        if self.threshold and self.threshold > 0:
            scores = (errors / self.threshold) * 50
        else:
            scores = errors * 10
        
        scores = np.minimum(scores, 100)
        return scores
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get feature importance based on reconstruction error contribution."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        X_reconstructed = self.model.predict(X_scaled, verbose=0)
        
        feature_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=0)
        
        if feature_errors.sum() > 0:
            feature_importance = feature_errors / feature_errors.sum()
        else:
            feature_importance = np.ones(len(feature_errors)) / len(feature_errors)
        
        return feature_importance

# ============================================================================
# FEATURE ATTRIBUTION CLASS
# ============================================================================

class FeatureAttributor:
    """Handles feature attribution for anomaly detection results."""
    
    def __init__(self, detector: AnomalyDetector, feature_names: List[str]):
        self.detector = detector
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
    def calculate_attributions(self, 
                             X: pd.DataFrame, 
                             anomaly_scores: np.ndarray) -> List[List[str]]:
        """Calculate feature attributions for each data point."""
        attributions = []
        
        if len(X) > 1000:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        for i in range(len(X)):
            if len(X) > 1000 and i % 100 == 0:
                progress = i / len(X)
                progress_bar.progress(progress)
                progress_text.text(f"Calculating feature attributions: {i}/{len(X)}")
            
            sample_attribution = self._get_sample_attribution(
                X.iloc[i:i+1], 
                anomaly_scores[i]
            )
            attributions.append(sample_attribution)
        
        if len(X) > 1000:
            progress_bar.progress(1.0)
            progress_text.text("Feature attribution completed!")
            
        return attributions
    
    def _get_sample_attribution(self, 
                               sample: pd.DataFrame, 
                               anomaly_score: float) -> List[str]:
        """Get feature attribution for a single sample."""
        if anomaly_score <= 10:
            return [''] * 7
        
        try:
            feature_importance = self.detector.get_feature_importance(sample)
            sample_values = np.abs(sample.values[0])
            
            if np.max(sample_values) > 0:
                sample_values_norm = sample_values / np.max(sample_values)
            else:
                sample_values_norm = sample_values
            
            combined_scores = feature_importance * (1 + sample_values_norm)
            
        except Exception:
            combined_scores = self._value_based_attribution(sample)
        
        top_indices = np.argsort(combined_scores)[::-1]
        top_features = [self.feature_names[i] for i in top_indices[:7]]
        
        while len(top_features) < 7:
            top_features.append('')
        
        return top_features
    
    def _value_based_attribution(self, sample: pd.DataFrame) -> np.ndarray:
        """Calculate feature attribution based on value magnitudes and deviations."""
        sample_values = sample.values[0]
        magnitude_scores = np.abs(sample_values)
        
        try:
            if hasattr(self.detector.detector, 'scaler'):
                scaler = self.detector.detector.scaler
                if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                    z_scores = np.abs((sample_values - scaler.mean_) / scaler.scale_)
                    combined_scores = magnitude_scores * (1 + z_scores)
                else:
                    combined_scores = magnitude_scores
            else:
                combined_scores = magnitude_scores
        except Exception:
            combined_scores = magnitude_scores
        
        if np.sum(combined_scores) > 0:
            combined_scores = combined_scores / np.sum(combined_scores)
        
        return combined_scores
    
    def get_global_feature_importance(self, 
                                    X: pd.DataFrame, 
                                    anomaly_scores: np.ndarray,
                                    threshold: float = 30) -> Dict[str, float]:
        """Calculate global feature importance across all anomalies."""
        anomaly_mask = anomaly_scores > threshold
        anomalous_samples = X[anomaly_mask]
        
        if len(anomalous_samples) == 0:
            return {feature: 0.0 for feature in self.feature_names}
        
        try:
            global_importance = self.detector.get_feature_importance(anomalous_samples)
        except Exception:
            global_importance = np.mean(np.abs(anomalous_samples.values), axis=0)
            global_importance = global_importance / np.sum(global_importance)
        
        importance_dict = {
            feature: float(importance) 
            for feature, importance in zip(self.feature_names, global_importance)
        }
        
# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

# Enhanced page configuration
st.set_page_config(
    page_title="AI Anomaly Detection Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': "mailto:support@anomalydetection.com",
        'About': "# Advanced AI Anomaly Detection Platform\n Built with Streamlit and ML"
    }
)

# Custom CSS for modern styling
def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            margin: 1rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif !important;
            color: #2c3e50 !important;
            font-weight: 600;
        }
        
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        }
        
        .hero-title {
            font-size: 3rem !important;
            font-weight: 700 !important;
            margin-bottom: 1rem !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .hero-subtitle {
            font-size: 1.3rem !important;
            opacity: 0.9;
            font-weight: 300 !important;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #4CAF50, #45a049) !important;
            color: white !important;
            border-radius: 15px !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            font-family: 'Poppins', sans-serif !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4) !important;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.05);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.12);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin: 0;
        }
        
        .metric-label {
            font-size: 0.95rem;
            color: #7f8c8d;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .severity-normal { color: #27ae60; }
        .severity-slight { color: #f39c12; }
        .severity-moderate { color: #e67e22; }
        .severity-significant { color: #e74c3c; }
        .severity-severe { color: #8e44ad; }
        
        .info-card {
            background: rgba(52, 152, 219, 0.1);
            border-left: 5px solid #3498db;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .success-card {
            background: rgba(39, 174, 96, 0.1);
            border-left: 5px solid #27ae60;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .warning-card {
            background: rgba(243, 156, 18, 0.1);
            border-left: 5px solid #f39c12;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .error-card {
            background: rgba(231, 76, 60, 0.1);
            border-left: 5px solid #e74c3c;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    defaults = {
        'current_page': 'Data Upload',
        'processed_data': None,
        'model_trained': False,
        'results_ready': False,
        'results_df': None,
        'anomaly_scores': None,
        'feature_attributions': None,
        'selected_features': None,
        'theme': 'light',
        'notifications_enabled': True,
        'auto_refresh': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Navigation system
def create_navigation():
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="color: white; margin: 0;"> AI Anomaly</h2>
            <p style="color: #bdc3c7; font-size: 0.9rem;">Detection Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    pages = [
        ("", "Data Upload"),
        ("", "Configuration"),
        ("", "Model Training"),
        ("", "Results & Analytics"),
        ("", "Settings")
    ]
    
    for icon, page_name in pages:
        if st.sidebar.button(f"{icon} {page_name}", key=f"nav_{page_name}", use_container_width=True):
            st.session_state.current_page = page_name
            st.rerun()
    
    st.sidebar.markdown("---")
    
    if st.session_state.processed_data is not None:
        st.sidebar.markdown("###  Quick Stats")
        rows, cols = st.session_state.processed_data.shape
        st.sidebar.metric("Data Points", f"{rows:,}")
        st.sidebar.metric("Features", cols)
        
        if st.session_state.results_ready:
            severe_count = len([s for s in st.session_state.anomaly_scores if s > 90])
            st.sidebar.metric(" Severe Anomalies", severe_count)

# Hero section
def display_hero_section():
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title"> AI Anomaly Detection Platform</h1>
            <p class="hero-subtitle">Advanced machine learning for multivariate time series anomaly detection</p>
        </div>
    """, unsafe_allow_html=True)

# Enhanced metric cards
def display_metric_cards(scores):
    st.markdown("###  Anomaly Distribution")
    
    severity_data = [
        ("Normal", len([s for s in scores if s <= 10]), "#27ae60"),
        ("Slight", len([s for s in scores if 11 <= s <= 30]), "#f39c12"),
        ("Moderate", len([s for s in scores if 31 <= s <= 60]), "#e67e22"),
        ("Significant", len([s for s in scores if 61 <= s <= 90]), "#e74c3c"),
        ("Severe", len([s for s in scores if s > 90]), "#8e44ad")
    ]
    
    cols = st.columns(5)
    
    for i, (label, count, color) in enumerate(severity_data):
        with cols[i]:
            percentage = (count / len(scores)) * 100
            delta = f"{percentage:.1f}%"
            
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {color};">{count}</div>
                    <div class="metric-label">{label}</div>
                    <div style="color: {color}; font-size: 0.9rem; margin-top: 0.5rem;">{delta}</div>
                </div>
            """, unsafe_allow_html=True)

# Enhanced data upload section
def data_upload_page():
    st.markdown("##  Data Upload & Validation")
    
    uploaded_file = st.file_uploader(
        "",
        type=['csv'],
        help="Upload your multivariate time series CSV file",
        label_visibility="collapsed"
    )
    
    with st.expander(" Data Format Requirements", expanded=not uploaded_file):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                **Required Format:**
                - CSV file with headers
                - Numerical columns for analysis
                - Optional timestamp column
                - Minimum 100 rows recommended
                
                **Supported Features:**
                - Multiple time series variables
                - Missing value handling
                - Automatic data type detection
            """)
        
        with col2:
            st.code("""
timestamp,temperature,pressure,vibration
2024-01-01 00:00:00,25.3,101.2,0.15
2024-01-01 01:00:00,25.1,101.5,0.14
2024-01-01 02:00:00,25.4,101.1,0.16
            """, language="csv")
    
    if uploaded_file is not None:
        try:
            with st.spinner(" Loading and validating data..."):
                time.sleep(0.5)
                processor = DataProcessor()
                data = processor.load_data(uploaded_file)
                
            st.markdown(f"""
                <div class="success-card">
                    <h4> Data Loaded Successfully!</h4>
                    <p><strong>Shape:</strong> {data.shape[0]:,} rows × {data.shape[1]} columns</p>
                </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs([" Data Preview", " Statistics", " Data Types"])
            
            with tab1:
                st.markdown("**First 10 rows:**")
                st.dataframe(data.head(10), use_container_width=True)
                
                st.markdown("**Last 5 rows:**")
                st.dataframe(data.tail(5), use_container_width=True)
            
            with tab2:
                st.markdown("**Descriptive Statistics:**")
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(data[numeric_cols].describe(), use_container_width=True)
                else:
                    st.warning("No numeric columns found for statistical analysis.")
            
            with tab3:
                st.markdown("**Column Information:**")
                info_df = pd.DataFrame({
                    'Column': data.columns,
                    'Data Type': data.dtypes.astype(str),
                    'Non-Null Count': data.count().values,
                    'Null Count': data.isnull().sum().values,
                    'Unique Values': [data[col].nunique() for col in data.columns]
                })
                st.dataframe(info_df, use_container_width=True)
            
            st.session_state.raw_data = data
            st.session_state.current_page = 'Configuration'
            
            if st.button(" Proceed to Configuration", type="primary", use_container_width=True):
                st.rerun()
                
        except Exception as e:
            st.markdown(f"""
                <div class="error-card">
                    <h4> Error Loading Data</h4>
                    <p>{str(e)}</p>
                    <p>Please ensure your CSV file follows the required format.</p>
                </div>
            """, unsafe_allow_html=True)

# Enhanced configuration page
def configuration_page():
    if 'raw_data' not in st.session_state:
        st.markdown("""
            <div class="warning-card">
                <h4> No Data Loaded</h4>
                <p>Please upload data first.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("← Back to Data Upload"):
            st.session_state.current_page = 'Data Upload'
            st.rerun()
        return
    
    data = st.session_state.raw_data
    st.markdown("##  Feature Selection & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Timestamp Column")
        timestamp_cols = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
        if not timestamp_cols:
            timestamp_cols = ['None']
        
        timestamp_col = st.selectbox(
            "Select timestamp column",
            timestamp_cols,
            help="Column containing time information"
        )
        
        if timestamp_col == 'None':
            timestamp_col = None
    
    with col2:
        st.markdown("###  Feature Columns")
        if timestamp_col:
            feature_options = [col for col in data.columns if col != timestamp_col]
        else:
            feature_options = list(data.columns)
        
        selected_features = st.multiselect(
            "Select features for anomaly detection",
            feature_options,
            default=feature_options[:min(10, len(feature_options))],
            help="Choose numerical columns for analysis"
        )
    
    if not selected_features:
        st.error(" Please select at least one feature column.")
        return
    
    st.markdown("###  Normal Period Definition")
    
    with st.expander(" Training Period Configuration", expanded=True):
        if timestamp_col and timestamp_col in data.columns:
            min_time = pd.to_datetime(data[timestamp_col]).min()
            max_time = pd.to_datetime(data[timestamp_col]).max()
            
            col1, col2 = st.columns(2)
            with col1:
                normal_start = st.date_input(
                    "Training start date",
                    value=min_time.date(),
                    min_value=min_time.date(),
                    max_value=max_time.date()
                )
            with col2:
                normal_end = st.date_input(
                    "Training end date",
                    value=min_time.date() + timedelta(days=7),
                    min_value=min_time.date(),
                    max_value=max_time.date()
                )
            
            training_days = (normal_end - normal_start).days
            st.info(f" Training period: {training_days} days")
            
        else:
            total_rows = len(data)
            col1, col2 = st.columns(2)
            with col1:
                normal_start_idx = st.number_input(
                    "Start row index",
                    min_value=0,
                    max_value=total_rows-1,
                    value=0
                )
            with col2:
                normal_end_idx = st.number_input(
                    "End row index",
                    min_value=normal_start_idx+1,
                    max_value=total_rows,
                    value=min(total_rows//3, 1000)
                )
            
            training_rows = normal_end_idx - normal_start_idx
            st.info(f" Training samples: {training_rows}")
    
    st.markdown("###  Algorithm Configuration")
    
    algorithm = st.selectbox(
        "Select ML Algorithm",
        ["PCA-based","Isolation Forest", "Autoencoder"],
        help="Choose the anomaly detection algorithm"
    )
    
    with st.expander(" Advanced Parameters", expanded=False):
        if algorithm == "Isolation Forest":
            col1, col2 = st.columns(2)
            with col1:
                contamination = st.slider(
                    "Expected anomaly rate",
                    0.01, 0.5, 0.1, 0.01,
                    help="Expected proportion of anomalies"
                )
            with col2:
                n_estimators = st.slider(
                    "Number of trees",
                    50, 500, 100, 10
                )
        
        elif algorithm == "PCA-based":
            n_components = st.slider(
                "PCA Components",
                1, min(len(selected_features), 20),
                min(5, len(selected_features)),
                help="Number of principal components"
            )
        
        elif algorithm == "Autoencoder":
            col1, col2 = st.columns(2)
            with col1:
                encoding_dim = st.slider(
                    "Encoding dimension",
                    2, min(len(selected_features), 50),
                    min(len(selected_features)//2, 10)
                )
            with col2:
                epochs = st.slider(
                    "Training epochs",
                    10, 200, 50, 10
                )
    
    if st.button("💾 Save Configuration & Proceed", type="primary", use_container_width=True):
        try:
            processor = DataProcessor()
            processed_data = processor.preprocess_data(
                data,
                feature_columns=selected_features,
                timestamp_column=timestamp_col
            )
            
            st.session_state.processed_data = processed_data
            st.session_state.selected_features = selected_features
            st.session_state.timestamp_col = timestamp_col
            st.session_state.algorithm = algorithm
            st.session_state.normal_period = {
                'start': normal_start if timestamp_col else normal_start_idx,
                'end': normal_end if timestamp_col else normal_end_idx
            }
            
            if algorithm == "Isolation Forest":
                st.session_state.algo_params = {
                    "contamination": contamination,
                    "n_estimators": n_estimators
                }
            elif algorithm == "PCA-based":
                st.session_state.algo_params = {"n_components": n_components}
            elif algorithm == "Autoencoder":
                st.session_state.algo_params = {
                    "encoding_dim": encoding_dim,
                    "epochs": epochs
                }
            
            st.session_state.current_page = 'Model Training'
            st.success(" Configuration saved! Ready for training.")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f" Configuration error: {str(e)}")

# Enhanced model training page
def model_training_page():
    if st.session_state.processed_data is None:
        st.warning(" Please complete data upload and configuration first.")
        return
    
    st.markdown("##  Model Training & Anomaly Detection")
    
    with st.expander(" Training Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algorithm", st.session_state.algorithm)
        with col2:
            st.metric("Features", len(st.session_state.selected_features))
        with col3:
            st.metric("Data Points", len(st.session_state.processed_data))
    
    if st.button(" Start Training & Detection", type="primary", use_container_width=True):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text(" Initializing ML model...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            detector = AnomalyDetector(
                algorithm=st.session_state.algorithm.lower().replace('-', '_').replace(' ', '_'),
                **st.session_state.algo_params
            )
            
            status_text.text(" Preparing training data...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            if st.session_state.timestamp_col:
                timestamp_series = pd.to_datetime(st.session_state.raw_data[st.session_state.timestamp_col])
                normal_start_dt = pd.to_datetime(st.session_state.normal_period['start'])
                normal_end_dt = pd.to_datetime(st.session_state.normal_period['end'])
                normal_mask = (timestamp_series >= normal_start_dt) & (timestamp_series <= normal_end_dt)
                normal_data = st.session_state.processed_data[normal_mask]
            else:
                start_idx = st.session_state.normal_period['start']
                end_idx = st.session_state.normal_period['end']
                normal_data = st.session_state.processed_data.iloc[start_idx:end_idx]
            
            if len(normal_data) == 0:
                st.error(" No data found in the specified normal period.")
                return
            
            status_text.text(" Training ML model...")
            progress_bar.progress(50)
            
            detector.fit(normal_data)
            
            status_text.text(" Detecting anomalies...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            anomaly_scores = detector.predict(st.session_state.processed_data)
            
            status_text.text(" Calculating feature attributions...")
            progress_bar.progress(90)
            time.sleep(0.5)
            
            attributor = FeatureAttributor(detector, st.session_state.selected_features)
            feature_attributions = attributor.calculate_attributions(
                st.session_state.processed_data, 
                anomaly_scores
            )
            
            results_df = st.session_state.raw_data.copy()
            results_df['Abnormality_score'] = [format_anomaly_score(score) for score in anomaly_scores]
            
            for i in range(7):
                col_name = f'top_feature_{i+1}'
                results_df[col_name] = [
                    attrs[i] if i < len(attrs) else '' 
                    for attrs in feature_attributions
                ]
            
            st.session_state.results_ready = True
            st.session_state.results_df = results_df
            st.session_state.anomaly_scores = anomaly_scores
            st.session_state.feature_attributions = feature_attributions
            st.session_state.model_trained = True
            
            progress_bar.progress(100)
            status_text.text(" Training completed successfully!")
            
            time.sleep(1)
            st.success(" Anomaly detection completed! Proceeding to results...")
            st.session_state.current_page = 'Results & Analytics'
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f" Training failed: {str(e)}")
            with st.expander("Debug Information"):
                st.code(traceback.format_exc())

# Enhanced results page
def results_analytics_page():
    if not st.session_state.results_ready:
        st.warning(" No results available. Please complete model training first.")
        return
    
    st.markdown("##  Results & Analytics Dashboard")
    
    scores = st.session_state.anomaly_scores
    
    display_metric_cards(scores)
    
    st.markdown("---")
    st.markdown("### 📱 Real-time Alert System")
    
    alert_col1, alert_col2 = st.columns([2, 1])
    with alert_col1:
        user_number = st.text_input(
            "Phone Number (with country code)",
            placeholder="+1234567890",
            help="Enter verified Twilio phone number"
        )
    
    with alert_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📨 Send SMS Alert", type="secondary"):
            severe_anomalies = len([s for s in scores if s > 90])
            if severe_anomalies > 0 and user_number.strip():
                try:
                    from twilio.rest import Client
                    ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
                    AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
                    TWILIO_NUMBER = "+17754587081"
                    
                    client = Client(ACCOUNT_SID, AUTH_TOKEN)
                    message = f"ALERT: {severe_anomalies} severe anomalies detected in your data!"
                    client.messages.create(
                        body=message,
                        from_=TWILIO_NUMBER,
                        to=user_number
                    )
                    st.success(f" SMS alert sent to {user_number}!")
                except Exception as e:
                    st.error(f"❌ SMS failed: {e}")
            elif severe_anomalies == 0:
                st.info(" No severe anomalies detected.")
            else:
                st.warning(" Please enter a valid phone number.")
    
    st.markdown("---")
    st.markdown("###  Interactive Visualizations")
    
    # Anomaly timeline
    fig_timeline = go.Figure()
    
    colors = []
    for score in scores:
        if score <= 10:
            colors.append('#27ae60')
        elif score <= 30:
            colors.append('#f39c12')
        elif score <= 60:
            colors.append('#e67e22')
        elif score <= 90:
            colors.append('#e74c3c')
        else:
            colors.append('#8e44ad')
    
    fig_timeline.add_trace(go.Scatter(
        y=scores,
        mode='markers+lines',
        marker=dict(
            color=colors,
            size=8,
            line=dict(width=1, color='white')
        ),
        line=dict(width=2, color='rgba(0,0,0,0.3)'),
        name='Anomaly Score',
        hovertemplate='<b>Point %{x}</b><br>Score: %{y:.1f}<br><extra></extra>'
    ))
    
    thresholds = [
        (10, "Normal/Slight", "rgba(39, 174, 96, 0.3)"),
        (30, "Slight/Moderate", "rgba(243, 156, 18, 0.3)"),
        (60, "Moderate/Significant", "rgba(230, 126, 34, 0.3)"),
        (90, "Significant/Severe", "rgba(231, 76, 60, 0.3)")
    ]
    
    for threshold, label, color in thresholds:
        fig_timeline.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=color.replace('0.3', '0.8'),
            annotation_text=label,
            annotation_position="right"
        )
    
    fig_timeline.update_layout(
        title={
            'text': 'Anomaly Scores Timeline',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Data Point Index",
        yaxis_title="Anomaly Score",
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        severity_labels = ["Normal", "Slight", "Moderate", "Significant", "Severe"]
        severity_counts = [
            len([s for s in scores if s <= 10]),
            len([s for s in scores if 11 <= s <= 30]),
            len([s for s in scores if 31 <= s <= 60]),
            len([s for s in scores if 61 <= s <= 90]),
            len([s for s in scores if s > 90]),
        ]
        
        fig_pie = px.pie(
            values=severity_counts,
            names=severity_labels,
            title="Severity Distribution",
            color_discrete_sequence=['#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_hist = px.histogram(
            x=scores,
            nbins=30,
            title="Score Distribution",
            labels={'x': 'Anomaly Score', 'y': 'Frequency'}
        )
        fig_hist.update_traces(marker_color='#3498db')
        fig_hist.update_layout(template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("###  Top Anomalies Analysis")
    
    top_indices = np.argsort(scores)[-15:][::-1]
    top_scores = [scores[i] for i in top_indices]
    
    fig_top = go.Figure()
    
    colors_top = [
        '#8e44ad' if s > 90 else
        '#e74c3c' if s > 60 else
        '#e67e22' if s > 30 else
        '#f39c12'
        for s in top_scores
    ]
    
    fig_top.add_trace(go.Bar(
        x=top_scores,
        y=[f"Point {i}" for i in top_indices],
        orientation='h',
        marker_color=colors_top,
        text=[f"{s:.1f}" for s in top_scores],
        textposition='inside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>'
    ))
    
    fig_top.update_layout(
        title="Top 15 Anomalies",
        xaxis_title="Anomaly Score",
        yaxis_title="Data Points",
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_top, use_container_width=True)
    
    if hasattr(st.session_state, 'feature_attributions'):
        st.markdown("###  Feature Contribution Analysis")
        
        all_features = []
        for attrs in st.session_state.feature_attributions:
            for feat in attrs[:3]:
                if feat and feat.strip():
                    all_features.append(feat)
        
        if all_features:
            feature_counts = Counter(all_features)
            
            top_features = feature_counts.most_common(10)
            feat_names = [f[0] for f in top_features]
            feat_counts = [f[1] for f in top_features]
            
            fig_features = px.bar(
                x=feat_counts,
                y=feat_names,
                orientation='h',
                title="Most Contributing Features",
                labels={'x': 'Contribution Count', 'y': 'Features'}
            )
            fig_features.update_traces(marker_color='#2ecc71')
            fig_features.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig_features, use_container_width=True)
    
    st.markdown("###  Detailed Results Table")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_score = st.slider("Minimum Score", 0, 100, 0)
    with filter_col2:
        max_score = st.slider("Maximum Score", 0, 100, 100)
    with filter_col3:
        show_rows = st.selectbox("Show rows", [50, 100, 500, "All"])
    
    filtered_df = st.session_state.results_df[
        (st.session_state.results_df['Abnormality_score'] >= min_score) &
        (st.session_state.results_df['Abnormality_score'] <= max_score)
    ].sort_values('Abnormality_score', ascending=False)
    
    if show_rows != "All":
        filtered_df = filtered_df.head(show_rows)
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )
    
    st.markdown("###  Export Results")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        csv_buffer = io.StringIO()
        st.session_state.results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label=" Download Enhanced CSV",
            data=csv_data,
            file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with download_col2:
        summary = create_summary_report(scores)
        st.download_button(
            label=" Download Summary Report",
            data=summary,
            file_name=f"anomaly_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def create_summary_report(scores):
    severity_counts = {
        "Normal": len([s for s in scores if s <= 10]),
        "Slight": len([s for s in scores if 11 <= s <= 30]),
        "Moderate": len([s for s in scores if 31 <= s <= 60]),
        "Significant": len([s for s in scores if 61 <= s <= 90]),
        "Severe": len([s for s in scores if s > 90])
    }
    
    report = f"""
ANOMALY DETECTION SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

OVERVIEW:
- Total Data Points: {len(scores):,}
- Algorithm Used: {st.session_state.get('algorithm', 'Unknown')}
- Features Analyzed: {len(st.session_state.get('selected_features', []))}

SEVERITY BREAKDOWN:
- Normal (0-10): {severity_counts['Normal']} ({severity_counts['Normal']/len(scores)*100:.1f}%)
- Slight (11-30): {severity_counts['Slight']} ({severity_counts['Slight']/len(scores)*100:.1f}%)
- Moderate (31-60): {severity_counts['Moderate']} ({severity_counts['Moderate']/len(scores)*100:.1f}%)
- Significant (61-90): {severity_counts['Significant']} ({severity_counts['Significant']/len(scores)*100:.1f}%)
- Severe (91-100): {severity_counts['Severe']} ({severity_counts['Severe']/len(scores)*100:.1f}%)

STATISTICS:
- Mean Score: {np.mean(scores):.2f}
- Median Score: {np.median(scores):.2f}
- Max Score: {np.max(scores):.2f}
- Standard Deviation: {np.std(scores):.2f}

TOP ANOMALIES:
"""
    
    top_indices = np.argsort(scores)[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        report += f"{i:2d}. Row {idx:4d}: Score {scores[idx]:6.1f}\n"
    
    report += f"\n{'='*50}\nEnd of Report"
    
    return report

# Settings page
def settings_page():
    st.markdown("##  Application Settings")
    
    st.markdown("###  Theme & Appearance")
    
    col1, col2 = st.columns(2)
    with col1:
        theme_mode = st.selectbox(
            "Color Theme",
            ["Light", "Dark", "Auto"],
            index=0 if st.session_state.theme == 'light' else 1
        )
    
    with col2:
        chart_style = st.selectbox(
            "Chart Style",
            ["Modern", "Classic", "Minimal"],
            index=0
        )
    
    st.markdown("###  Notification Preferences")
    
    notifications_enabled = st.checkbox(
        "Enable notifications",
        value=st.session_state.notifications_enabled
    )
    
    if notifications_enabled:
        col1, col2 = st.columns(2)
        with col1:
            email_notifications = st.checkbox("Email alerts")
            sms_notifications = st.checkbox("SMS alerts", value=True)
        with col2:
            severe_only = st.checkbox("Severe anomalies only", value=True)
            real_time = st.checkbox("Real-time alerts")
    
    st.markdown("###  Performance & Processing")
    
    col1, col2 = st.columns(2)
    with col1:
        max_data_points = st.number_input(
            "Max data points",
            min_value=1000,
            max_value=100000,
            value=50000,
            step=1000
        )
    with col2:
        processing_threads = st.selectbox(
            "Processing threads",
            [1, 2, 4, 8],
            index=1
        )
    
    auto_refresh = st.checkbox(
        "Auto-refresh results",
        value=st.session_state.auto_refresh,
        help="Automatically refresh results every 30 seconds"
    )
    
    if st.button(" Save Settings", type="primary"):
        st.session_state.theme = theme_mode.lower()
        st.session_state.notifications_enabled = notifications_enabled
        st.session_state.auto_refresh = auto_refresh
        st.success("Settings saved successfully!")
    
    if st.button(" Reset to Defaults"):
        for key in ['theme', 'notifications_enabled', 'auto_refresh']:
            if key in st.session_state:
                del st.session_state[key]
        st.success(" Settings reset to defaults!")
        st.rerun()

# Main application flow
def main():
    load_custom_css()
    initialize_session_state()
    create_navigation()
    
    if st.session_state.current_page == 'Data Upload':
        display_hero_section()
    
    if st.session_state.current_page == 'Data Upload':
        data_upload_page()
    elif st.session_state.current_page == 'Configuration':
        configuration_page()
    elif st.session_state.current_page == 'Model Training':
        model_training_page()
    elif st.session_state.current_page == 'Results & Analytics':
        results_analytics_page()
    elif st.session_state.current_page == 'Settings':
        settings_page()
    
    if st.session_state.get('auto_refresh', False) and st.session_state.results_ready:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
