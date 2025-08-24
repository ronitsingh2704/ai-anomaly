"""
Feature Attribution Module for Anomaly Detection
Identifies the most contributing features for each detected anomaly.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from anomaly_models import AnomalyDetector
import streamlit as st

class FeatureAttributor:
    """Handles feature attribution for anomaly detection results."""
    
    def __init__(self, detector: AnomalyDetector, feature_names: List[str]):
        """
        Initialize feature attributor.
        
        Args:
            detector: Trained anomaly detector
            feature_names: List of feature names
        """
        self.detector = detector
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
    def calculate_attributions(self, 
                             X: pd.DataFrame, 
                             anomaly_scores: np.ndarray) -> List[List[str]]:
        """
        Calculate feature attributions for each data point.
        
        Args:
            X: Feature data
            anomaly_scores: Anomaly scores for each data point
            
        Returns:
            List[List[str]]: List of top contributing features for each data point
        """
        attributions = []
        
        # Use progress bar for large datasets
        if len(X) > 1000:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        for i in range(len(X)):
            if len(X) > 1000 and i % 100 == 0:
                progress = i / len(X)
                progress_bar.progress(progress)
                progress_text.text(f"Calculating feature attributions: {i}/{len(X)}")
            
            # Get feature attribution for this sample
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
        """
        Get feature attribution for a single sample.
        
        Args:
            sample: Single sample dataframe
            anomaly_score: Anomaly score for this sample
            
        Returns:
            List[str]: Top contributing feature names (up to 7)
        """
        if anomaly_score <= 10:  # Normal data
            return [''] * 7  # Return empty strings for normal data
        
        # Method 1: Use model-specific feature importance
        try:
            feature_importance = self.detector.get_feature_importance(sample)
            
            # Combine with sample values for better attribution
            sample_values = np.abs(sample.values[0])
            
            # Normalize sample values
            if np.max(sample_values) > 0:
                sample_values_norm = sample_values / np.max(sample_values)
            else:
                sample_values_norm = sample_values
            
            # Combined score: feature importance weighted by sample magnitude
            combined_scores = feature_importance * (1 + sample_values_norm)
            
        except Exception:
            # Fallback: Use value-based attribution
            combined_scores = self._value_based_attribution(sample)
        
        # Get top features
        top_indices = np.argsort(combined_scores)[::-1]
        top_features = [self.feature_names[i] for i in top_indices[:7]]
        
        # Pad with empty strings if needed
        while len(top_features) < 7:
            top_features.append('')
        
        return top_features
    
    def _value_based_attribution(self, sample: pd.DataFrame) -> np.ndarray:
        """
        Calculate feature attribution based on value magnitudes and deviations.
        
        Args:
            sample: Single sample dataframe
            
        Returns:
            np.ndarray: Attribution scores for each feature
        """
        sample_values = sample.values[0]
        
        # Method 1: Absolute values (magnitude-based)
        magnitude_scores = np.abs(sample_values)
        
        # Method 2: Z-score based (if we have training statistics)
        try:
            # Try to get feature statistics from the detector
            if hasattr(self.detector.detector, 'scaler'):
                scaler = self.detector.detector.scaler
                if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                    # Calculate z-scores
                    z_scores = np.abs((sample_values - scaler.mean_) / scaler.scale_)
                    combined_scores = magnitude_scores * (1 + z_scores)
                else:
                    combined_scores = magnitude_scores
            else:
                combined_scores = magnitude_scores
        except Exception:
            combined_scores = magnitude_scores
        
        # Normalize scores
        if np.sum(combined_scores) > 0:
            combined_scores = combined_scores / np.sum(combined_scores)
        
        return combined_scores
    
    def _perturbation_based_attribution(self, 
                                      sample: pd.DataFrame, 
                                      original_score: float) -> np.ndarray:
        """
        Calculate feature attribution using perturbation analysis.
        Note: This is computationally expensive and used as fallback.
        
        Args:
            sample: Single sample dataframe
            original_score: Original anomaly score
            
        Returns:
            np.ndarray: Attribution scores for each feature
        """
        attributions = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            # Create perturbed sample (set feature to median/mean)
            perturbed_sample = sample.copy()
            
            # Use zero as neutral value (assuming data is scaled)
            perturbed_sample.iloc[0, i] = 0.0
            
            try:
                # Get score for perturbed sample
                perturbed_score = self.detector.predict(perturbed_sample)[0]
                
                # Attribution is the difference in scores
                attributions[i] = abs(original_score - perturbed_score)
                
            except Exception:
                # If perturbation fails, use value magnitude
                attributions[i] = abs(sample.iloc[0, i])
        
        # Normalize
        if np.sum(attributions) > 0:
            attributions = attributions / np.sum(attributions)
        
        return attributions
    
    def get_global_feature_importance(self, 
                                    X: pd.DataFrame, 
                                    anomaly_scores: np.ndarray,
                                    threshold: float = 30) -> Dict[str, float]:
        """
        Calculate global feature importance across all anomalies.
        
        Args:
            X: Feature data
            anomaly_scores: Anomaly scores
            threshold: Score threshold to consider as anomaly
            
        Returns:
            Dict[str, float]: Global feature importance scores
        """
        # Filter to anomalous samples only
        anomaly_mask = anomaly_scores > threshold
        anomalous_samples = X[anomaly_mask]
        
        if len(anomalous_samples) == 0:
            return {feature: 0.0 for feature in self.feature_names}
        
        # Calculate average feature importance for anomalous samples
        try:
            global_importance = self.detector.get_feature_importance(anomalous_samples)
        except Exception:
            # Fallback: use value-based importance
            global_importance = np.mean(np.abs(anomalous_samples.values), axis=0)
            global_importance = global_importance / np.sum(global_importance)
        
        # Convert to dictionary
        importance_dict = {
            feature: float(importance) 
            for feature, importance in zip(self.feature_names, global_importance)
        }
        
        return importance_dict
    
    def get_feature_contribution_summary(self, 
                                       attributions: List[List[str]], 
                                       anomaly_scores: np.ndarray) -> pd.DataFrame:
        """
        Create a summary of feature contributions across all anomalies.
        
        Args:
            attributions: Feature attributions for each sample
            anomaly_scores: Anomaly scores
            
        Returns:
            pd.DataFrame: Summary of feature contributions
        """
        feature_counts = {feature: 0 for feature in self.feature_names}
        feature_ranks = {feature: [] for feature in self.feature_names}
        
        # Count feature appearances and track ranks
        for i, (attrs, score) in enumerate(zip(attributions, anomaly_scores)):
            if score > 10:  # Only consider anomalous samples
                for rank, feature in enumerate(attrs):
                    if feature and feature in feature_counts:
                        feature_counts[feature] += 1
                        feature_ranks[feature].append(rank + 1)  # 1-indexed rank
        
        # Calculate average ranks
        avg_ranks = {}
        for feature in self.feature_names:
            if feature_ranks[feature]:
                avg_ranks[feature] = np.mean(feature_ranks[feature])
            else:
                avg_ranks[feature] = 8  # Worst possible rank + 1
        
        # Create summary dataframe
        summary_df = pd.DataFrame({
            'Feature': list(feature_counts.keys()),
            'Anomaly_Count': list(feature_counts.values()),
            'Average_Rank': [avg_ranks[f] for f in feature_counts.keys()],
            'Contribution_Rate': [
                count / max(1, len([s for s in anomaly_scores if s > 10])) 
                for count in feature_counts.values()
            ]
        })
        
        # Sort by contribution rate and average rank
        summary_df = summary_df.sort_values(
            ['Contribution_Rate', 'Average_Rank'], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        return summary_df
