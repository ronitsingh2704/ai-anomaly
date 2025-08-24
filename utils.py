"""
Utility Functions for Anomaly Detection System
Contains helper functions for validation, formatting, and common operations.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import re

def validate_training_period(start_time: Union[datetime, int], 
                           end_time: Union[datetime, int],
                           timestamp_column: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate that the training period meets minimum requirements.
    
    Args:
        start_time: Start of training period
        end_time: End of training period
        timestamp_column: Name of timestamp column (if any)
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        if timestamp_column:
            # Working with actual timestamps
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)
                
            duration = end_time - start_time
            
            if duration.total_seconds() < 0:
                return False, "End time must be after start time"
            
            if duration.total_seconds() < 72 * 3600:  # 72 hours
                return False, "Training period should be at least 72 hours for reliable results"
                
        else:
            # Working with row indices
            if end_time <= start_time:
                return False, "End index must be greater than start index"
            
            if end_time - start_time < 100:
                return False, "Training period should contain at least 100 data points"
                
        return True, ""
        
    except Exception as e:
        return False, f"Error validating training period: {str(e)}"

def format_anomaly_score(score: float) -> int:
    """
    Format anomaly score to integer between 0-100.
    
    Args:
        score: Raw anomaly score
        
    Returns:
        int: Formatted score (0-100)
    """
    # Ensure score is numeric
    if pd.isna(score) or np.isinf(score):
        return 0
    
    # Clamp to 0-100 range and round to integer
    formatted_score = max(0, min(100, round(float(score))))
    
    return formatted_score

def get_severity_label(score: int) -> str:
    """
    Get severity label for anomaly score.
    
    Args:
        score: Anomaly score (0-100)
        
    Returns:
        str: Severity label
    """
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
    """
    Get color code for anomaly score visualization.
    
    Args:
        score: Anomaly score (0-100)
        
    Returns:
        str: Color code
    """
    if score <= 10:
        return "#2ECC71"  # Green
    elif score <= 30:
        return "#F1C40F"  # Yellow
    elif score <= 60:
        return "#E67E22"  # Orange
    elif score <= 90:
        return "#E74C3C"  # Red
    else:
        return "#8E44AD"  # Purple

def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str, Dict]:
    """
    Validate CSV structure for anomaly detection.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple[bool, str, Dict]: (is_valid, error_message, validation_info)
    """
    validation_info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': 0,
        'timestamp_columns': 0,
        'missing_data_percentage': 0,
        'constant_columns': 0
    }
    
    try:
        # Check basic structure
        if df.empty:
            return False, "CSV file is empty", validation_info
        
        if len(df.columns) == 0:
            return False, "No columns found in CSV", validation_info
        
        # Count column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        validation_info['numeric_columns'] = len(numeric_cols)
        
        # Check for timestamp columns
        timestamp_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'time' in col.lower() or 'date' in col.lower():
                timestamp_cols.append(col)
        validation_info['timestamp_columns'] = len(timestamp_cols)
        
        # Check missing data
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        validation_info['missing_data_percentage'] = (missing_cells / total_cells) * 100
        
        # Check for constant columns
        constant_cols = []
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                constant_cols.append(col)
        validation_info['constant_columns'] = len(constant_cols)
        
        # Validation checks
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

def create_sample_data(n_rows: int = 1000, n_features: int = 5, anomaly_rate: float = 0.1) -> pd.DataFrame:
    """
    Create sample time series data for testing (only if explicitly requested).
    
    Args:
        n_rows: Number of rows
        n_features: Number of features
        anomaly_rate: Rate of anomalies to inject
        
    Returns:
        pd.DataFrame: Sample data
    """
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_rows)]
    
    # Generate normal data
    data = {}
    data['timestamp'] = timestamps
    
    for i in range(n_features):
        # Base signal with some trend and seasonality
        trend = np.linspace(0, 2, n_rows)
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_rows) / 24)  # Daily pattern
        noise = np.random.normal(0, 0.5, n_rows)
        
        signal = 50 + trend + seasonal + noise
        
        # Inject anomalies
        n_anomalies = int(n_rows * anomaly_rate)
        anomaly_indices = np.random.choice(n_rows, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Random anomaly type
            if np.random.random() > 0.5:
                signal[idx] += np.random.normal(20, 5)  # Spike
            else:
                signal[idx] -= np.random.normal(15, 3)  # Dip
        
        data[f'feature_{i+1}'] = signal
    
    return pd.DataFrame(data)

def export_results_summary(results_df: pd.DataFrame, 
                          anomaly_scores: np.ndarray,
                          feature_names: List[str]) -> str:
    """
    Create a text summary of anomaly detection results.
    
    Args:
        results_df: Results dataframe
        anomaly_scores: Array of anomaly scores
        feature_names: List of feature names
        
    Returns:
        str: Text summary
    """
    summary_lines = []
    summary_lines.append("=" * 50)
    summary_lines.append("ANOMALY DETECTION RESULTS SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append("")
    
    # Basic statistics
    summary_lines.append(f"Total data points: {len(results_df)}")
    summary_lines.append(f"Features analyzed: {len(feature_names)}")
    summary_lines.append("")
    
    # Severity distribution
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
    
    # Top anomalies
    top_anomaly_indices = np.argsort(anomaly_scores)[-10:][::-1]
    summary_lines.append("TOP 10 ANOMALIES:")
    for i, idx in enumerate(top_anomaly_indices, 1):
        score = anomaly_scores[idx]
        severity = get_severity_label(score)
        summary_lines.append(f"  {i:2d}. Row {idx:4d}: Score {score:3.0f} ({severity})")
    
    summary_lines.append("")
    
    # Feature contribution summary
    all_features = []
    for _, row in results_df.iterrows():
        if row['Abnormality_score'] > 10:  # Only anomalous samples
            for i in range(1, 8):
                feature = row[f'top_feature_{i}']
                if feature and feature.strip():
                    all_features.append(feature)
    
    if all_features:
        from collections import Counter
        feature_counts = Counter(all_features)
        summary_lines.append("MOST CONTRIBUTING FEATURES:")
        for feature, count in feature_counts.most_common(10):
            percentage = (count / len([s for s in anomaly_scores if s > 10])) * 100
            summary_lines.append(f"  {feature}: {count} times ({percentage:.1f}%)")
    
    summary_lines.append("")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 50)
    
    return "\n".join(summary_lines)

def validate_feature_names(feature_names: List[str]) -> Tuple[bool, str]:
    """
    Validate feature names for safety and consistency.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not feature_names:
        return False, "No feature names provided"
    
    # Check for duplicates
    if len(feature_names) != len(set(feature_names)):
        return False, "Duplicate feature names found"
    
    # Check for empty or whitespace-only names
    for name in feature_names:
        if not name or not name.strip():
            return False, "Empty or whitespace-only feature names found"
    
    # Check for problematic characters
    for name in feature_names:
        if not re.match(r'^[a-zA-Z0-9_\-\.\s]+$', name):
            return False, f"Invalid characters in feature name: {name}"
    
    return True, ""

def memory_usage_check(df: pd.DataFrame, max_mb: int = 500) -> Tuple[bool, float]:
    """
    Check if dataframe memory usage is within acceptable limits.
    
    Args:
        df: Dataframe to check
        max_mb: Maximum memory usage in MB
        
    Returns:
        Tuple[bool, float]: (is_acceptable, memory_usage_mb)
    """
    memory_usage = df.memory_usage(deep=True).sum()
    memory_mb = memory_usage / (1024 * 1024)  # Convert to MB
    
    return memory_mb <= max_mb, memory_mb
