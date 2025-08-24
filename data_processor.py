"""
Data Processing Module for Anomaly Detection System
Handles data loading, validation, preprocessing, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
import io
import streamlit as st
from datetime import datetime
import re

class DataProcessor:
    """Handles all data processing operations for the anomaly detection system."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.feature_columns: Optional[List[str]] = None
        self.timestamp_column: Optional[str] = None
        self.scaler = None
        
    def load_data(self, uploaded_file: io.BytesIO) -> pd.DataFrame:
        """
        Load and perform initial validation of CSV data.
        
        Args:
            uploaded_file: Uploaded CSV file buffer
            
        Returns:
            pd.DataFrame: Loaded dataframe
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if df is None:
                raise ValueError("Unable to decode the CSV file. Please check the file encoding.")
            
            # Basic validation
            if df.empty:
                raise ValueError("The uploaded file is empty.")
                
            if len(df.columns) == 0:
                raise ValueError("No columns found in the CSV file.")
                
            # Check for minimum data requirements
            if len(df) < 100:
                st.warning("⚠️ Dataset contains fewer than 100 rows. Consider using more data for better results.")
                
            if len(df) > 10000:
                st.warning("⚠️ Dataset contains more than 10,000 rows. Processing may take longer.")
                
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Basic data type inference
            df = self._infer_data_types(df)
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    def _infer_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer and convert appropriate data types.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with corrected types
        """
        df_copy = df.copy()
        
        for col in df_copy.columns:
            # Try to convert to numeric
            if df_copy[col].dtype == 'object':
                # First check if it looks like a timestamp
                if self._is_timestamp_column(df_copy[col]):
                    try:
                        df_copy[col] = pd.to_datetime(df_copy[col], infer_datetime_format=True)
                    except:
                        pass  # Keep as object if conversion fails
                else:
                    # Try numeric conversion
                    numeric_converted = pd.to_numeric(df_copy[col], errors='coerce')
                    # If more than 50% of values can be converted, use numeric
                    if numeric_converted.notna().sum() / len(df_copy) > 0.5:
                        df_copy[col] = numeric_converted
                        
        return df_copy
    
    def _is_timestamp_column(self, series: pd.Series) -> bool:
        """
        Check if a series likely contains timestamp data.
        
        Args:
            series: Pandas series to check
            
        Returns:
            bool: True if likely a timestamp column
        """
        if series.dtype != 'object':
            return False
            
        # Sample a few non-null values
        sample_values = series.dropna().head(10)
        
        if len(sample_values) == 0:
            return False
            
        timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
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
        """
        Identify potential timestamp columns in the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            List[str]: List of potential timestamp column names
        """
        timestamp_cols = []
        
        for col in df.columns:
            # Check by name patterns
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['time', 'date', 'timestamp', 'ts', 'datetime']):
                timestamp_cols.append(col)
                continue
                
            # Check by data type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                timestamp_cols.append(col)
                continue
                
            # Check by content
            if self._is_timestamp_column(df[col]):
                timestamp_cols.append(col)
                
        return timestamp_cols
    
    def identify_numeric_columns(self, df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> List[str]:
        """
        Identify numeric columns suitable for anomaly detection.
        
        Args:
            df: Input dataframe
            exclude_columns: Columns to exclude from consideration
            
        Returns:
            List[str]: List of numeric column names
        """
        if exclude_columns is None:
            exclude_columns = []
            
        numeric_cols = []
        
        for col in df.columns:
            if col in exclude_columns:
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if column has sufficient variance
                if df[col].nunique() > 1 and df[col].std() > 0:
                    numeric_cols.append(col)
                    
        return numeric_cols
    
    def preprocess_data(self, 
                       df: pd.DataFrame, 
                       feature_columns: List[str],
                       timestamp_column: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess data for anomaly detection.
        
        Args:
            df: Input dataframe
            feature_columns: List of feature column names
            timestamp_column: Optional timestamp column name
            
        Returns:
            pd.DataFrame: Preprocessed feature data
            
        Raises:
            ValueError: If preprocessing fails
        """
        try:
            # Validate feature columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found in data: {missing_cols}")
            
            # Extract feature data
            feature_data = df[feature_columns].copy()
            
            # Handle missing values
            feature_data = self._handle_missing_values(feature_data)
            
            # Remove constant features
            feature_data = self._remove_constant_features(feature_data)
            
            # Validate we still have features
            if feature_data.shape[1] == 0:
                raise ValueError("No valid features remaining after preprocessing.")
            
            # Update feature columns list
            self.feature_columns = list(feature_data.columns)
            self.timestamp_column = timestamp_column
            
            # Check for sufficient data variation
            self._validate_data_quality(feature_data)
            
            return feature_data
            
        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_processed = df.copy()
        
        # Calculate missing value percentage for each column
        missing_percentages = (df_processed.isnull().sum() / len(df_processed)) * 100
        
        # Drop columns with more than 50% missing values
        cols_to_drop = missing_percentages[missing_percentages > 50].index.tolist()
        if cols_to_drop:
            st.warning(f"⚠️ Dropping columns with >50% missing values: {cols_to_drop}")
            df_processed = df_processed.drop(columns=cols_to_drop)
        
        # For remaining columns, use different strategies based on missing percentage
        for col in df_processed.columns:
            missing_pct = missing_percentages[col]
            
            if missing_pct > 0:
                if missing_pct <= 5:
                    # Forward fill for small amounts of missing data
                    df_processed[col] = df_processed[col].ffill().bfill()
                elif missing_pct <= 20:
                    # Use median for moderate missing data
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    # Use interpolation for larger amounts
                    df_processed[col] = df_processed[col].interpolate(method='linear').bfill()
        
        # Final check - drop any rows that still have missing values
        rows_before = len(df_processed)
        df_processed = df_processed.dropna()
        rows_after = len(df_processed)
        
        if rows_before != rows_after:
            st.info(f"ℹ️ Removed {rows_before - rows_after} rows with remaining missing values.")
        
        return df_processed
    
    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with zero or near-zero variance.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with constant features removed
        """
        df_processed = df.copy()
        
        # Calculate variance for each column
        variances = df_processed.var()
        
        # Remove columns with zero or near-zero variance
        constant_cols = variances[variances < 1e-10].index.tolist()
        
        if constant_cols:
            st.info(f"ℹ️ Removing constant features: {constant_cols}")
            df_processed = df_processed.drop(columns=constant_cols)
        
        return df_processed
    
    def _validate_data_quality(self, df: pd.DataFrame) -> None:
        """
        Validate data quality for anomaly detection.
        
        Args:
            df: Preprocessed feature dataframe
            
        Raises:
            ValueError: If data quality is insufficient
        """
        if len(df) < 50:
            raise ValueError("Insufficient data: At least 50 rows required for anomaly detection.")
        
        if df.shape[1] == 0:
            raise ValueError("No valid features available for anomaly detection.")
        
        # Check for infinite values
        if np.isinf(df.values).any():
            raise ValueError("Dataset contains infinite values. Please clean your data.")
        
        # Check for extremely large values that might cause numerical issues
        max_values = df.max()
        if (max_values > 1e10).any():
            st.warning("⚠️ Some features have very large values. Consider normalizing your data.")
        
        # Check for sufficient variation in the data
        total_variance = df.var().sum()
        if total_variance < 1e-6:
            st.warning("⚠️ Data has very low variance. Anomaly detection may not be meaningful.")
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive statistics for features.
        
        Args:
            df: Feature dataframe
            
        Returns:
            Dict: Dictionary containing feature statistics
        """
        stats = {
            'n_features': df.shape[1],
            'n_samples': df.shape[0],
            'feature_names': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'statistics': df.describe().to_dict()
        }
        
        return stats
