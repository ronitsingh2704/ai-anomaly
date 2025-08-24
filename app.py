"""
Enhanced Streamlit-based Anomaly Detection System for Multivariate Time Series Data
Modern UI with improved layout, navigation, styling, and user experience.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from typing import Dict, List, Tuple, Optional
import traceback
from datetime import datetime, timedelta
import time
import json

from data_processor import DataProcessor
from anomaly_models import AnomalyDetector
from feature_attribution import FeatureAttributor
from utils import validate_training_period, format_anomaly_score

# Enhanced page configuration
st.set_page_config(
    page_title="AI Anomaly Detection Platform",
    page_icon="🚀",
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
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2c3e50, #34495e);
            color: white;
        }
        
        .nav-pill {
            background: rgba(255, 255, 255, 0.1);
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            margin: 0.5rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .nav-pill:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateX(5px);
        }
        
        .nav-pill.active {
            background: linear-gradient(45deg, #3498db, #2980b9);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        .data-upload-zone {
            border: 3px dashed #bdc3c7;
            border-radius: 15px;
            padding: 3rem;
            text-align: center;
            background: rgba(236, 240, 241, 0.3);
            transition: all 0.3s ease;
        }
        
        .data-upload-zone:hover {
            border-color: #3498db;
            background: rgba(52, 152, 219, 0.1);
        }
        
        /* Animated loading */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Dark mode toggle styles */
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: rgba(0,0,0,0.7);
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .theme-toggle:hover {
            background: rgba(0,0,0,0.9);
            transform: scale(1.1);
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
            <h2 style="color: white; margin: 0;">🚀 AI Anomaly</h2>
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
    
    # Add separator
    st.sidebar.markdown("---")
    
    # Quick stats if data is loaded
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
            
            # Custom metric with color
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
    
    # Upload zone
    uploaded_file = st.file_uploader(
        "",
        type=['csv'],
        help="Upload your multivariate time series CSV file",
        label_visibility="collapsed"
    )
    
    # Instructions
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
            with st.spinner("🔄 Loading and validating data..."):
                time.sleep(0.5)  # Brief pause for UX
                processor = DataProcessor()
                data = processor.load_data(uploaded_file)
                
            # Success message
            st.markdown(f"""
                <div class="success-card">
                    <h4> Data Loaded Successfully!</h4>
                    <p><strong>Shape:</strong> {data.shape[0]:,} rows × {data.shape[1]} columns</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Data preview tabs
            tab1, tab2, tab3 = st.tabs(["Data Preview", " Statistics", " Data Types"])
            
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
            
            # Store data and move to next step
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
    
    # Feature selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Timestamp Column")
        timestamp_cols = ['Time'] if 'Time' in data.columns else []
        if timestamp_cols:
            timestamp_col = st.selectbox(
                "Select timestamp column",
                timestamp_cols,
                help="Column containing time information"
            )
        else:
            st.info(" No timestamp column detected. Using row index.")
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
    
    # Normal period selection
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
            st.info(f"📈 Training period: {training_days} days")
            
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
            st.info(f"Training samples: {training_rows}")
    
    # Algorithm configuration
    st.markdown("###  Algorithm Configuration")
    
    algorithm = st.selectbox(
        "Select ML Algorithm",
        ["PCA-based", "Isolation Forest", "Autoencoder"],
        help="Choose the anomaly detection algorithm"
    )
    
    # Algorithm-specific parameters in columns
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
    
    # Store configuration and proceed
    if st.button(" Save Configuration & Proceed", type="primary", use_container_width=True):
        try:
            processor = DataProcessor()
            processed_data = processor.preprocess_data(
                data,
                feature_columns=selected_features,
                timestamp_column=timestamp_col
            )
            
            # Store configuration
            st.session_state.processed_data = processed_data
            st.session_state.selected_features = selected_features
            st.session_state.timestamp_col = timestamp_col
            st.session_state.algorithm = algorithm
            st.session_state.normal_period = {
                'start': normal_start if timestamp_col else normal_start_idx,
                'end': normal_end if timestamp_col else normal_end_idx
            }
            
            # Store algorithm parameters
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
    
    # Training summary
    with st.expander(" Training Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algorithm", st.session_state.algorithm)
        with col2:
            st.metric("Features", len(st.session_state.selected_features))
        with col3:
            st.metric("Data Points", len(st.session_state.processed_data))
    
    # Training button
    if st.button(" Start Training & Detection", type="primary", use_container_width=True):
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize detector
            status_text.text(" Initializing ML model...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            detector = AnomalyDetector(
                algorithm=st.session_state.algorithm.lower().replace('-', '_').replace(' ', '_'),
                **st.session_state.algo_params
            )
            
            # Step 2: Prepare training data
            status_text.text(" Preparing training data...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            # Filter training data based on normal period
            if st.session_state.timestamp_col:
                # Timestamp-based filtering
                timestamp_series = pd.to_datetime(st.session_state.raw_data[st.session_state.timestamp_col])
                normal_start_dt = pd.to_datetime(st.session_state.normal_period['start'])
                normal_end_dt = pd.to_datetime(st.session_state.normal_period['end'])
                normal_mask = (timestamp_series >= normal_start_dt) & (timestamp_series <= normal_end_dt)
                normal_data = st.session_state.processed_data[normal_mask]
            else:
                # Index-based filtering
                start_idx = st.session_state.normal_period['start']
                end_idx = st.session_state.normal_period['end']
                normal_data = st.session_state.processed_data.iloc[start_idx:end_idx]
            
            if len(normal_data) == 0:
                st.error(" No data found in the specified normal period.")
                return
            
            # Step 3: Train model
            status_text.text(" Training ML model...")
            progress_bar.progress(50)
            
            detector.fit(normal_data)
            
            # Step 4: Predict anomalies
            status_text.text(" Detecting anomalies...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            anomaly_scores = detector.predict(st.session_state.processed_data)
            
            # Step 5: Calculate feature attributions
            status_text.text(" Calculating feature attributions...")
            progress_bar.progress(90)
            time.sleep(0.5)
            
            attributor = FeatureAttributor(detector, st.session_state.selected_features)
            feature_attributions = attributor.calculate_attributions(
                st.session_state.processed_data, 
                anomaly_scores
            )
            
            # Prepare results
            results_df = st.session_state.raw_data.copy()
            results_df['Abnormality_score'] = [format_anomaly_score(score) for score in anomaly_scores]
            
            # Add top contributing features
            for i in range(7):
                col_name = f'top_feature_{i+1}'
                results_df[col_name] = [
                    attrs[i] if i < len(attrs) else '' 
                    for attrs in feature_attributions
                ]
            
            # Store results
            st.session_state.results_ready = True
            st.session_state.results_df = results_df
            st.session_state.anomaly_scores = anomaly_scores
            st.session_state.feature_attributions = feature_attributions
            st.session_state.model_trained = True
            
            # Complete
            progress_bar.progress(100)
            status_text.text(" Training completed successfully!")
            
            # Success message
            time.sleep(1)
            st.success(" Anomaly detection completed! Proceeding to results...")
            st.session_state.current_page = 'Results & Analytics'
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f" Training failed: {str(e)}")
            with st.expander(" Debug Information"):
                st.code(traceback.format_exc())

# Enhanced results page
def results_analytics_page():
    if not st.session_state.results_ready:
        st.warning(" No results available. Please complete model training first.")
        return
    
    st.markdown("##  Results & Analytics Dashboard")
    
    scores = st.session_state.anomaly_scores
    
    # Enhanced metrics display
    display_metric_cards(scores)
    
    # SMS Alert Section
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
                    ACCOUNT_SID = "your_twilio_sid_here"
                    AUTH_TOKEN = "your_auth_token_here"
                    TWILIO_NUMBER = ""
                    
                    client = Client(ACCOUNT_SID, AUTH_TOKEN)
                    message = f" ALERT: {severe_anomalies} severe anomalies detected in your data!"
                    client.messages.create(
                        body=message,
                        from_=TWILIO_NUMBER,
                        to=user_number
                    )
                    st.success(f" SMS alert sent to {user_number}!")
                except Exception as e:
                    st.error(f" SMS failed: {e}")
            elif severe_anomalies == 0:
                st.info(" No severe anomalies detected.")
            else:
                st.warning(" Please enter a valid phone number.")
    
    # Interactive visualizations
    st.markdown("---")
    st.markdown("### Interactive Visualizations")
    
    # Anomaly timeline with enhanced styling
    fig_timeline = go.Figure()
    
    # Color mapping for severity
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
    
    # Add severity thresholds
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
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity pie chart
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
        # Score histogram
        fig_hist = px.histogram(
            x=scores,
            nbins=30,
            title="Score Distribution",
            labels={'x': 'Anomaly Score', 'y': 'Frequency'}
        )
        fig_hist.update_traces(marker_color='#3498db')
        fig_hist.update_layout(template='plotly_white')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Top anomalies
    st.markdown("###  Top Anomalies Analysis")
    
    top_indices = np.argsort(scores)[-15:][::-1]
    top_scores = [scores[i] for i in top_indices]
    
    # Enhanced top anomalies chart
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
    
    # Feature importance analysis
    if hasattr(st.session_state, 'feature_attributions'):
        st.markdown("###  Feature Contribution Analysis")
        
        # Global feature importance
        all_features = []
        for attrs in st.session_state.feature_attributions:
            for feat in attrs[:3]:  # Top 3 features
                if feat and feat.strip():
                    all_features.append(feat)
        
        if all_features:
            from collections import Counter
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
    
    # Interactive data table
    st.markdown("###  Detailed Results Table")
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_score = st.slider("Minimum Score", 0, 100, 0)
    with filter_col2:
        max_score = st.slider("Maximum Score", 0, 100, 100)
    with filter_col3:
        show_rows = st.selectbox("Show rows", [50, 100, 500, "All"])
    
    # Filter data
    filtered_df = st.session_state.results_df[
        (st.session_state.results_df['Abnormality_score'] >= min_score) &
        (st.session_state.results_df['Abnormality_score'] <= max_score)
    ].sort_values('Abnormality_score', ascending=False)
    
    if show_rows != "All":
        filtered_df = filtered_df.head(show_rows)
    
    # Enhanced table display
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )
    
    # Download section
    st.markdown("###  Export Results")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        # Enhanced CSV export
        csv_buffer = io.StringIO()
        st.session_state.results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Enhanced CSV",
            data=csv_data,
            file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with download_col2:
        # Summary report
        summary = create_summary_report(scores)
        st.download_button(
            label=" Download Summary Report",
            data=summary,
            file_name=f"anomaly_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Settings page
def settings_page():
    st.markdown("##  Application Settings")
    
    # Theme settings
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
    
    # Notification settings
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
    
    # Performance settings
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
    
    # Auto-refresh
    auto_refresh = st.checkbox(
        "Auto-refresh results",
        value=st.session_state.auto_refresh,
        help="Automatically refresh results every 30 seconds"
    )
    
    # Save settings
    if st.button(" Save Settings", type="primary"):
        st.session_state.theme = theme_mode.lower()
        st.session_state.notifications_enabled = notifications_enabled
        st.session_state.auto_refresh = auto_refresh
        st.success(" Settings saved successfully!")
    
    # Reset settings
    if st.button(" Reset to Defaults"):
        for key in ['theme', 'notifications_enabled', 'auto_refresh']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Settings reset to defaults!")
        st.rerun()

# Helper function for summary report
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

# Main application flow
def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Create navigation
    create_navigation()
    
    # Display hero section on first page
    if st.session_state.current_page == 'Data Upload':
        display_hero_section()
    
    # Route to appropriate page
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
    
    # Auto-refresh functionality
    if st.session_state.get('auto_refresh', False) and st.session_state.results_ready:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()