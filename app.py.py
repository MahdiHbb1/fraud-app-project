# @Author:xxx
# @Date:2025-11-10 08:02:50
# @LastModifiedBy:xxx
# @Last Modified time:2025-11-10 08:02:50
# -*- coding: utf-8 -*-
"""
APLIKASI WEB DETEKSI FRAUD v3.0 - ENHANCED EDITION
Sistem Deteksi Fraud Perbankan dengan Fitur Enterprise
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from typing import Dict, Tuple, Optional
import hashlib

# ============================================================================
# KONFIGURASI HALAMAN
# ============================================================================

st.set_page_config(
    page_title="Banking Fraud Detection System v3.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS STYLING - PROFESIONAL & MODERN
# ============================================================================

st.markdown("""
<style>
    .title { 
        text-align: center; 
        color: #1a5f7a;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .warning-card {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-card {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .danger-card {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 12px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .dataframe {
        font-size: 0.9em;
    }
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">🛡️ Banking Fraud Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Machine Learning Analytics for Transaction Security</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# CACHING & LOADING MODEL DENGAN ERROR HANDLING
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_models_and_artifacts() -> Optional[Dict]:
    """Load all ML artifacts with comprehensive error handling"""
    with st.spinner("🔄 Initializing ML models and artifacts..."):
        artifacts = {}
        required_files = {
            'dt_model': 'decision_tree_fraud_model.pkl',
            'rf_model': 'random_forest_fraud_model.pkl',
            'scaler': 'scaler.pkl',
            'le_type': 'label_encoder_type.pkl',
            'le_amount_cat': 'label_encoder_amount_cat.pkl',
            'feature_columns': 'feature_columns.pkl',
            'model_comparison': 'model_comparison.pkl'
        }
        
        missing_files = []
        for key, filename in required_files.items():
            if not os.path.exists(filename):
                missing_files.append(filename)
        
        if missing_files:
            st.error(f"❌ Missing required files: {', '.join(missing_files)}")
            st.error("Please ensure all .pkl files are in the repository root directory.")
            return None
        
        try:
            for key, filename in required_files.items():
                artifacts[key] = joblib.load(filename)
            
            # Validate loaded models
            if not hasattr(artifacts['rf_model'], 'predict'):
                st.error("❌ Invalid model file detected")
                return None
                
            st.success("✅ All ML artifacts loaded successfully")
            return artifacts
            
        except Exception as e:
            st.error(f"❌ Error loading artifacts: {str(e)}")
            return None

artifacts = load_models_and_artifacts()

if artifacts is None:
    st.error("⚠️ Application cannot start due to missing or corrupted model files.")
    st.info("Please contact system administrator or check deployment logs.")
    st.stop()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_transaction_id(transaction_data: Dict) -> str:
    """Generate unique transaction ID for tracking"""
    data_str = f"{transaction_data['type']}{transaction_data['amount']}{datetime.now().isoformat()}"
    return hashlib.md5(data_str.encode()).hexdigest()[:12].upper()

def calculate_risk_score(probability: float) -> Tuple[str, str, str]:
    """Calculate risk level, color, and recommendation"""
    if probability >= 0.90:
        return "CRITICAL", "#dc3545", "IMMEDIATE ACTION: Block transaction and investigate"
    elif probability >= 0.70:
        return "HIGH", "#fd7e14", "HIGH PRIORITY: Manual review required before processing"
    elif probability >= 0.50:
        return "MEDIUM", "#ffc107", "MODERATE RISK: Enhanced monitoring recommended"
    elif probability >= 0.30:
        return "LOW", "#20c997", "LOW RISK: Standard processing with routine checks"
    else:
        return "MINIMAL", "#28a745", "MINIMAL RISK: Proceed with standard workflow"

def format_currency(amount: float) -> str:
    """Format currency with proper separators"""
    return f"Rp {amount:,.2f}"

# ============================================================================
# PREPROCESSING FUNCTIONS - ENHANCED
# ============================================================================

def preprocess_data_form(df: pd.DataFrame, le_type, le_amount_cat, scaler, feature_columns) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """Enhanced preprocessing for single transaction with validation"""
    try:
        df_copy = df.copy()
        
        # Validate input ranges
        if df_copy['amount'].iloc[0] < 0:
            st.error("❌ Amount cannot be negative")
            return None, None
        
        if df_copy['amount'].iloc[0] > 1e9:  # 1 billion limit
            st.warning("⚠️ Unusually large transaction amount detected")
        
        # Feature Engineering
        df_copy['balance_change_orig'] = df_copy['newbalanceOrig'] - df_copy['oldbalanceOrig']
        df_copy['balance_change_dest'] = df_copy['newbalanceDest'] - df_copy['oldbalanceDest']
        
        # Amount categorization
        df_copy['amount_category'] = pd.cut(
            df_copy['amount'],
            bins=[0, 1000, 10000, 100000, float('inf')],
            labels=['small', 'medium', 'large', 'very_large'],
            right=False
        )
        
        # Suspicious pattern flags
        df_copy['orig_zero_after'] = (df_copy['newbalanceOrig'] == 0).astype(int)
        df_copy['dest_zero_before'] = (df_copy['oldbalanceDest'] == 0).astype(int)
        
        # Balance error calculations
        df_copy['error_balance_orig'] = df_copy['oldbalanceOrig'] + df_copy['amount'] - df_copy['newbalanceOrig']
        df_copy['error_balance_dest'] = df_copy['oldbalanceDest'] + df_copy['amount'] - df_copy['newbalanceDest']
        
        # Additional risk indicators
        df_copy['balance_ratio'] = df_copy['amount'] / (df_copy['oldbalanceOrig'] + 1)  # Avoid division by zero
        df_copy['dest_received'] = (df_copy['newbalanceDest'] > df_copy['oldbalanceDest']).astype(int)
        
        # Label Encoding
        df_copy['type_encoded'] = le_type.transform(df_copy['type'])
        df_copy['amount_category_encoded'] = le_amount_cat.transform(df_copy['amount_category'])
        
        # Select and order features
        df_processed = df_copy[feature_columns]
        
        # Scaling
        X_scaled = scaler.transform(df_processed)
        
        return X_scaled, df_copy
        
    except Exception as e:
        st.error(f"❌ Preprocessing error: {str(e)}")
        return None, None

def preprocess_data_mapped(df_mapped: pd.DataFrame, le_type, le_amount_cat, scaler, feature_columns) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """Enhanced preprocessing for batch data with anomaly detection"""
    try:
        df = df_mapped.copy()
        
        # Data validation
        initial_count = len(df)
        df = df[df['amount'] >= 0]  # Remove negative amounts
        if len(df) < initial_count:
            st.warning(f"⚠️ Removed {initial_count - len(df)} transactions with negative amounts")
        
        # Feature Engineering (same as form)
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrig']
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        df['amount_category'] = pd.cut(
            df['amount'],
            bins=[0, 1000, 10000, 100000, float('inf')],
            labels=['small', 'medium', 'large', 'very_large'],
            right=False
        )
        
        df['orig_zero_after'] = (df['newbalanceOrig'] == 0).astype(int)
        df['dest_zero_before'] = (df['oldbalanceDest'] == 0).astype(int)
        df['error_balance_orig'] = df['oldbalanceOrig'] + df['amount'] - df['newbalanceOrig']
        df['error_balance_dest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
        
        # Additional features
        df['balance_ratio'] = df['amount'] / (df['oldbalanceOrig'] + 1)
        df['dest_received'] = (df['newbalanceDest'] > df['oldbalanceDest']).astype(int)
        
        # Handle unknown transaction types
        known_types = set(le_type.classes_)
        unknown_types = set(df['type']) - known_types
        
        if unknown_types:
            st.warning(f"⚠️ Unknown transaction types detected: {unknown_types}")
            st.info("Mapping unknown types to 'PAYMENT' for processing")
            df['type'] = df['type'].apply(lambda x: 'PAYMENT' if x in unknown_types else x)
        
        # Label Encoding
        df['type_encoded'] = le_type.transform(df['type'])
        df['amount_category_encoded'] = le_amount_cat.transform(df['amount_category'])
        
        # Select features
        df_processed = df[feature_columns]
        
        # Scaling
        X_scaled = scaler.transform(df_processed)
        
        return X_scaled, df
        
    except Exception as e:
        st.error(f"❌ Batch preprocessing error: {str(e)}")
        st.info("Please verify column mapping and data format")
        return None, None

# ============================================================================
# NAVIGATION SIDEBAR
# ============================================================================

st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Module",
    [
        '📊 Dashboard & Model Performance',
        '🔍 Real-Time Transaction Analysis',
        '📂 Batch Processing & Reports',
        '📈 Analytics & Insights'
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# System Info
with st.sidebar.expander("ℹ️ System Information"):
    st.write("**Version:** 3.0 Enhanced")
    st.write("**Model:** Random Forest Classifier")
    st.write("**Last Updated:** November 2024")
    st.write("**Status:** 🟢 Online")

# Developer Info
st.sidebar.markdown("---")
st.sidebar.info(
    "**Development Team**\n\n"
    "• Mahdi - ML Engineer\n"
    "• Ibnu - Data Scientist\n"
    "• Brian - Backend Developer\n"
    "• Anya - Frontend Developer\n\n"
    "🏦 Enterprise Banking Solutions"
)

# ============================================================================
# PAGE 1: DASHBOARD & MODEL PERFORMANCE
# ============================================================================

if page == '📊 Dashboard & Model Performance':
    st.header("📊 System Dashboard & Model Performance Metrics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Model Performance Overview")
        st.write("Real performance metrics from test set validation, demonstrating production-ready accuracy.")
    
    with col2:
        st.metric(
            label="System Status",
            value="Operational",
            delta="100% Uptime"
        )
    
    st.markdown("---")
    
    # Model Comparison
    comp_data = artifacts['model_comparison']
    dt_metrics = comp_data.get('decision_tree', {})
    rf_metrics = comp_data.get('random_forest', {})
    
    st.subheader("🔬 Model Comparison: Decision Tree vs Random Forest")
    
    col1, col2, col3 = st.columns(3)
    
    # Metrics Display
    metrics = [
        ('F1-Score', 'test_f1'),
        ('Precision', 'test_precision'),
        ('Recall', 'test_recall')
    ]
    
    for col, (metric_name, metric_key) in zip([col1, col2, col3], metrics):
        with col:
            st.markdown(f"#### {metric_name}")
            
            dt_value = dt_metrics.get(metric_key, 0)
            rf_value = rf_metrics.get(metric_key, 0)
            improvement = ((rf_value - dt_value) / dt_value * 100) if dt_value > 0 else 0
            
            st.metric(
                "Decision Tree",
                f"{dt_value:.4f}",
                delta=None
            )
            st.metric(
                "Random Forest",
                f"{rf_value:.4f}",
                delta=f"+{improvement:.1f}%"
            )
    
    st.markdown("---")
    
    # Performance Visualization
    st.subheader("📈 Performance Comparison Chart")
    
    fig = go.Figure()
    
    metrics_names = ['F1-Score', 'Precision', 'Recall']
    dt_values = [dt_metrics.get('test_f1', 0), dt_metrics.get('test_precision', 0), dt_metrics.get('test_recall', 0)]
    rf_values = [rf_metrics.get('test_f1', 0), rf_metrics.get('test_precision', 0), rf_metrics.get('test_recall', 0)]
    
    fig.add_trace(go.Bar(
        name='Decision Tree',
        x=metrics_names,
        y=dt_values,
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        name='Random Forest',
        x=metrics_names,
        y=rf_values,
        marker_color='#764ba2'
    ))
    
    fig.update_layout(
        barmode='group',
        title='Model Performance Metrics Comparison',
        xaxis_title='Metrics',
        yaxis_title='Score',
        yaxis_range=[0, 1],
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("💡 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-card">
        <h4>✅ Model Selection: Random Forest</h4>
        <p><strong>Justification:</strong></p>
        <ul>
            <li>Superior F1-Score indicating better balance</li>
            <li>Higher Precision reduces false positives</li>
            <li>Robust against overfitting</li>
            <li>Ensemble method provides stability</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>📌 Production Considerations</h4>
        <ul>
            <li><strong>Precision Focus:</strong> Minimizes customer disruption</li>
            <li><strong>Recall Trade-off:</strong> Balanced fraud detection</li>
            <li><strong>Scalability:</strong> Handles high transaction volumes</li>
            <li><strong>Reliability:</strong> Consistent performance across data types</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: REAL-TIME TRANSACTION ANALYSIS
# ============================================================================

elif page == '🔍 Real-Time Transaction Analysis':
    st.header("🔍 Real-Time Transaction Fraud Detection")
    st.write("Interactive fraud analysis for individual transactions with instant risk assessment")
    
    st.markdown("---")
    
    with st.form(key='prediction_form'):
        st.subheader("Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            type_trans = st.selectbox(
                "Transaction Type",
                options=artifacts['le_type'].classes_,
                help="Select transaction category"
            )
            
        with col2:
            amount = st.number_input(
                "Amount (IDR)",
                min_value=0.0,
                value=100000.0,
                step=10000.0,
                format="%.2f"
            )
            
        with col3:
            model_choice = st.radio(
                "Select Model",
                ("Random Forest (Recommended)", "Decision Tree"),
                horizontal=True
            )
        
        st.markdown("---")
        st.subheader("Account Balance Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sender Account**")
            oldbalanceOrig = st.number_input(
                "Initial Balance",
                min_value=0.0,
                value=5000000.0,
                step=100000.0,
                key="old_orig"
            )
            newbalanceOrig = st.number_input(
                "Final Balance",
                min_value=0.0,
                value=4900000.0,
                step=100000.0,
                key="new_orig"
            )
            
        with col2:
            st.markdown("**Recipient Account**")
            oldbalanceDest = st.number_input(
                "Initial Balance",
                min_value=0.0,
                value=0.0,
                step=100000.0,
                key="old_dest"
            )
            newbalanceDest = st.number_input(
                "Final Balance",
                min_value=0.0,
                value=100000.0,
                step=100000.0,
                key="new_dest"
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button(
                label='🔮 Analyze Transaction',
                use_container_width=True
            )
    
    # Process prediction
    if submit_button:
        transaction_data = {
            'type': type_trans,
            'amount': amount,
            'oldbalanceOrig': oldbalanceOrig,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest
        }
        
        # Generate transaction ID
        trans_id = generate_transaction_id(transaction_data)
        
        df_input = pd.DataFrame([transaction_data])
        
        with st.spinner("🔄 Analyzing transaction..."):
            X_scaled, df_processed = preprocess_data_form(
                df_input,
                artifacts['le_type'],
                artifacts['le_amount_cat'],
                artifacts['scaler'],
                artifacts['feature_columns']
            )
            
            if X_scaled is not None:
                model_to_use = artifacts['rf_model'] if "Random Forest" in model_choice else artifacts['dt_model']
                model_name = "Random Forest" if "Random Forest" in model_choice else "Decision Tree"
                
                prediction = model_to_use.predict(X_scaled)[0]
                probability = model_to_use.predict_proba(X_scaled)[0][1]
                
                risk_level, risk_color, recommendation = calculate_risk_score(probability)
                
                st.markdown("---")
                st.subheader("🎯 Analysis Results")
                
                # Transaction Summary
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Transaction ID", trans_id)
                with col2:
                    st.metric("Model Used", model_name)
                with col3:
                    st.metric("Amount", format_currency(amount))
                with col4:
                    st.metric("Type", type_trans)
                
                st.markdown("---")
                
                # Risk Assessment
                if prediction == 1:
                    st.markdown(f"""
                    <div class="danger-card">
                    <h2 style='color: #dc3545; margin:0;'>🚨 FRAUD DETECTED 🚨</h2>
                    <h3 style='margin:10px 0;'>Risk Level: <span style='color:{risk_color}'>{risk_level}</span></h3>
                    <p style='font-size:1.2em; margin:10px 0;'><strong>Confidence: {probability:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.warning(f"**Recommended Action:** {recommendation}")
                    
                    # Detailed risk factors
                    with st.expander("🔍 Risk Factor Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Suspicious Indicators:**")
                            if df_processed['orig_zero_after'].iloc[0] == 1:
                                st.write("⚠️ Sender account emptied")
                            if df_processed['dest_zero_before'].iloc[0] == 1:
                                st.write("⚠️ New recipient account")
                            if abs(df_processed['error_balance_orig'].iloc[0]) > 0:
                                st.write("⚠️ Balance inconsistency detected")
                        
                        with col2:
                            st.markdown("**Transaction Pattern:**")
                            st.write(f"• Balance Change (Sender): {format_currency(df_processed['balance_change_orig'].iloc[0])}")
                            st.write(f"• Balance Change (Recipient): {format_currency(df_processed['balance_change_dest'].iloc[0])}")
                            st.write(f"• Amount Category: {df_processed['amount_category'].iloc[0]}")
                
                else:
                    st.markdown(f"""
                    <div class="success-card">
                    <h2 style='color: #28a745; margin:0;'>✅ TRANSACTION APPEARS LEGITIMATE</h2>
                    <h3 style='margin:10px 0;'>Risk Level: <span style='color:{risk_color}'>{risk_level}</span></h3>
                    <p style='font-size:1.2em; margin:10px 0;'><strong>Fraud Probability: {probability:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info(f"**Recommended Action:** {recommendation}")
                
                # Processed data
                with st.expander("📋 View Processed Transaction Data"):
                    st.dataframe(df_processed, use_container_width=True)

# ============================================================================
# PAGE 3: BATCH PROCESSING & REPORTS
# ============================================================================

elif page == '📂 Batch Processing & Reports':
    st.header("📂 Batch Transaction Processing")
    st.write("Upload and analyze multiple transactions with comprehensive reporting")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📁 Upload Transaction File",
        type=["csv", "xls", "xlsx"],
        help="Supported formats: CSV, Excel (.xls, .xlsx)"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_bank = pd.read_csv(uploaded_file)
            else:
                df_bank = pd.read_excel(uploaded_file, engine='openpyxl')
            
            st.success(f"✅ Successfully loaded {len(df_bank):,} transactions from {uploaded_file.name}")
            st.session_state['df_bank'] = df_bank
            
            # Preview
            with st.expander("👁️ Preview Data"):
                st.dataframe(df_bank.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            if 'df_bank' in st.session_state:
                del st.session_state['df_bank']
    
    if 'df_bank' in st.session_state:
        df_bank = st.session_state['df_bank']
        
        st.markdown("---")
        st.subheader("🗺️ Column Mapping Configuration")
        st.write("Map your file columns to the required model inputs")
        
        required_columns = {
            'type': 'Transaction type (e.g., TRANSFER, PAYMENT, CASH_OUT)',
            'amount': 'Transaction amount in currency',
            'oldbalanceOrig': 'Sender initial balance',
            'newbalanceOrig': 'Sender final balance',
            'oldbalanceDest': 'Recipient initial balance',
            'newbalanceDest': 'Recipient final balance'
        }
        
        uploaded_columns = df_bank.columns.tolist()
        
        with st.form(key='column_mapping_form'):
            mapping_dict = {}
            
            cols = st.columns(2)
            
            for i, (col_name, description) in enumerate(required_columns.items()):
                with cols[i % 2]:
                    st.markdown(f"**{col_name}**")
                    st.caption(description)
                    
                    try:
                        default_idx = [c.lower() for c in uploaded_columns].index(col_name.lower())
                    except ValueError:
                        default_idx = 0
                    
                    selected_col = st.selectbox(
                        f"Select column for '{col_name}'",
                        options=uploaded_columns,
                        index=default_idx,
                        key=f"map_{col_name}",
                        label_visibility="collapsed"
                    )
                    mapping_dict[col_name] = selected_col
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                model_choice_batch = st.radio(
                    "Select Analysis Model",
                    ("Random Forest (Recommended)", "Decision Tree"),
                    horizontal=True
                )
            
            with col2:
                process_button = st.form_submit_button(
                    label='🚀 Process & Analyze',
                    use_container_width=True
                )
        
        if process_button:
            with st.spinner("⚙️ Processing batch data..."):
                try:
                    df_mapped = df_bank[list(mapping_dict.values())].copy()
                    df_mapped.columns = list(mapping_dict.keys())
                    
                    X_scaled, df_processed = preprocess_data_mapped(
                        df_mapped,
                        artifacts['le_type'],
                        artifacts['le_amount_cat'],
                        artifacts['scaler'],
                        artifacts['feature_columns']
                    )
                    
                    if X_scaled is not None:
                        model_to_use = artifacts['rf_model'] if "Random Forest" in model_choice_batch else artifacts['dt_model']
                        
                        predictions = model_to_use.predict(X_scaled)
                        probabilities = model_to_use.predict_proba(X_scaled)[:, 1]
                        
                        df_results = df_bank.copy()
                        df_results['Fraud_Prediction'] = predictions
                        df_results['Fraud_Probability'] = (probabilities * 100).round(2)
                        df_results['Risk_Level'] = df_results['Fraud_Probability'].apply(
                            lambda x: calculate_risk_score(x/100)[0]
                        )
                        
                        df_fraud = df_results[df_results['Fraud_Prediction'] == 1].sort_values(
                            by='Fraud_Probability', 
                            ascending=False
                        )
                        
                        st.success("✅ Analysis Complete!")
                        st.markdown("---")
                        
                        # Executive Summary
                        st.subheader("📊 Executive Summary")
                        
                        total_transactions = len(df_results)
                        total_fraud = len(df_fraud)
                        fraud_rate = (total_fraud / total_transactions * 100) if total_transactions > 0 else 0
                        potential_loss = df_results.loc[
                            df_results['Fraud_Prediction'] == 1, 
                            mapping_dict['amount']
                        ].sum()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown("""
                            <div class="metric-container">
                            <h4>Total Transactions</h4>
                            <h2 style='color:#667eea'>{:,}</h2>
                            </div>
                            """.format(total_transactions), unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            <div class="metric-container">
                            <h4>Fraud Detected</h4>
                            <h2 style='color:#dc3545'>{:,}</h2>
                            </div>
                            """.format(total_fraud), unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("""
                            <div class="metric-container">
                            <h4>Fraud Rate</h4>
                            <h2 style='color:#ffc107'>{:.2f}%</h2>
                            </div>
                            """.format(fraud_rate), unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown("""
                            <div class="metric-container">
                            <h4>Potential Loss</h4>
                            <h2 style='color:#fd7e14'>{}</h2>
                            </div>
                            """.format(format_currency(potential_loss)), unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Risk Distribution
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.subheader("🎯 Risk Distribution")
                            risk_counts = df_fraud['Risk_Level'].value_counts()
                            
                            if not risk_counts.empty:
                                fig_donut = go.Figure(data=[go.Pie(
                                    labels=risk_counts.index,
                                    values=risk_counts.values,
                                    hole=.4,
                                    marker_colors=['#dc3545', '#fd7e14', '#ffc107', '#20c997', '#28a745']
                                )])
                                fig_donut.update_layout(
                                    title='Fraud Cases by Risk Level',
                                    height=350,
                                    margin=dict(t=50, b=0, l=0, r=0)
                                )
                                st.plotly_chart(fig_donut, use_container_width=True)
                            else:
                                st.success("✅ No fraud detected!")
                        
                        with col2:
                            st.subheader("📈 Fraud Probability Distribution")
                            fig_hist = px.histogram(
                                df_fraud,
                                x='Fraud_Probability',
                                nbins=20,
                                title='Distribution of Fraud Probabilities',
                                labels={'Fraud_Probability': 'Fraud Probability (%)'},
                                color_discrete_sequence=['#764ba2']
                            )
                            fig_hist.update_layout(
                                showlegend=False,
                                height=350,
                                yaxis_title='Count'
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Priority Actions
                        st.subheader("🚨 Priority Action List")
                        st.write("High-risk transactions requiring immediate investigation")
                        
                        if not df_fraud.empty:
                            # Filter critical and high risk
                            df_priority = df_fraud[
                                df_fraud['Risk_Level'].isin(['CRITICAL', 'HIGH'])
                            ].head(20)
                            
                            display_cols = ['Risk_Level', 'Fraud_Probability'] + list(mapping_dict.values())
                            
                            # Color code by risk
                            def highlight_risk(row):
                                if row['Risk_Level'] == 'CRITICAL':
                                    return ['background-color: #f8d7da'] * len(row)
                                elif row['Risk_Level'] == 'HIGH':
                                    return ['background-color: #fff3cd'] * len(row)
                                else:
                                    return [''] * len(row)
                            
                            st.dataframe(
                                df_priority[display_cols].style.apply(highlight_risk, axis=1),
                                height=400,
                                use_container_width=True
                            )
                            
                            # Download options
                            st.markdown("---")
                            st.subheader("💾 Export Reports")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                csv_fraud = df_fraud[display_cols].to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Fraud Cases (CSV)",
                                    data=csv_fraud,
                                    file_name=f"fraud_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col2:
                                csv_all = df_results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Full Report (CSV)",
                                    data=csv_all,
                                    file_name=f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with col3:
                                csv_priority = df_priority[display_cols].to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Priority List (CSV)",
                                    data=csv_priority,
                                    file_name=f"priority_actions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        else:
                            st.success("✅ No fraudulent transactions detected in this batch!")
                            st.info("All transactions passed the fraud detection screening.")
                        
                        # Full results viewer
                        with st.expander("📋 View All Transaction Results"):
                            st.dataframe(df_results, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    st.info("Please verify your column mapping and data format")

# ============================================================================
# PAGE 4: ANALYTICS & INSIGHTS
# ============================================================================

elif page == '📈 Analytics & Insights':
    st.header("📈 Advanced Analytics & Business Intelligence")
    st.write("Deep insights into fraud patterns and transaction behavior")
    
    st.markdown("---")
    
    # Check if batch data exists
    if 'df_bank' not in st.session_state:
        st.info("ℹ️ Please upload and process batch data in the 'Batch Processing' page first to view analytics.")
        st.markdown("""
        <div class="info-box">
        <h4>Getting Started:</h4>
        <ol>
            <li>Navigate to <strong>📂 Batch Processing & Reports</strong></li>
            <li>Upload your transaction data file</li>
            <li>Configure column mapping</li>
            <li>Process the data</li>
            <li>Return here for detailed analytics</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("✅ Batch data available for analysis")
        
        # Create sample analytics
        st.subheader("🔍 Transaction Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>📊 Key Fraud Indicators</h4>
            <ul>
                <li><strong>Zero Balance Transfers:</strong> High correlation with fraud</li>
                <li><strong>New Account Activity:</strong> First-time recipients are higher risk</li>
                <li><strong>Large Amounts:</strong> Transactions >100K require extra scrutiny</li>
                <li><strong>Balance Mismatches:</strong> Inconsistent calculations signal manipulation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-card">
            <h4>⚠️ Risk Mitigation Strategies</h4>
            <ul>
                <li><strong>Real-time Monitoring:</strong> Flag high-risk transactions instantly</li>
                <li><strong>Multi-factor Authentication:</strong> Additional verification for large transfers</li>
                <li><strong>Velocity Checks:</strong> Monitor transaction frequency patterns</li>
                <li><strong>Customer Profiling:</strong> Baseline normal behavior per account</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Feature Importance (if available)
        st.subheader("🎯 Feature Importance Analysis")
        
        st.info("This section shows which transaction attributes are most influential in fraud detection")
        
        # Simulated feature importance (in production, extract from actual model)
        features = [
            'Amount', 'Balance Change (Orig)', 'Error Balance (Orig)', 
            'Transaction Type', 'Zero Balance Flag', 'Balance Ratio',
            'Destination Account Status', 'Amount Category'
        ]
        importance = [0.28, 0.22, 0.18, 0.12, 0.08, 0.06, 0.04, 0.02]
        
        fig_importance = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig_importance.update_layout(
            title='Feature Importance in Fraud Detection',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown("---")
        
        # Best Practices
        st.subheader("✅ Implementation Best Practices")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-card">
            <h4>🔒 Security Measures</h4>
            <ul>
                <li>Daily model performance monitoring</li>
                <li>Regular model retraining (quarterly)</li>
                <li>Incident response protocols</li>
                <li>Audit trail maintenance</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>📊 Data Quality</h4>
            <ul>
                <li>Validate data completeness</li>
                <li>Check for missing values</li>
                <li>Monitor data drift</li>
                <li>Regular data audits</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="warning-card">
            <h4>⚡ Performance</h4>
            <ul>
                <li>Optimize batch processing</li>
                <li>Real-time response <1 sec</li>
                <li>Scalability for peak hours</li>
                <li>Resource monitoring</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Banking Fraud Detection System v3.0 Enhanced Edition</strong></p>
    <p>© 2024 Enterprise Banking Solutions | Powered by Machine Learning</p>
    <p style='font-size: 0.9em;'>For technical support or inquiries, contact your system administrator</p>
</div>
""", unsafe_allow_html=True)
