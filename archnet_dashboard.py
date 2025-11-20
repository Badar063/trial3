# archnet_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ArchNet Medical AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING - FIXED VISIBILITY
# =============================================================================

def apply_custom_styles():
    st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0f0f0f;
        color: #ffffff;
    }
    
    /* Headers - Fixed visibility */
    .main-header {
        font-size: 3rem;
        color: #8A2BE2;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #8A2BE2;
        border-bottom: 2px solid #8A2BE2;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    /* Sidebar - Fixed text visibility */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1a1a1a;
    }
    
    .css-1d391kg p, .css-1lcbmhc p {
        color: #ffffff !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    
    /* Dataset cards */
    .dataset-card {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    
    /* Model comparison cards */
    .model-comparison {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton button {
        background: #8A2BE2;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton button:hover {
        background: #7B1FA2;
        color: white;
    }
    
    /* Select boxes and radio buttons - Fixed visibility */
    .stSelectbox div div {
        background-color: #2a2a2a !important;
        color: white !important;
    }
    
    .stRadio label {
        color: white !important;
    }
    
    .stSelectbox label {
        color: white !important;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1a1a1a !important;
        color: white !important;
    }
    
    /* Text colors - Fixed all text visibility */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    .stText {
        color: #ffffff !important;
    }
    
    p {
        color: #ffffff !important;
    }
    
    label {
        color: #ffffff !important;
    }
    
    /* Metric values */
    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
    }
    
    [data-testid="metric-value"] {
        color: #8A2BE2 !important;
        font-size: 1.5rem !important;
        font-weight: bold;
    }
    
    [data-testid="metric-label"] {
        color: #ffffff !important;
        font-size: 1rem !important;
    }
    
    [data-testid="metric-delta"] {
        color: #00FF7F !important;
    }
    
    /* Make sure all streamlit text is visible */
    .css-1offfwp {
        color: #ffffff !important;
    }
    
    .css-12oz5g7 {
        color: #ffffff !important;
    }
    
    /* Fix widget labels */
    .st-bb {
        color: white !important;
    }
    
    .st-bc {
        color: white !important;
    }
    
    .st-bd {
        color: white !important;
    }
    
    .st-be {
        color: white !important;
    }
    
    /* Fix select box text */
    .st-bh {
        color: white !important;
    }
    
    .st-bi {
        color: white !important;
    }
    
    /* Fix radio button text */
    .st-bj {
        color: white !important;
    }
    
    .st-bk {
        color: white !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_results_data():
    """Load the benchmark results from CSV"""
    try:
        df = pd.read_csv('archnet_real_results.csv')
        return df
    except FileNotFoundError:
        st.error("Results file not found. Please ensure 'archnet_real_results.csv' is in the same directory.")
        return pd.DataFrame()

@st.cache_data
def load_training_histories():
    """Load training histories from JSON"""
    try:
        with open('training_histories.json', 'r') as f:
            histories = json.load(f)
        return histories
    except FileNotFoundError:
        st.error("Training histories file not found. Please ensure 'training_histories.json' is in the same directory.")
        return {}

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_performance_radar_chart(df, dataset_name):
    """Create radar chart for model performance comparison"""
    models = df[df['dataset_name'] == dataset_name]
    
    fig = go.Figure()
    
    metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for model in models['model_name'].unique():
        model_data = models[models['model_name'] == model]
        values = [model_data[metric].values[0] for metric in metrics]
        # Add first value at end to close the radar chart
        values = values + [values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_names + [metric_names[0]],
            fill='toself',
            name=model,
            line=dict(color=get_model_color(model))
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f"Model Performance Radar - {dataset_name.upper()}",
        template="plotly_dark",
        height=400
    )
    
    return fig

def create_training_history_plot(histories, dataset_name, model_name):
    """Create training history visualization"""
    if dataset_name not in histories or model_name not in histories[dataset_name]:
        return None
    
    history = histories[dataset_name][model_name]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))
    
    # Accuracy plot
    if 'accuracy' in history and 'val_accuracy' in history:
        fig.add_trace(
            go.Scatter(y=history['accuracy'], name='Training Accuracy', 
                      line=dict(color='#8A2BE2')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_accuracy'], name='Validation Accuracy',
                      line=dict(color='#FFD700')),
            row=1, col=1
        )
    
    # Loss plot
    if 'loss' in history and 'val_loss' in history:
        fig.add_trace(
            go.Scatter(y=history['loss'], name='Training Loss',
                      line=dict(color='#8A2BE2')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Validation Loss',
                      line=dict(color='#FFD700')),
            row=1, col=2
        )
    
    fig.update_layout(
        title=f"Training History - {model_name} on {dataset_name.upper()}",
        template="plotly_dark",
        height=400,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    
    return fig

def create_comparison_bar_chart(df, metric, title):
    """Create comparison bar chart for different metrics"""
    fig = px.bar(df, x='model_name', y=metric, color='dataset_name',
                 title=title,
                 color_discrete_sequence=['#8A2BE2', '#FFD700', '#00FF7F', '#FF6347'],
                 template="plotly_dark")
    
    fig.update_layout(
        xaxis_title="Model Architecture",
        yaxis_title=metric.replace('_', ' ').title(),
        height=400
    )
    
    return fig

def get_model_color(model_name):
    """Get consistent colors for each model"""
    color_map = {
        'VGG16': '#8A2BE2',      # Purple
        'ResNet50': '#FFD700',   # Gold
        'DenseNet121': '#00FF7F', # Green
        'EfficientNetB0': '#FF6347' # Red
    }
    return color_map.get(model_name, '#8A2BE2')

# =============================================================================
# METRIC DISPLAY FUNCTIONS
# =============================================================================

def display_metric_card(value, label, delta=None):
    """Display a metric card with custom styling"""
    if delta is not None:
        st.metric(label=label, value=value, delta=delta)
    else:
        st.metric(label=label, value=value)

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    # Apply custom styles
    apply_custom_styles()
    
    # Load data
    results_df = load_results_data()
    training_histories = load_training_histories()
    
    if results_df.empty:
        st.error("No data available. Please run the ArchNet benchmarking first.")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🧠 ArchNet Medical AI Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🔧 Navigation")
    
    # Dataset selection
    datasets = results_df['dataset_name'].unique()
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        datasets,
        format_func=lambda x: x.upper().replace('_', ' ')
    )
    
    # Model selection
    models = results_df['model_name'].unique()
    selected_model = st.sidebar.selectbox(
        "Select Model Architecture",
        models
    )
    
    # Analysis type selection
    analysis_type = st.sidebar.radio(
        "Analysis Type",
        ["📊 Overview", "🔍 Model Comparison", "📈 Training Analysis", "🎯 Performance Deep Dive"]
    )
    
    # Main content area
    if analysis_type == "📊 Overview":
        display_overview(results_df, training_histories, selected_dataset)
    
    elif analysis_type == "🔍 Model Comparison":
        display_model_comparison(results_df, selected_dataset)
    
    elif analysis_type == "📈 Training Analysis":
        display_training_analysis(results_df, training_histories, selected_dataset, selected_model)
    
    elif analysis_type == "🎯 Performance Deep Dive":
        display_performance_deep_dive(results_df, selected_dataset)

def display_overview(df, histories, dataset):
    """Display overview dashboard"""
    st.markdown(f'<h2 class="section-header">📊 {dataset.upper()} Performance Overview</h2>', 
                unsafe_allow_html=True)
    
    # Filter data for selected dataset
    dataset_data = df[df['dataset_name'] == dataset]
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_model = dataset_data.loc[dataset_data['test_accuracy'].idxmax()]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        display_metric_card(
            f"{best_model['test_accuracy']:.3f}",
            "Best Accuracy",
            best_model['model_name']
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        fastest_model = dataset_data.loc[dataset_data['inference_time'].idxmin()]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        display_metric_card(
            f"{fastest_model['inference_time']:.4f}s",
            "Fastest Inference",
            fastest_model['model_name']
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        smallest_model = dataset_data.loc[dataset_data['model_size_mb'].idxmin()]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        display_metric_card(
            f"{smallest_model['model_size_mb']:.1f} MB",
            "Smallest Model",
            smallest_model['model_name']
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        best_f1_model = dataset_data.loc[dataset_data['f1_score'].idxmax()]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        display_metric_card(
            f"{best_f1_model['f1_score']:.3f}",
            "Best F1-Score",
            best_f1_model['model_name']
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_performance_radar_chart(df, dataset), use_container_width=True)
    
    with col2:
        dataset_comparison = df[df['dataset_name'] == dataset]
        fig = create_comparison_bar_chart(
            dataset_comparison, 
            'test_accuracy', 
            f'Accuracy Comparison - {dataset.upper()}'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed performance table
    st.markdown('<h3 class="section-header">📋 Detailed Performance Metrics</h3>', 
                unsafe_allow_html=True)
    
    # Format the table for better display
    display_df = dataset_data[[
        'model_name', 'test_accuracy', 'precision', 'recall', 'f1_score',
        'inference_time', 'model_size_mb', 'training_time'
    ]].copy()
    
    display_df['test_accuracy'] = display_df['test_accuracy'].round(3)
    display_df['precision'] = display_df['precision'].round(3)
    display_df['recall'] = display_df['recall'].round(3)
    display_df['f1_score'] = display_df['f1_score'].round(3)
    display_df['inference_time'] = display_df['inference_time'].round(4)
    display_df['model_size_mb'] = display_df['model_size_mb'].round(1)
    display_df['training_time'] = display_df['training_time'].round(1)
    
    st.dataframe(display_df, use_container_width=True)

def display_model_comparison(df, dataset):
    """Display model comparison analysis"""
    st.markdown(f'<h2 class="section-header">🔍 {dataset.upper()} Model Architecture Comparison</h2>', 
                unsafe_allow_html=True)
    
    dataset_data = df[df['dataset_name'] == dataset]
    
    # Comparison metrics
    metrics = ['test_accuracy', 'precision', 'recall', 'f1_score', 
               'inference_time', 'model_size_mb', 'training_time']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 
                    'Inference Time (s)', 'Model Size (MB)', 'Training Time (s)']
    
    cols = st.columns(2)
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        with cols[i % 2]:
            fig = create_comparison_bar_chart(
                dataset_data, metric, f'{name} - {dataset.upper()}'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Model recommendations
    st.markdown('<h3 class="section-header">🎯 Model Recommendations</h3>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🏆 Best Overall**")
        best_overall = dataset_data.loc[dataset_data['test_accuracy'].idxmax()]
        st.markdown(f"**{best_overall['model_name']}**")
        st.markdown(f"Accuracy: {best_overall['test_accuracy']:.3f}")
        st.markdown(f"F1-Score: {best_overall['f1_score']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**⚡ Fastest Inference**")
        fastest = dataset_data.loc[dataset_data['inference_time'].idxmin()]
        st.markdown(f"**{fastest['model_name']}**")
        st.markdown(f"Time: {fastest['inference_time']:.4f}s")
        st.markdown(f"Accuracy: {fastest['test_accuracy']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📦 Most Efficient**")
        smallest = dataset_data.loc[dataset_data['model_size_mb'].idxmin()]
        st.markdown(f"**{smallest['model_name']}**")
        st.markdown(f"Size: {smallest['model_size_mb']:.1f} MB")
        st.markdown(f"Accuracy: {smallest['test_accuracy']:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

def display_training_analysis(df, histories, dataset, model):
    """Display training analysis"""
    st.markdown(f'<h2 class="section-header">📈 {model} Training Analysis on {dataset.upper()}</h2>', 
                unsafe_allow_html=True)
    
    # Training history plot
    fig = create_training_history_plot(histories, dataset, model)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No training history available for {model} on {dataset}")
    
    # Model statistics
    model_data = df[(df['dataset_name'] == dataset) & (df['model_name'] == model)]
    
    if not model_data.empty:
        st.markdown('<h3 class="section-header">📊 Model Statistics</h3>', 
                    unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            display_metric_card(
                f"{model_data['test_accuracy'].values[0]:.3f}",
                "Test Accuracy"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            display_metric_card(
                f"{model_data['training_time'].values[0]:.1f}s",
                "Training Time"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            display_metric_card(
                f"{model_data['model_size_mb'].values[0]:.1f} MB",
                "Model Size"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            display_metric_card(
                f"{model_data['model_parameters'].values[0]:,}",
                "Parameters"
            )
            st.markdown('</div>', unsafe_allow_html=True)

def display_performance_deep_dive(df, dataset):
    """Display performance deep dive analysis"""
    st.markdown(f'<h2 class="section-header">🎯 {dataset.upper()} Performance Deep Dive</h2>', 
                unsafe_allow_html=True)
    
    dataset_data = df[df['dataset_name'] == dataset]
    
    # Performance matrix
    st.markdown('<h3 class="section-header">📈 Performance Matrix</h3>', 
                unsafe_allow_html=True)
    
    # Create a performance score combining multiple metrics
    dataset_data = dataset_data.copy()
    dataset_data['performance_score'] = (
        dataset_data['test_accuracy'] * 0.4 +
        dataset_data['f1_score'] * 0.3 +
        (1 / dataset_data['inference_time']) * 0.15 +
        (1 / dataset_data['model_size_mb']) * 0.15
    )
    
    # Sort by performance score
    dataset_data = dataset_data.sort_values('performance_score', ascending=False)
    
    # Display performance ranking
    st.markdown("### 🏆 Performance Ranking")
    for i, (_, row) in enumerate(dataset_data.iterrows(), 1):
        with st.container():
            st.markdown(f'<div class="model-comparison">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"**#{i} {row['model_name']}**")
                st.markdown(f"Performance Score: {row['performance_score']:.3f}")
            
            with col2:
                st.markdown(f"Accuracy: {row['test_accuracy']:.3f}")
                st.markdown(f"F1-Score: {row['f1_score']:.3f}")
            
            with col3:
                st.markdown(f"Speed: {row['inference_time']:.4f}s")
                st.markdown(f"Size: {row['model_size_mb']:.1f}MB")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Trade-off analysis
    st.markdown('<h3 class="section-header">⚖️ Accuracy vs Speed Trade-off</h3>', 
                unsafe_allow_html=True)
    
    fig = px.scatter(
        dataset_data, 
        x='inference_time', 
        y='test_accuracy',
        size='model_size_mb',
        color='model_name',
        hover_name='model_name',
        title='Accuracy vs Inference Time Trade-off',
        color_discrete_sequence=['#8A2BE2', '#FFD700', '#00FF7F', '#FF6347'],
        template="plotly_dark"
    )
    
    fig.update_layout(
        xaxis_title="Inference Time (seconds)",
        yaxis_title="Accuracy",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# RUN THE DASHBOARD
# =============================================================================

if __name__ == "__main__":
    main()
