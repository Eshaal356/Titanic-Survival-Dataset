"""
Visualization Module for Titanic Survival Dashboard
Creates interactive and animated charts using Plotly, Matplotlib, Seaborn
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st


# ============================================
# CONFIGURATION
# ============================================

# Color schemes
COLOR_SURVIVED = '#10b981'  # Green
COLOR_DIED = '#ef4444'      # Red
COLOR_PALETTE = ['#00f5ff', '#a855f7', '#ec4899', '#10b981', '#f59e0b', '#06b6d4']

# Plotly template
PLOTLY_TEMPLATE = 'plotly_dark'


# ============================================
# 3D VISUALIZATIONS
# ============================================

def plot_3d_scatter(df):
    """
    Create interactive 3D scatter plot: Age vs Fare vs Pclass
    Color by Survival status
    """
    fig = px.scatter_3d(
        df,
        x='Age',
        y='Fare',
        z='Pclass',
        color='Survived',
        color_discrete_map={0: COLOR_DIED, 1: COLOR_SURVIVED},
        labels={'Survived': 'Survived', 'Pclass': 'Passenger Class'},
        title='3D Interactive Visualization: Age vs Fare vs Class',
        template=PLOTLY_TEMPLATE,
        hover_data=['Sex', 'Embarked'],
        opacity=0.7
    )
    
    # Update layout for better aesthetics
    fig.update_layout(
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Fare',
            zaxis_title='Passenger Class',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=600,
        font=dict(size=12),
        legend=dict(
            title='Survival Status',
            orientation='v',
            yanchor='top',
            y=0.99,
            xanchor='right',
            x=0.99
        )
    )
    
    # Add animation frame for rotation
    fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color='white')))
    
    return fig


# ============================================
# DISTRIBUTION CHARTS
# ============================================

def plot_survival_distribution(df):
    """Create pie chart for survival distribution"""
    survival_counts = df['Survived'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Did Not Survive', 'Survived'],
        values=survival_counts.values,
        hole=0.4,
        marker=dict(colors=[COLOR_DIED, COLOR_SURVIVED]),
        textinfo='label+percent',
        textposition='outside',
        pull=[0.1, 0]
    )])
    
    fig.update_layout(
        title='Overall Survival Distribution',
        template=PLOTLY_TEMPLATE,
        height=400,
        showlegend=True,
        font=dict(size=12)
    )
    
    return fig


def plot_class_survival_sunburst(df):
    """Create sunburst chart for Class vs Survival"""
    # Prepare data
    df_chart = df.copy()
    df_chart['Survived_Label'] = df_chart['Survived'].map({0: 'Died', 1: 'Survived'})
    df_chart['Pclass_Label'] = df_chart['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    
    fig = px.sunburst(
        df_chart,
        path=['Pclass_Label', 'Survived_Label'],
        title='Survival by Passenger Class',
        color='Survived',
        color_discrete_map={0: COLOR_DIED, 1: COLOR_SURVIVED},
        template=PLOTLY_TEMPLATE,
        height=500
    )
    
    fig.update_traces(textinfo='label+percent parent')
    
    return fig


def plot_age_distribution(df):
    """Interactive histogram for Age distribution"""
    fig = px.histogram(
        df,
        x='Age',
        color='Survived',
        nbins=30,
        title='Age Distribution by Survival',
        labels={'Survived': 'Survived'},
        color_discrete_map={0: COLOR_DIED, 1: COLOR_SURVIVED},
        template=PLOTLY_TEMPLATE,
        barmode='overlay',
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title='Age',
        yaxis_title='Count',
        height=400
    )
    
    return fig


def plot_fare_distribution(df):
    """Interactive histogram for Fare distribution"""
    # Cap extreme values for better visualization
    df_plot = df.copy()
    df_plot['Fare_Capped'] = df_plot['Fare'].clip(upper=df_plot['Fare'].quantile(0.95))
    
    fig = px.histogram(
        df_plot,
        x='Fare_Capped',
        color='Survived',
        nbins=30,
        title='Fare Distribution by Survival (Capped at 95th percentile)',
        labels={'Survived': 'Survived', 'Fare_Capped': 'Fare'},
        color_discrete_map={0: COLOR_DIED, 1: COLOR_SURVIVED},
        template=PLOTLY_TEMPLATE,
        barmode='overlay',
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title='Fare',
        yaxis_title='Count',
        height=400
    )
    
    return fig


# ============================================
# CORRELATION & RELATIONSHIPS
# ============================================

def plot_correlation_heatmap(df):
    """Create correlation heatmap for numerical features"""
    # Select numerical columns
    numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    df_corr = df[numerical_cols].corr()
    
    fig = px.imshow(
        df_corr,
        text_auto='.2f',
        aspect='auto',
        title='Feature Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        template=PLOTLY_TEMPLATE,
        labels=dict(color="Correlation")
    )
    
    fig.update_layout(
        height=500,
        xaxis_title='',
        yaxis_title=''
    )
    
    return fig


def plot_gender_survival(df):
    """Animated bar chart: Gender vs Survival"""
    gender_survival = df.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
    gender_survival['Survived_Label'] = gender_survival['Survived'].map({0: 'Did Not Survive', 1: 'Survived'})
    
    fig = px.bar(
        gender_survival,
        x='Sex',
        y='Count',
        color='Survived_Label',
        title='Survival Count by Gender',
        labels={'Sex': 'Gender', 'Count': 'Number of Passengers'},
        color_discrete_map={'Did Not Survive': COLOR_DIED, 'Survived': COLOR_SURVIVED},
        template=PLOTLY_TEMPLATE,
        barmode='group',
        text='Count'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=400, xaxis_title='Gender', yaxis_title='Count')
    
    return fig


# ============================================
# MODEL COMPARISON CHARTS
# ============================================

def plot_model_comparison_radar(results):
    """
    Spider/Radar chart comparing model metrics
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    models = list(results.keys())
    
    fig = go.Figure()
    
    for model_name in models:
        values = [results[model_name][metric] for metric in metrics]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=model_name,
            fill='toself',
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title='Model Performance Comparison (Radar Chart)',
        template=PLOTLY_TEMPLATE,
        height=500
    )
    
    return fig


def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix heatmap"""
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Did Not Survive', 'Survived'],
        y=['Did Not Survive', 'Survived'],
        title=f'Confusion Matrix - {model_name}',
        color_continuous_scale='Blues',
        template=PLOTLY_TEMPLATE
    )
    
    fig.update_layout(height=400)
    
    return fig


def plot_roc_curve(results):
    """Plot ROC curves for all models"""
    fig = go.Figure()
    
    for model_name, data in results.items():
        fpr, tpr, _ = data['roc_curve']
        auc = data['roc_auc']
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            name=f'{model_name} (AUC={auc:.3f})',
            mode='lines',
            line=dict(width=2)
        ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves - Model Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template=PLOTLY_TEMPLATE,
        height=500,
        showlegend=True
    )
    
    return fig


def plot_precision_recall_curve(results):
    """Plot Precision-Recall curves for all models"""
    fig = go.Figure()
    
    for model_name, data in results.items():
        precision, recall, _ = data['precision_recall_curve']
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            name=model_name,
            mode='lines',
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Precision-Recall Curves - Model Comparison',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template=PLOTLY_TEMPLATE,
        height=500,
        showlegend=True
    )
    
    return fig


def plot_feature_importance(importance_df, model_name, top_n=15):
    """Plot feature importance horizontal bar chart"""
    top_features = importance_df.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Feature Importances - {model_name}',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        template=PLOTLY_TEMPLATE,
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


# ============================================
# GAUGE CHARTS
# ============================================

def plot_survival_gauge(probability):
    """
    Create gauge chart for survival probability
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Survival Probability", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': COLOR_SURVIVED if probability > 0.5 else COLOR_DIED},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=300,
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig


# ============================================
# TIMELINE VISUALIZATION
# ============================================

def create_pipeline_timeline():
    """Create visual timeline for feature engineering pipeline"""
    steps = [
        {"step": 1, "name": "Load Data", "icon": "📊"},
        {"step": 2, "name": "Clean Missing Values", "icon": "🧹"},
        {"step": 3, "name": "Engineer Features", "icon": "⚙️"},
        {"step": 4, "name": "Encode Variables", "icon": "🔢"},
        {"step": 5, "name": "Scale Features", "icon": "📏"},
        {"step": 6, "name": "Train Models", "icon": "🤖"}
    ]
    
    fig = go.Figure()
    
    # Add markers for each step
    for step in steps:
        fig.add_trace(go.Scatter(
            x=[step['step']],
            y=[1],
            mode='markers+text',
            marker=dict(size=30, color=COLOR_PALETTE[step['step']-1]),
            text=f"{step['icon']}<br>{step['name']}",
            textposition='top center',
            textfont=dict(size=12),
            showlegend=False,
            hovertemplate=f"<b>Step {step['step']}</b><br>{step['name']}<extra></extra>"
        ))
    
    # Add connecting line
    fig.add_trace(go.Scatter(
        x=list(range(1, 7)),
        y=[1]*6,
        mode='lines',
        line=dict(color='rgba(168, 85, 247, 0.5)', width=3, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Data Processing Pipeline',
        template=PLOTLY_TEMPLATE,
        height=300,
        xaxis=dict(
            title='',
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 7]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, 2]
        ),
        hovermode='closest'
    )
    
    return fig
