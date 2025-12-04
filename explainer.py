"""
SHAP Explainability Module for Titanic Survival Prediction
Provides model interpretation and explanation visualizations
"""

import shap
import streamlit as st
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set matplotlib style
plt.style.use('default')


# ============================================
# SHAP EXPLAINER INITIALIZATION
# ============================================

@st.cache_resource
def get_shap_explainer(_model, X_train):
    """Initialize SHAP explainer for tree-based models"""
    try:
        # TreeExplainer for tree-based models
        if hasattr(_model, 'predict_proba'):
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer.shap_values(X_train)
            # For binary classification, get values for positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            return explainer, shap_values
        else:
            return None, None
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}")
        return None, None


# ============================================
# SHAP VISUALIZATIONS
# ============================================

def plot_shap_summary(explainer, X_sample, max_display=15):
    """
    Create SHAP summary plot (beeswarm plot)
    Shows feature importance and impact distribution
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create beeswarm plot
        shap.summary_plot(
            shap_values, 
            X_sample, 
            max_display=max_display,
            show=False,
            plot_type="dot"
        )
        
        plt.title('SHAP Summary Plot - Feature Impact on Survival', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=11)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating SHAP summary plot: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def plot_shap_bar(explainer, X_sample, max_display=15):
    """
    Create SHAP bar plot
    Shows mean absolute SHAP values (feature importance)
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create bar plot
        shap.summary_plot(
            shap_values, 
            X_sample, 
            max_display=max_display,
            show=False,
            plot_type="bar"
        )
        
        plt.title('SHAP Feature Importance - Mean |SHAP Value|', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Mean Absolute SHAP Value', fontsize=11)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating SHAP bar plot: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def plot_shap_force(explainer, X_sample, index=0, base_values=None):
    """
    Create SHAP force plot for individual prediction
    Shows how features contribute to a specific prediction
    """
    try:
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get expected value (base value)
        if base_values is None:
            if isinstance(explainer.expected_value, list):
                base_value = explainer.expected_value[1]
            else:
                base_value = explainer.expected_value
        else:
            base_value = base_values
        
        # Create force plot
        force_plot = shap.force_plot(
            base_value,
            shap_values[index],
            X_sample.iloc[index],
            matplotlib=True,
            show=False
        )
        
        return force_plot
    except Exception as e:
        print(f"Error creating SHAP force plot: {e}")
        return None


def plot_shap_waterfall(explainer, X_sample, index=0):
    """
    Create SHAP waterfall plot for individual prediction
    Shows cumulative feature contributions
    """
    try:
        shap_values = explainer(X_sample)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_values[index], max_display=15, show=False)
        
        plt.title(f'SHAP Waterfall Plot - Prediction Breakdown', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"Error creating SHAP waterfall plot: {e}")
        return None


def get_shap_explanation_text(explainer, X_sample, index=0, top_n=3):
    """
    Generate human-readable explanation from SHAP values
    Returns top contributing features and their impact
    """
    try:
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get feature contributions for this instance
        feature_values = X_sample.iloc[index]
        feature_shap = shap_values[index]
        
        # Create DataFrame of contributions
        contributions = pd.DataFrame({
            'feature': X_sample.columns,
            'value': feature_values.values,
            'shap': feature_shap
        })
        
        # Sort by absolute SHAP value
        contributions['abs_shap'] = np.abs(contributions['shap'])
        contributions = contributions.sort_values('abs_shap', ascending=False)
        
        # Get top contributing features
        top_features = contributions.head(top_n)
        
        # Generate explanation text
        explanation_parts = []
        for _, row in top_features.iterrows():
            feature = row['feature']
            value = row['value']
            shap_val = row['shap']
            
            impact = "increases" if shap_val > 0 else "decreases"
            explanation_parts.append(
                f"**{feature}** (value: {value:.2f}) {impact} survival probability by {abs(shap_val):.3f}"
            )
        
        explanation = "\n\n".join(explanation_parts)
        
        return explanation, contributions
    except Exception as e:
        print(f"Error generating SHAP explanation text: {e}")
        return "Explanation not available", None


def get_feature_impact_summary(explainer, X_sample):
    """
    Calculate overall feature importance from SHAP values
    Returns DataFrame with mean absolute SHAP values
    """
    try:
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'feature': X_sample.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        return feature_importance
    except Exception as e:
        print(f"Error calculating feature impact: {e}")
        return None


# ============================================
# INTERACTIVE SHAP DISPLAY
# ============================================

def display_shap_analysis(model, X_train, X_test, feature_names):
    """
    Complete SHAP analysis display for Streamlit
    """
    st.markdown("### 🔍 SHAP (SHapley Additive exPlanations) Analysis")
    
    # Initialize explainer
    with st.spinner("Initializing SHAP explainer..."):
        explainer, _ = get_shap_explainer(model, X_train)
    
    if explainer is None:
        st.warning("SHAP analysis not available for this model type.")
        return
    
    # Use test set for explanations
    X_sample = X_test.head(100)  # Limit for performance
    
    # Tabs for different SHAP visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Summary Plot", "📈 Feature Importance", "🎯 Individual Prediction"])
    
    with tab1:
        st.markdown("**SHAP Summary Plot (Beeswarm)**")
        st.markdown("Shows how each feature impacts predictions across all samples.")
        
        with st.spinner("Generating summary plot..."):
            fig_summary = plot_shap_summary(explainer, X_sample)
            if fig_summary:
                st.pyplot(fig_summary)
                plt.close()
            else:
                st.error("Failed to generate SHAP summary plot. Check console for errors.")
        
        with st.expander("ℹ️ How to read this plot"):
            st.markdown("""
            - **Position on Y-axis**: Feature name
            - **Position on X-axis**: SHAP value (impact on prediction)
            - **Color**: Feature value (red = high, blue = low)
            - **Each dot**: One passenger
            
            **Example**: If "Sex" shows many red dots on the right, it means being female (high value) 
            increases survival probability.
            """)
    
    with tab2:
        st.markdown("**SHAP Feature Importance (Bar Plot)**")
        st.markdown("Average absolute impact of each feature on predictions.")
        
        with st.spinner("Generating importance plot..."):
            fig_bar = plot_shap_bar(explainer, X_sample)
            if fig_bar:
                st.pyplot(fig_bar)
                plt.close()
            else:
                st.error("Failed to generate SHAP bar plot. Check console for errors.")
        
        # Feature importance table
        importance_df = get_feature_impact_summary(explainer, X_sample)
        if importance_df is not None:
            st.markdown("**Top 10 Most Important Features**")
            st.dataframe(
                importance_df.head(10).style.background_gradient(cmap='Blues'),
                use_container_width=True
            )
    
    with tab3:
        st.markdown("**Individual Prediction Explanation**")
        
        # Select instance to explain
        instance_idx = st.selectbox(
            "Select a passenger to explain:",
            range(min(50, len(X_sample))),
            format_func=lambda x: f"Passenger {x+1}"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Force Plot**")
            st.markdown("Shows how features push prediction higher or lower.")
            
            with st.spinner("Generating force plot..."):
                fig_force = plot_shap_force(explainer, X_sample, index=instance_idx)
                if fig_force:
                    st.pyplot(fig_force)
                    plt.close()
                else:
                    st.warning("Force plot not available. Showing waterfall plot instead.")
                    fig_waterfall = plot_shap_waterfall(explainer, X_sample, index=instance_idx)
                    if fig_waterfall:
                        st.pyplot(fig_waterfall)
                        plt.close()
                    else:
                        st.error("Failed to generate individual explanation plots.")
        
        with col2:
            st.markdown("**Explanation**")
            explanation_text, contributions = get_shap_explanation_text(
                explainer, X_sample, index=instance_idx, top_n=5
            )
            st.markdown(explanation_text)
        
        # Show instance values
        with st.expander("🔎 View passenger details"):
            instance_data = X_sample.iloc[instance_idx]
            st.json(instance_data.to_dict())
