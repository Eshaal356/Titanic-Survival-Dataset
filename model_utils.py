"""
Model Utilities for Titanic Survival Prediction Dashboard
Handles data preprocessing, feature engineering, model training, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import joblib
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


# ============================================
# DATA LOADING
# ============================================

@st.cache_data
def load_data(filepath='Titanic-Dataset.csv'):
    """Load the Titanic dataset"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# ============================================
# FEATURE ENGINEERING
# ============================================

def extract_title(name):
    """Extract title from passenger name"""
    if pd.isna(name):
        return 'Unknown'
    title = name.split(',')[1].split('.')[0].strip()
    # Group rare titles
    if title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        return 'Rare'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Mme':
        return 'Mrs'
    return title


def extract_deck(cabin):
    """Extract deck from cabin number"""
    if pd.isna(cabin) or cabin == 'Unknown':
        return 'Unknown'
    return cabin[0]


def preprocess_data(df, is_training=True, scaler=None):
    """
    Complete preprocessing pipeline:
    - Handle missing values
    - Feature engineering
    - Encoding
    - Scaling
    """
    df = df.copy()
    
    # Add missing columns that are expected (for prediction inputs)
    if 'Cabin' not in df.columns:
        df['Cabin'] = 'Unknown'
    if 'Name' not in df.columns:
        df['Name'] = 'Unknown, Mr. John'
    
    # ===== MISSING VALUE HANDLING =====
    # Age: Median imputation
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Embarked: Mode imputation
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Cabin: Mark as Unknown
    df['Cabin'].fillna('Unknown', inplace=True)
    
    # Fare: Median imputation
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # ===== FEATURE ENGINEERING =====
    # Title extraction
    df['Title'] = df['Name'].apply(extract_title)
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Is alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Age binning
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], 
                            labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Fare binning - Use cut with fixed bins to avoid errors during prediction
    # Bins based on typical Titanic fare distribution: 0-8 (Low), 8-15 (Med), 15-30 (High), 30+ (VeryHigh)
    df['FareGroup'] = pd.cut(df['Fare'], bins=[-1, 7.91, 14.45, 31, 1000], 
                            labels=['Low', 'Medium', 'High', 'VeryHigh'])
    
    # Deck extraction
    df['Deck'] = df['Cabin'].apply(extract_deck)
    
    # ===== OUTLIER HANDLING =====
    # Cap extreme fare values
    fare_99th = df['Fare'].quantile(0.99)
    df.loc[df['Fare'] > fare_99th, 'Fare'] = fare_99th
    
    # ===== FEATURE SELECTION =====
    # Drop unnecessary columns
    features_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_processed = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    
    # ===== ENCODING =====
    # Convert categorical to numeric
    df_processed['Sex'] = df_processed['Sex'].map({'male': 0, 'female': 1})
    
    # One-hot encoding for other categoricals
    categorical_cols = ['Embarked', 'Title', 'AgeGroup', 'FareGroup', 'Deck']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # ===== SCALING =====
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    
    if is_training:
        scaler = StandardScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        return df_processed, scaler
    else:
        if scaler is not None:
            df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
        return df_processed


@st.cache_data
def load_and_preprocess_data(filepath='Titanic-Dataset.csv'):
    """Load and preprocess the complete dataset"""
    df = load_data(filepath)
    if df is None:
        return None, None, None, None, None, None
    
    # Separate features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Preprocess
    X_processed, scaler = preprocess_data(X, is_training=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler, df


# ============================================
# MODEL TRAINING
# ============================================

@st.cache_resource
def train_models():
    """Train all models and return results"""
    # Load data
    X_train, X_test, y_train, y_test, scaler, df_original = load_and_preprocess_data()
    
    if X_train is None:
        return None
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=5, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_curve': roc_curve(y_test, y_pred_proba),
            'precision_recall_curve': precision_recall_curve(y_test, y_pred_proba),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            results[name]['feature_importances'] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
    
    # Save best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    try:
        joblib.dump(best_model, 'models/best_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
    except Exception as e:
        print(f"Error saving models: {e}")
    
    return results, scaler, X_train.columns.tolist()


# ============================================
# PREDICTION
# ============================================

def predict_survival(input_data, model, scaler, feature_columns):
    """
    Predict survival for new passenger data
    
    Parameters:
    - input_data: dict with keys: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    - model: trained model
    - scaler: fitted scaler
    - feature_columns: list of feature names from training
    
    Returns:
    - prediction: 0 or 1
    - probability: float between 0 and 1
    """
    # Create DataFrame from input
    df_input = pd.DataFrame([input_data])
    
    # Preprocess (same as training)
    df_processed = preprocess_data(df_input, is_training=False, scaler=scaler)
    
    # Ensure all columns from training are present
    for col in feature_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Reorder columns to match training
    df_processed = df_processed[feature_columns]
    
    # Predict
    prediction = model.predict(df_processed)[0]
    probability = model.predict_proba(df_processed)[0][1]
    
    return prediction, probability


# ============================================
# STATISTICS
# ============================================

def get_dataset_statistics(df):
    """Get key statistics about the dataset"""
    stats = {
        'total_passengers': len(df),
        'survivors': df['Survived'].sum(),
        'survival_rate': df['Survived'].mean() * 100,
        'male_count': (df['Sex'] == 'male').sum(),
        'female_count': (df['Sex'] == 'female').sum(),
        'male_survival_rate': df[df['Sex'] == 'male']['Survived'].mean() * 100,
        'female_survival_rate': df[df['Sex'] == 'female']['Survived'].mean() * 100,
        'class_1_survival': df[df['Pclass'] == 1]['Survived'].mean() * 100,
        'class_2_survival': df[df['Pclass'] == 2]['Survived'].mean() * 100,
        'class_3_survival': df[df['Pclass'] == 3]['Survived'].mean() * 100,
        'avg_age': df['Age'].mean(),
        'avg_fare': df['Fare'].mean()
    }
    return stats


# ============================================
# FEATURE IMPORTANCE
# ============================================

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df
    elif hasattr(model, 'coef_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        return importance_df
    else:
        return None
