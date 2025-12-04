# 🚢 Titanic Survival Prediction Dashboard

**A Premium, Production-Grade Streamlit Application**

## 📋 Overview

This is a comprehensive machine learning dashboard for predicting Titanic passenger survival. Built with modern web technologies and featuring a stunning glassmorphism UI, this application demonstrates advanced ML techniques, model explainability, and interactive data visualization.

### ✨ Key Features

- **🎨 Modern UI/UX**: Glassmorphism design with dark/light theme toggle
- **📊 Interactive 3D Visualization**: Explore data in three dimensions
- **🤖 Multiple ML Models**: Compare 4 different algorithms
- **🔍 SHAP Explainability**: Understand model predictions
- **🎯 Real-time Predictions**: Interactive prediction interface
- **📄 PDF Report Generation**: Professional downloadable reports
- **⚡ Responsive Design**: Works on desktop and mobile

## 🛠️ Technology Stack

- **Framework**: Streamlit
- **ML Libraries**: scikit-learn, XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Explainability**: SHAP
- **PDF Generation**: ReportLab
- **Styling**: Custom CSS with glassmorphism

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or navigate to the project directory**
   ```bash
   cd c:\Users\pak\Desktop\Titanic_internship
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify dataset is present**
   - Ensure `Titanic-Dataset.csv` is in the project root
   - The file should contain columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   - The app will automatically open in a browser
   - Default URL: http://localhost:8501

## 📖 User Guide

### Navigation

The dashboard contains 7 main pages accessible via the sidebar:

#### 🏠 Home
- Project overview and introduction
- Achievement badges showing key metrics
- Developer profile section
- Quick navigation to other pages

#### 📊 Data Exploration
- Dataset overview with statistics
- 3D interactive scatter plot (Age vs Fare vs Class)
- Survival distribution charts
- Correlation heatmap
- Gender and class analysis

#### ⚙️ Feature Engineering
- Visual pipeline timeline
- Detailed explanation of preprocessing steps:
  - Missing value handling
  - Title extraction
  - Family features (FamilySize, IsAlone)
  - Age and Fare binning
  - Deck extraction
  - Encoding and scaling

#### 🤖 Model Training & Comparison
- Performance metrics table for all models
- Radar chart comparing model metrics
- ROC curves and Precision-Recall curves
- Confusion matrices
- Feature importance analysis
- Models included:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Gradient Boosting

#### 🔍 Model Explainability
- SHAP (SHapley Additive exPlanations) analysis
- Summary plots showing feature impact
- Feature importance rankings
- Individual prediction explanations
- Force plots and waterfall charts

#### 🎯 Make Prediction
- Interactive form for passenger data input
- Real-time survival prediction
- Animated probability gauge
- Confidence metrics
- Explanation of prediction reasoning

#### 📄 Download Report
- Generate a comprehensive PDF report
- Download cleaned dataset (CSV)
- Access trained models
- Export analysis results

## 🎨 Features Showcase

### Glassmorphism UI
- Frosted glass effect panels
- Backdrop blur and transparency
- Gradient borders with glow effects
- Smooth hover animations

### Dark/Light Theme
- Toggle between dark and light modes
- Persistent theme preference
- Optimized color schemes for both themes

### Advanced Animations
- Fade-in page transitions
- Animated counters for metrics
- Smooth chart rendering
- Interactive hover effects

### 3D Visualizations
- Fully interactive 3D scatter plots
- Rotate, zoom, and pan controls
- Colored by survival status
- Hover tooltips with detailed info

## 📁 Project Structure

```
Titanic_internship/
├── app.py                      # Main Streamlit application
├── model_utils.py              # ML pipeline and training
├── explainer.py                # SHAP explainability
├── visuals.py                  # Visualization functions
├── requirements.txt            # Python dependencies
├── Titanic-Dataset.csv         # Dataset (required)
├── assets/
│   ├── style.css              # Custom glassmorphism CSS
│   └── profile_placeholder.png # Profile image
└── models/
    ├── best_model.pkl         # Saved best model (generated)
    └── scaler.pkl             # Saved scaler (generated)
```

### Customizing UI

Edit `assets/style.css` to customize:
- Colors (CSS variables in `:root`)
- Glassmorphism effects
- Animations
- Layout spacing

## 📊 Model Performance

The dashboard trains and compares 4 models:

| Model | Expected Accuracy | Key Strength |
|-------|------------------|--------------|
| Logistic Regression | ~80% | Fast, interpretable |
| Random Forest | ~83% | Robust, handles non-linearity |
| XGBoost | ~85% | High performance, feature importance |
| Gradient Boosting | ~84% | Strong generalization |

*Actual performance may vary based on data preprocessing and feature engineering*

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

## 📝 Credits

**Developer**: Eshaal Malik 
**Title**: Aspiring Data Scientist | ML Enthusiast  
**Date**: December 2025

## 📄 License

This project is created for educational and portfolio purposes.

## 🙏 Acknowledgments

- Titanic dataset from Kaggle
- Streamlit for the amazing framework
- SHAP library for model interpretability
- Plotly for interactive visualizations

---

**Built with ❤️ for data science excellence**

