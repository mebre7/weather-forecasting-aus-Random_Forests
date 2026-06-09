# Weather Forecasting Australia - Decision Trees & Random Forests

A machine learning project focused on predicting whether it will rain tomorrow in Australia using Decision Tree and Random Forest algorithms.

## 📋 Project Overview

This project implements a binary classification model to predict rainfall in Australia. Using historical weather data, the model learns patterns from various meteorological features to forecast whether it will rain the next day.

The dataset used is the **weatherAUS.csv** dataset, which contains weather observations from multiple locations across Australia.
- Source: [Kaggle - Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

## 🎯 Objective

Predict whether it will rain tomorrow (`RainTomorrow`) based on weather conditions including:
- Temperature (min, max, at 9am, at 3pm)
- Humidity levels
- Wind speed and direction
- Atmospheric pressure
- Rainfall amounts
- Sunshine hours
- Evaporation rates
- Cloud cover

## 📁 Project Structure

```
weather-forecasting-aus-Random_Forests/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── data/
│   ├── weatherAUS.csv                 # Original dataset
│   ├── dt_workflow.png                # Decision tree workflow diagram
│   ├── img.png                        # Accuracy visualization
│   ├── train_datasets/
│   │   ├── train_inputs.parquet       # Training input features
│   │   └── train_target.parquet       # Training target variable
│   ├── val_datasets/
│   │   ├── val_inputs.parquet         # Validation input features
│   │   └── val_target.parquet         # Validation target variable
│   └── test_datasets/
│       ├── test_inputs.parquet        # Test input features
│       └── test_target.parquet        # Test target variable
├── models/                            # Directory for saved models
└── notebook/
    └── decision_trees_and_random_forest.ipynb  # Main analysis notebook
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd weather-forecasting-aus-Random_Forests
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook notebook/decision_trees_and_random_forest.ipynb
```

## 📊 Data Preprocessing Pipeline

The project follows a comprehensive data preprocessing pipeline:

### 1. Data Cleaning
- Drop rows with missing target values (`RainToday`, `RainTomorrow`)
- Handle missing values in numeric columns using `SimpleImputer` (mean strategy)
- Replace NaN values in categorical columns with 'Unknown'

### 2. Train/Validation/Test Split
The data is split temporally by year:
- **Training**: Data before 2015
- **Validation**: Data from 2015
- **Test**: Data after 2015

### 3. Feature Engineering
- **Numeric Features (16)**: Scaled using `MinMaxScaler` to (0, 1) range
- **Categorical Features (5)**: Encoded using `OneHotEncoder`
  - Location (49 unique values)
  - WindGustDir, WindDir9am, WindDir3pm (16 unique values each)
  - RainToday (2 unique values)

### 4. Final Feature Set
After preprocessing, the dataset contains **118 features** (102 encoded categorical + 16 scaled numeric).

## 🌳 Decision Tree Model

### Implementation
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

### Results
| Metric | Score |
|--------|-------|
| Training Accuracy | ~100% |
| Validation Accuracy | ~84.5% |

The initial model showed significant **overfitting** with a deep tree (49 layers).

### Hyperparameter Tuning

#### Max Depth Tuning
Experiments with `max_depth` values from 1 to 20 revealed:
- **Optimal max_depth = 7** (lowest validation error)
- Final accuracy: **~85.1%**

#### Max Leaf Nodes
Alternative regularization using `max_leaf_nodes=128` was also explored.

### Feature Importance
The top features identified by the Decision Tree:
1. **Humidity3pm** - Most important feature (used as root node)
2. Other humidity and pressure-related features

## 🌲 Random Forest Model

### Implementation
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1, random_state=42)
model.fit(X_train, y_train)
```

### Results
| Metric | Score |
|--------|-------|
| Training Accuracy | ~99.9% |
| Validation Accuracy | ~85.8% |

Random Forest showed **better generalization** compared to a single Decision Tree, demonstrating the power of ensemble methods.

### Hyperparameter Tuning

Experiments with `n_estimators`:
- **10 estimators**: Faster training, slightly lower accuracy
- **100 estimators** (default): Good balance
- **500 estimators**: Longer training time, marginal accuracy improvement

### Feature Importance
Random Forest provides more balanced feature importance distribution compared to a single Decision Tree.

## 📈 Key Findings

1. **Humidity at 3pm** is the most predictive feature for rainfall
2. **Ensemble methods** (Random Forest) outperform single Decision Trees
3. **Hyperparameter tuning** significantly improves model generalization
4. **Temporal data splitting** (by year) is appropriate for time-series weather data
5. The model achieves approximately **85% accuracy** on validation data

## 🛠️ Technologies Used

- **Python** - Programming language
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning library
  - DecisionTreeClassifier
  - RandomForestClassifier
  - SimpleImputer
  - MinMaxScaler
  - OneHotEncoder
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive visualizations
- **Jupyter Notebook** - Interactive development environment

## 📝 Key Concepts Demonstrated

- **Gini Impurity**: Measure used for feature selection in decision trees
- **Overfitting & Regularization**: Addressed through hyperparameter tuning
- **Feature Importance**: Understanding which features drive predictions
- **Cross-validation**: Using separate train/val/test sets
- **Ensemble Learning**: Combining multiple models for better predictions

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests with improvements or suggestions.

## 📧 Contact

For questions or feedback, please open an issue in the repository.