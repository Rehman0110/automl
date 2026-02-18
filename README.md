# AutoML System

An automated machine learning system that trains and compares multiple models on any CSV dataset, selecting the best performer. Includes a Streamlit web interface for interactive use.

## Features

- **Automatic problem detection** — classifies tasks as classification or regression based on the target column
- **Multiple model comparison** — trains and evaluates several algorithms with hyperparameter tuning via `RandomizedSearchCV`
- **Preprocessing pipeline** — automatically scales numeric features and one-hot encodes categorical features
- **Model serialization** — saves the best model to disk using joblib
- **Web UI** — interactive Streamlit app for uploading data, selecting a target, and viewing results

## Supported Models

| Classification | Regression |
|---|---|
| Logistic Regression | Linear Regression |
| Random Forest | Ridge |
| Gradient Boosting | Lasso |
| SVM | Random Forest |
| KNN | Gradient Boosting |
| Decision Tree | SVR |
| Naive Bayes | KNN |
| | Decision Tree |

## Project Structure

```
├── app.py                  # Streamlit web application
├── train.py                # CLI training script
├── requirements.txt        # Python dependencies
└── src/
    ├── automl.py           # Core AutoML pipeline
    ├── models.py           # Model definitions and hyperparameter grids
    ├── preprocessing.py    # Feature preprocessing (scaling, encoding)
    └── serializer.py       # Model save/load utilities
```

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web App (Streamlit)

```bash
streamlit run app.py
```

1. Upload a CSV file
2. Select the target column
3. Click **Run AutoML**
4. View the leaderboard and the best model is saved automatically

### CLI

Edit `train.py` to set your CSV path and target column, then run:

```bash
python train.py
```

The best model will be saved as `best_model.pkl`.

## Loading a Saved Model

```python
from src.serializer import load_model

model = load_model("best_model.pkl")
predictions = model.predict(new_data)
```
