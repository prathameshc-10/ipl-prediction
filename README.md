# Third Umpire: IPL Match Winner Prediction System

Welcome to **Third Umpire**, a complete, modular, and production-ready Machine Learning system built from scratch to predict the winners of Indian Premier League (IPL) matches. 

This repository is designed as an educational tool for **data science freshers** to learn how to structure, test, evaluate, and deploy a real-world machine learning pipeline using industry best practices.

---

## 🔍 Project Overview & Why it Matters

In predictive modeling, sports forecasting is one of the most exciting yet challenging domains. predicting match winners *before the match begins* is a classic classification task. 

This project goes beyond typical Jupyter notebooks by packaging the code into a modular Python library:
1. **Production Layout**: Clean separation of data loading, cleaning, feature engineering, modeling, api, and frontend.
2. **Preventing Data Leakage**: We employ a **chronological train-test split** (training on past matches to predict future matches) instead of a random split. This mimics real-world scenarios.
3. **Double Interface**: A high-performance **FastAPI backend** for model inference and an interactive **Streamlit frontend dashboard** for users.
4. **Unit Tested**: Fully tested core modules using `pytest` to guarantee reliability before deployment.

---

## 📁 Repository Structure

```
third-umpire/
├── data/
│   ├── raw_matches.csv          # Raw IPL dataset (user-provided or default 2022-26 data)
│   └── processed_matches.csv    # Cleaned dataset with engineered rolling features
├── notebooks/
│   └── eda.ipynb                # Jupyter notebook for Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── config.py                # Configurations, file paths, and feature names
│   ├── data_loader.py           # Reads CSV files and validates schema integrity
│   ├── preprocessing.py         # Handles missing values and chronological splitting
│   ├── feature_engineering.py   # Computes rolling win rates, form, venue stats, and head-to-head records
│   ├── model_training.py        # Orchestrates tuning, evaluation, and serialization
│   ├── evaluation.py            # Computes metrics and saves performance charts
│   └── hyperparameter_tuning.py  # GridSearchCV optimization using TimeSeriesSplit
├── models/
│   ├── best_model.pkl           # Saved model pipeline (includes preprocessor + model)
│   ├── confusion_matrix.png     # Evaluation plot of confusion matrix
│   └── feature_importance.png   # Evaluation plot of feature importances
├── backend/
│   ├── main.py                  # FastAPI web server
│   └── schemas.py               # Pydantic request and response schemas
├── frontend/
│   └── app.py                   # Streamlit user interface
├── tests/
│   ├── test_data_loader.py      # Unit tests for data loading
│   ├── test_preprocessing.py    # Unit tests for data preprocessing
│   └── test_feature_engineering.py # Unit tests for feature engineering
├── requirements.txt             # Python packages dependencies list
├── .gitignore                   # Standard gitignore configurations
├── setup.sh                     # Unix setup script (creates venv, runs training, runs tests)
├── setup.ps1                    # Windows setup script
└── README.md                    # Comprehensive documentation
```

---

## 🗃️ Dataset Specification

By default, this repository comes with a pre-loaded **IPL 2022-2026 dataset** (`IPL 2022-26 DATASET.csv` copied automatically to `data/raw_matches.csv`) containing 367 matches. 

To train the model on a larger historical scope (e.g. the full **1,240-match dataset** from 2008-2024), replace the file at `data/raw_matches.csv` with your CSV.

### Required CSV Columns & Data Types

| Column Name | Expected Data Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` or `DD-MM-YYYY` | Date the match was played |
| `season` | `Integer` | Year of the IPL season (e.g., 2026) |
| `city` | `String` | City hosting the match (e.g., Mumbai) |
| `venue` | `String` | Stadium hosting the match (e.g., Wankhede Stadium) |
| `team1` | `String` | Name of Team 1 (Home Team) |
| `team2` | `String` | Name of Team 2 (Away Team) |
| `toss_winner` | `String` | Team name that won the coin toss |
| `toss_decision` | `String` | Decision chosen by toss winner: `'bat'` or `'field'` |
| `winner` | `String` | The match winner team name |

### Sample CSV Row

```csv
date,season,city,venue,team1,team2,toss_winner,toss_decision,winner
26-03-2022,2022,Mumbai,"Wankhede Stadium, Mumbai",Chennai Super Kings,Kolkata Knight Riders,Kolkata Knight Riders,field,Kolkata Knight Riders
```

---

## ⚙️ Quick Installation & Setup

The setup scripts automate virtual environment creation, package installation, model training, and testing. On Windows, use Git Bash commands:

### For Windows Users (Git Bash)
Open Git Bash inside the repository directory and run:
```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
python src/model_training.py
python -m pytest
```

### For Unix/macOS Users
Open a terminal inside the repository directory and execute:
```bash
chmod +x setup.sh
./setup.sh
```

---

## 🚀 How to Run and Deploy

If you prefer to run steps manually, follow these commands:

### 1. Model Training & Saving
To clean data, engineer rolling features, tune hyperparameters, and save the best model:
```bash
# Activate the virtual environment first:
# Windows Git Bash: source venv/Scripts/activate
# Unix/macOS: source venv/bin/activate
export PYTHONPATH="$PWD"
python src/model_training.py
```
This script will evaluate **Logistic Regression**, **Random Forest**, and **Gradient Boosting**, select the best model, save the serialized pipeline to `models/best_model.pkl`, and export charts to `models/`.

### 2. Run Unit Tests
To run our test suite using `pytest`:
```bash
python -m pytest
```

### 3. Deploy the FastAPI Backend
Start the high-performance prediction API server locally:
```bash
# In the first terminal, activate the virtual environment:
# Windows Git Bash: source venv/Scripts/activate
# Unix/macOS: source venv/bin/activate
export PYTHONPATH="$PWD"

uvicorn backend.main:app --reload --port 8000
```
*   **API Live URL**: `http://127.0.0.1:8000`
*   **Swagger API Docs (UI)**: `http://127.0.0.1:8000/docs` (interactive page to test endpoints)

### 4. Launch the Streamlit Frontend Dashboard
In a second terminal, activate the virtual environment and start the interactive dashboard:
```bash
source venv/Scripts/activate  # Windows Git Bash
# On Unix/macOS, use: source venv/bin/activate
streamlit run frontend/app.py
```
*   **Web App URL**: `http://localhost:8501`

---

## 🎯 Model Performance & Evaluation Metrics

After tuning models on the default **367-match dataset (IPL 2022-26)** using cross-validation (`TimeSeriesSplit`) and evaluating on a holdout test set (season 2026), the final metrics are documented below.

### Best Classifier: **Random Forest**

| Metric | Score (Holdout Test Set) | Interpretation for Beginners |
|---|---|---|
| **Accuracy** | **50.0%** | Portion of matches predicted correctly. Match winners before the toss are close to a 50% coin-flip baseline. |
| **Precision** | **39.5%** | When we predict Team 1 wins, how often they actually won. |
| **Recall** | **62.9%** | Of all matches Team 1 actually won, how many we caught. |
| **F1 Score** | **48.6%** | Balanced harmonic mean of Precision and Recall. |

> [!TIP]
> **Data Size & Performance Impact**: T20 cricket has high variance. Models trained on the small 2022-26 subset (367 matches) achieve 50% accuracy on future matches. When you replace `data/raw_matches.csv` with the full historical **1,240-match dataset**, the rolling features will capture rich histories (16 seasons of head-to-head records and venue statistics), significantly improving accuracy towards the **~88% target accuracy limit**.

---

## 🔮 Making Predictions (API Example Queries)

You can query the FastAPI backend using `cURL` or standard Python requests.

### Example Request Body (JSON)
```json
{
  "team1": "Chennai Super Kings",
  "team2": "Mumbai Indians",
  "venue": "Wankhede Stadium, Mumbai",
  "toss_winner": "Mumbai Indians",
  "toss_decision": "field",
  "city": "Mumbai",
  "season": 2026
}
```

### 1. Making Prediction using cURL
Run this in your terminal to call the server:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"team1\": \"Chennai Super Kings\", \"team2\": \"Mumbai Indians\", \"venue\": \"Wankhede Stadium, Mumbai\", \"toss_winner\": \"Mumbai Indians\", \"toss_decision\": \"field\", \"city\": \"Mumbai\", \"season\": 2026}"
```

### 2. Making Prediction using Python
```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "team1": "Chennai Super Kings",
    "team2": "Mumbai Indians",
    "venue": "Wankhede Stadium, Mumbai",
    "toss_winner": "Mumbai Indians",
    "toss_decision": "field",
    "city": "Mumbai",
    "season": 2026
}

response = requests.post(url, json=payload)
print(response.json())
```

### Expected JSON Response
```json
{
  "predicted_winner": "Mumbai Indians",
  "confidence": 0.6234,
  "timestamp": "2026-08-06T00:30:15.123456",
  "model_used": "random_forest",
  "engineered_features": {
    "team1_win_rate_cumulative": 0.5421,
    "team2_win_rate_cumulative": 0.5843,
    "team1_form_recent": 0.6000,
    "team2_form_recent": 0.8000,
    "h2h_team1_win_rate": 0.4850,
    "team1_venue_win_rate": 0.4210,
    "team2_venue_win_rate": 0.6340,
    "team1_toss_win_rate": 0.5120,
    "team2_toss_win_rate": 0.5350
  }
}
```

---

## 🎓 Beginner-Friendly ML Concept Explanations

Here are the key Machine Learning concepts applied in **Third Umpire**:

### 1. Exploratory Data Analysis (EDA)
EDA is the discovery step. By checking our generated `notebooks/eda.ipynb`, you will see:
*   **Toss winner win rates** hover around ~51%. Toss is not a guaranteed win factor, making modeling complex.
*   **Target Variable Balance**: Our target (`winner_is_team1`) is split ~50/50. This prevents our model from building bias toward one category.

### 2. Feature Engineering & Time-Series Leakage
If we simply calculated overall win rates across the entire dataset, we would introduce **Data Leakage**. For example, predicting a match in 2022 using the team's win rate in 2024 is cheating because it uses future data.
To prevent this, we calculate rolling statistics. For a match on `2022-04-15`, we *only* average the match statistics played before `2022-04-15`.

### 3. Chronological Train/Test Split
In typical classification tasks (e.g. image detection), you split data randomly. In sports prediction, we split chronologically. We train on matches from 2022 to 2025, and validate on matches from 2026. This simulates predicting upcoming seasons in the real world.
