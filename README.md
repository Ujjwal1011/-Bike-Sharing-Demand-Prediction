
# 🚲 Bike Sharing Demand Prediction

Welcome to the Bike Sharing Demand Prediction project! This repository leverages machine learning and MLOps best practices to forecast bike sharing demand. All experiments are tracked using MLflow, with remote logging and dashboarding via DagsHub. CI/CD is managed through GitHub Actions for seamless automation.  

## 📁 Project Structure

```
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── config/
│   └── grid_config.yaml
├── data/
├── mlruns/
├── models/
├── scripts/
│   └── get_data.py
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   └── register_best_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Workflow Overview

1. **Data Ingestion**: Downloads raw data into `data/raw/` using `scripts/get_data.py`.
2. **Preprocessing**: Cleans and splits data into train/test sets via `src/preprocess.py`.
3. **Model Training**: Trains and tunes models (Linear Regression, Decision Tree, Random Forest, SVR) with GridSearchCV in `src/train.py`.
4. **Model Registration**: Registers the best-performing model in MLflow using `src/register_best_model.py`.
5. **Experiment Tracking**: All metrics, parameters, and models are logged to MLflow, with remote tracking on DagsHub.
6. **CI/CD Automation**: The pipeline is automatically executed on every push to `main` via GitHub Actions.

## 🤖 Models & Hyperparameters

The following models and their hyperparameter grids are defined in `src/config.py`:
- Linear Regression
- Decision Tree
- Random Forest
- Support Vector Regressor (SVR)

Each model is tuned using a grid search for optimal performance.

## 📊 MLflow & DagsHub Dashboard

- MLflow tracks experiments locally in `mlruns/` and remotely on DagsHub.
- Set your DagsHub token as a GitHub secret (`DAGSHUB_TOKEN`) for CI/CD.
- **DagsHub MLflow Dashboard:** [View Experiments & Models](https://dagshub.com/Ujjwal1011/-Bike-Sharing-Demand-Prediction.mlflow)

## 🚀 Getting Started

1. **Install dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Run the pipeline locally:**
   ```powershell
   python scripts/get_data.py
   python src/preprocess.py
   python src/train.py
   python src/register_best_model.py
   ```
3. **CI/CD:** On push to `main`, GitHub Actions will execute the full pipeline and upload MLflow artifacts.

## 🛠️ Configuration
- Model and grid search configs: `src/config.py`
- Example grid config for image classification (template): `config/grid_config.yaml`

## 📚 Data
- Source: [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)


