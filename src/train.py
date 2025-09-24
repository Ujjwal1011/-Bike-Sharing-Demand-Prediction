import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from config import MODEL_CONFIG
from mlflow.models.signature import infer_signature
import dagshub

def train_models():
    """
    Trains multiple models using a parent-child run structure in MLflow.
    Each model type has a parent run for tuning.
    Each hyperparameter combination is a nested child run.
    """

    dagshub.init(repo_owner='Ujjwal1011', repo_name='-Bike-Sharing-Demand-Prediction', mlflow=True)
    # mlflow.set_tracking_uri("https://dagshub.com/Ujjwal1011/-Bike-Sharing-Demand-Prediction.mlflow")

    # Load processed data
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop("cnt", axis=1)
    y_train = train_df["cnt"]
    X_test = test_df.drop("cnt", axis=1)
    y_test = test_df["cnt"]

    EXPERIMENT_NAME = "Bike Sharing Demand Prediction"
    mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name, config in MODEL_CONFIG.items():
        # Start a parent run for each model type
        with mlflow.start_run(run_name=f"Parent_{model_name}"):
            print(f"--- Starting Parent Run for {model_name} ---")
            
            best_score = float('inf')
            best_params = None

            # Get the parameter grid for the current model
            param_grid = list(ParameterGrid(config["params"]))

            # --- Child Runs for Hyperparameter Tuning ---
            for i, params in enumerate(param_grid):
                # Start a nested child run for each parameter combination
                with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                    print(f"  - Starting child run {i+1}/{len(param_grid)} with params: {params}")
                    
                    # Log the parameters for this trial
                    mlflow.log_params(params)
                    
                    # Create and train the model with the current parameters
                    model = config["model"].set_params(**params)
                    
                    # Use cross-validation for robust evaluation
                    scores = cross_val_score(model, X_train, y_train, 
                                             cv=3, scoring='neg_root_mean_squared_error')
                    
                    # Calculate mean RMSE (score is negative, so we negate it)
                    mean_rmse = -np.mean(scores)
                    mlflow.log_metric("cv_rmse", mean_rmse)
                    
                    # Keep track of the best score and params
                    if mean_rmse < best_score:
                        best_score = mean_rmse
                        best_params = params

            # --- Back in the Parent Run ---
            print(f"Best CV RMSE for {model_name}: {best_score}")
            print(f"Best Parameters: {best_params}")

            # Log the best parameters and score to the parent run
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_rmse", best_score)
            
            # Train the final best model on the full training data
            final_model = config["model"].set_params(**best_params)
            final_model.fit(X_train, y_train)

            # Evaluate on the hold-out test set
            y_pred = final_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_r2 = r2_score(y_test, y_pred)
            
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log the final model with a signature
            input_example = X_train.head(5)
            signature = infer_signature(input_example, final_model.predict(input_example))
            
            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example
            )
            print(f"--- Parent Run for {model_name} complete. ---")

if __name__ == "__main__":
    train_models()