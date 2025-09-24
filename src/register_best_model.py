import mlflow

def register_best_model():
    """
    Finds the best PARENT run from an experiment and registers its model.
    """
    EXPERIMENT_NAME = "Bike Sharing Demand Prediction"
    MODEL_REGISTRY_NAME = "BikeSharingPredictor"

    client = mlflow.tracking.MlflowClient()
    
    try:
        experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    except AttributeError:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return

    # Search only for parent runs (child runs have a 'mlflow.parentRunId' tag)
    # We want runs that DO NOT have this tag.
    query = "tags.'mlflow.parentRunId' IS NULL"
    
    # Search for the best parent run based on the test set RMSE
    best_run = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        order_by=["metrics.test_rmse ASC"],
        max_results=1
    )[0]

    best_run_id = best_run.info.run_id
    best_rmse = best_run.data.metrics["test_rmse"]
    
    # The artifact path is now the model name itself (e.g., 'RandomForest')
    model_name_from_run = best_run.data.tags['mlflow.runName'].replace('Parent_', '')
    model_uri = f"runs:/{best_run_id}/{model_name_from_run}"

    print(f"Best Parent Run Found: {best_run_id}")
    print(f"  Model: {model_name_from_run}")
    print(f"  Test RMSE: {best_rmse}")
    print(f"  Model URI: {model_uri}")
    
    print(f"Registering model as '{MODEL_REGISTRY_NAME}'...")
    model_version = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
    print(f"Model registered successfully. Version: {model_version.version}")

    client.transition_model_version_stage(
        name=MODEL_REGISTRY_NAME,
        version=model_version.version,
        stage="Staging"
    )
    print(f"Model version {model_version.version} transitioned to 'Staging'.")


if __name__ == "__main__":
    register_best_model()