import mlflow
# import dagshub

def register_best_model():
    """
    Finds the best PARENT run from an experiment and registers its model.
    """
    # Use your DagsHub username and repo name
    # dagshub.init(repo_owner='Ujjwal1011', repo_name='-Bike-Sharing-Demand-Prediction', mlflow=True)
    
    MODEL_REGISTRY_NAME = "BikeSharingPredictor"

    client = mlflow.tracking.MlflowClient()
    
    # Use the experiment name logged by the training script
    experiment_name = "Bike Sharing Demand Prediction"
    
    try:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    except AttributeError:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # --- THIS IS THE FIX ---
    # Search for runs where the name starts with "Parent_"
    query = "tags.'mlflow.runName' LIKE 'Parent_%'"
    
    # Search for the best parent run based on the test set RMSE
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=query,
        order_by=["metrics.test_rmse ASC"],
        max_results=1
    )

    if not runs:
        print("No parent runs found. Make sure the training script has completed successfully.")
        return

    best_run = runs[0]

    best_run_id = best_run.info.run_id
    best_rmse = best_run.data.metrics["test_rmse"]
    
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