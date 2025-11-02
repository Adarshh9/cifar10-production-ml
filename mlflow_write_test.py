"""Test if MLflow can write artifacts."""
import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test")

with mlflow.start_run():
    # Write test file
    test_file = "test.txt"
    with open(test_file, "w") as f:
        f.write("Hello MLflow!")
    
    # Log it
    mlflow.log_artifact(test_file)
    print("✅ Artifact logged!")
    
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    print(f"Check: http://localhost:5000/#/experiments/0/runs/{run_id}")
