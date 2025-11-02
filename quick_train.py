"""quick_train.py — Training + log & register model for MLflow v2.9.2"""
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# CONFIG
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "cifar10_production"
MODEL_NAME = "CIFAR10_ResNet18"
LOCAL_MODEL_DIR = Path("models")
LOCAL_MODEL_FILE = LOCAL_MODEL_DIR / "best_model.pth"
EPOCHS = 5
BATCH_SIZE = 32
LR = 0.001

def create_model_instance():
    # Adjust to your project create_model function import path
    from src.training.model import create_model
    return create_model(num_classes=10, weights=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Data (small subset for quick run)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data_", train=True, download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainset, range(1000))
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    model = create_model_instance().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training + log to MLflow
    with mlflow.start_run(run_name="quick_test_run") as run:
        run_id = run.info.run_id

        # params
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("lr", LR)
        mlflow.set_tag("model", "ResNet-18")

        print("\n🚀 Training...")
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / len(trainloader)
            epoch_acc = 100. * correct / total
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

        # Save local checkpoint (optional but useful)
        LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, LOCAL_MODEL_FILE)
        print(f"\n💾 Saved model to {LOCAL_MODEL_FILE}")

        # Log model to MLflow artifact_path="model"
        # This will create run artifact folder: <run_id>/artifacts/model/...
        print("📦 Logging model to MLflow (artifact_path='model') ...")
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model")

        # Also log the local checkpoint file as an artifact (redundant but explicit)
        mlflow.log_artifact(str(LOCAL_MODEL_FILE), artifact_path="model")

        print(f"✅ Run logged! ID: {run_id}")
        print(f"🏃 View run at: {MLFLOW_TRACKING_URI}/#/experiments/1/runs/{run_id}")

    # Register model in registry (v2.9.2 style)
    print("\n🚀 Registering model to Model Registry...")
    try:
        client.create_registered_model(MODEL_NAME)
        print(f"✅ Created registered model {MODEL_NAME}")
    except Exception:
        print(f"Model {MODEL_NAME} may already exist, continuing...")

    # The source must point to the artifact path created by log_model: runs:/{run_id}/model
    source = f"runs:/{run_id}/model"
    mv = client.create_model_version(
        name=MODEL_NAME,
        source=source,
        run_id=run_id
    )
    print(f"✅ Created version {mv.version} (waiting for creation)...")

    # Wait for the model version to be created (client has built-in wait in v2.9.2)
    # Transition to Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production"
    )
    print(f"✅ Version {mv.version} → Production!")

    print("\n🎉 Done. Check MLflow UI at:", MLFLOW_TRACKING_URI)

if __name__ == "__main__":
    main()
