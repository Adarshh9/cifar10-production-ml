"""Training script for CIFAR-10 ResNet model."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import time

from src.data.data_loader import CIFAR10DataModule
from src.training.model import create_model
from src.training.mlflow_utils import MLflowTracker
from src.training.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training manager with all production features."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: str,
        config: dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        self.use_amp = config.get('use_amp', True)
        
        # Tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {
            'train_loss': epoch_loss,
            'train_accuracy': epoch_acc
        }
    
    def validate(self, loader) -> dict:
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(loader)
        val_acc = 100. * correct / total
        
        return {
            'val_loss': val_loss,
            'val_accuracy': val_acc
        }
    
    def train(self, tracker: MLflowTracker):
        """Full training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            current_lr = self.scheduler.get_last_lr()[0]
            all_metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': float(current_lr)
            }
            tracker.log_metrics(all_metrics, step=epoch)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )
            
            # Save best model
            if val_metrics['val_accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_accuracy']
                self.save_checkpoint('models/best_model.pth')
                logger.info(f"✅ New best model! Val Acc: {self.best_val_acc:.2f}%")
        
        # Final test evaluation
        logger.info("\n" + "="*50)
        logger.info("Final Test Evaluation")
        logger.info("="*50)
        test_metrics = self.validate(self.test_loader)
        tracker.log_metrics(test_metrics)
        logger.info(f"🎯 Test Accuracy: {test_metrics['val_accuracy']:.2f}%")
        
        return test_metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)


def main(args):
    """Main training function."""
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Configuration
    config = {
        'model_name': 'CIFAR10_ResNet18',
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'use_amp': True,
        'num_classes': 10,
        'mlflow_uri': 'http://localhost:5000',
        'experiment_name': 'cifar10_production'
    }
    
    # Data
    logger.info("Loading CIFAR-10 dataset...")
    data_module = CIFAR10DataModule(
        data_dir='./data',
        batch_size=config['batch_size'],
        num_workers=4
    )
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    # Model
    logger.info("Creating ResNet-18 model...")
    model = create_model(num_classes=10, pretrained=args.pretrained)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # MLflow tracking
    tracker = MLflowTracker(
        tracking_uri=config['mlflow_uri'],
        experiment_name=config['experiment_name']
    )
    
    with tracker.start_run(run_name=f"resnet18_cifar10_epoch{args.epochs}"):
        # Log config
        tracker.log_params(config)
        tracker.set_tags({
            'model': 'ResNet-18',
            'dataset': 'CIFAR-10',
            'framework': 'pytorch'
        })
        
        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            config=config
        )
        
        start_time = time.time()
        test_metrics = trainer.train(tracker)
        training_time = time.time() - start_time
        
        logger.info(f"\n✅ Training completed in {training_time/60:.2f} minutes")
        
        # Log model to MLflow
        tracker.log_model(
            model,
            artifact_path="model",
            registered_model_name=config['model_name']
        )
        
        run_id = tracker.run_id
    
    # Register to production if accuracy is good
    if test_metrics['val_accuracy'] > 80.0:
        logger.info("\n🚀 Registering model to Production...")
        registry = ModelRegistry(config['mlflow_uri'])
        version = registry.register_model(run_id, config['model_name'])
        registry.transition_model_stage(config['model_name'], version, 'Production')
        logger.info(f"✅ Model v{version} promoted to Production!")
    else:
        logger.warning(f"⚠️ Test accuracy {test_metrics['val_accuracy']:.2f}% too low for production")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CIFAR-10 ResNet-18')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    args = parser.parse_args()
    main(args)
