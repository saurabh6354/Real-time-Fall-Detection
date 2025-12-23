"""
train.py - Training Script for Patient Safety LSTM

This script handles the complete training pipeline:
1. Data loading (real or mock data)
2. Model initialization with proper weight init
3. Training loop with learning rate scheduling
4. Validation and early stopping
5. Model checkpointing

Usage:
    python train.py                      # Train with mock data (default)
    python train.py --use-real-data      # Train with preprocessed data
    python train.py --epochs 100         # Custom number of epochs
    python train.py --resume checkpoint.pth  # Resume training
"""

import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, Optional
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config
from model import PatientSafetyLSTM, get_model_summary, count_parameters
from dataset import get_data_loaders, MockDataset


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training if it doesn't improve
    for a specified number of epochs (patience).
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to reset counter
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: The LSTM model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
    
    for batch_x, batch_y in pbar:
        # Move to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.1f}%'
        })
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[str, float]]:
    """
    Validate the model.
    
    Args:
        model: The LSTM model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, accuracy, per_class_metrics)
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class tracking
    class_correct = {i: 0 for i in range(config.NUM_CLASSES)}
    class_total = {i: 0 for i in range(config.NUM_CLASSES)}
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            # Per-class accuracy
            for label, pred in zip(batch_y, predicted):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Calculate per-class metrics
    per_class = {}
    for i in range(config.NUM_CLASSES):
        class_name = config.CLASS_LABELS[i]
        if class_total[i] > 0:
            per_class[class_name] = class_correct[i] / class_total[i]
        else:
            per_class[class_name] = 0.0
    
    return avg_loss, accuracy, per_class


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    filepath: Path
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': {
            'input_size': config.INPUT_SIZE,
            'hidden_size_1': config.HIDDEN_SIZE_1,
            'hidden_size_2': config.HIDDEN_SIZE_2,
            'num_classes': config.NUM_CLASSES,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"  ✓ Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None
) -> int:
    """Load model checkpoint and return the epoch."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"  ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = config.NUM_EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    device: Optional[torch.device] = None,
    save_dir: Path = config.MODELS_DIR,
    resume_from: Optional[Path] = None
) -> Dict[str, list]:
    """
    Full training loop with validation and checkpointing.
    
    Args:
        model: The LSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Initial learning rate
        device: Device to train on (default: auto-detect)
        save_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nTraining on: {device}")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and resume_from.exists():
        start_epoch = load_checkpoint(resume_from, model, optimizer)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, per_class = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.1f}%")
        print(f"  Per-class: {' | '.join([f'{k}: {v*100:.0f}%' for k, v in per_class.items()])}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                save_dir / "best_model.pth"
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            )
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pth'}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train the Patient Safety LSTM")
    parser.add_argument(
        '--epochs', type=int, default=config.NUM_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=config.BATCH_SIZE,
        help='Batch size'
    )
    parser.add_argument(
        '--lr', type=float, default=config.LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--use-real-data', action='store_true',
        help='Use real preprocessed data instead of mock data'
    )
    parser.add_argument(
        '--resume', type=Path, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--no-cuda', action='store_true',
        help='Disable CUDA even if available'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PATIENT SAFETY LSTM TRAINING")
    print("=" * 60)
    
    # Determine device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = PatientSafetyLSTM()
    print(get_model_summary(model, (config.SEQUENCE_LENGTH, config.INPUT_SIZE)))
    
    # Get data loaders
    use_mock = not args.use_real_data
    if use_mock:
        print("Using MOCK dataset for training (no real data)")
    
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        use_mock=use_mock
    )
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        resume_from=args.resume
    )
    
    print("\nTraining complete! Use inference.py to run real-time detection.")


if __name__ == "__main__":
    main()
