# train.py

"""
Contains functions for training and evaluating the deep learning model.

This module includes:
- evaluate_model: Computes loss and MAPE on a validation set.
- train_model: The main training loop with optimization, gradient scaling,
  early stopping, and learning rate scheduling.
- save_artifacts: Saves the trained model, feature processor, and config to a file.
"""

import time
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from tqdm.notebook import tqdm

from app.src.model import AdditiveAxisModel
from app.src.data_processing import load_and_split_data, FeatureProcessor, create_dataloaders
from typing import Tuple, Dict


def evaluate_model(model: AdditiveAxisModel, data_loader: torch.utils.data.DataLoader, device: str) -> Tuple[float, float]:
    """
    Calculates validation loss (MSE) and Mean Absolute Percentage Error (MAPE).

    Args:
        model (AdditiveAxisModel): The model to evaluate.
        data_loader (DataLoader): The DataLoader for the validation set.
        device (str): The device to run evaluation on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the average validation MSE loss and MAPE.
    """
    model.eval()
    total_loss, total_mape = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                else: batch[k] = {sk: sv.to(device) for sk, sv in v.items()}

            targets_price = batch['target_price']
            targets_log_dev = batch['target_log_deviation']

            # Use AMP for faster inference if on CUDA
            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                preds_log_dev = model(batch)
                loss = torch.mean((preds_log_dev - targets_log_dev).float().pow(2))

                predicted_log_price = preds_log_dev + batch['neighborhood_log_mean']
                price_preds = torch.expm1(predicted_log_price)
                mape = (torch.abs(price_preds - targets_price) / (targets_price + 1e-6)).mean()

            total_loss += loss.item()
            total_mape += mape.item()
    return total_loss / len(data_loader), total_mape / len(data_loader)


def train_model(model: AdditiveAxisModel, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, config: dict) -> Tuple[AdditiveAxisModel, pd.DataFrame]:
    """
    Main function to train the model, with early stopping based on validation MAPE.

    Args:
        model (AdditiveAxisModel): The instantiated model to be trained.
        train_loader (DataLoader): The DataLoader for the training set.
        val_loader (DataLoader): The DataLoader for the validation set.
        optimizer (Optimizer): The PyTorch optimizer.
        scheduler (LRScheduler): The learning rate scheduler.
        config (dict): The global configuration dictionary.

    Returns:
        tuple: A tuple containing:
            - AdditiveAxisModel: The best performing trained model.
            - pd.DataFrame: A history of training and validation metrics.
    """
    print("\n--- Starting Model Training ---")
    history, best_val_mape = [], float('inf')
    best_model_state, patience_counter = None, 0
    scaler = torch.amp.GradScaler(enabled=(config['DEVICE'] == "cuda"))
    start_time = time.time()

    header = f"{'Epoch':>5} | {'Time':>8} | {'Train RMSE':>12} | {'Val RMSE':>10} | {'Val MAPE (%)':>12} | {'Patience':>8}"
    print(header); print("-" * len(header))

    for epoch in range(config['N_EPOCHS']):
        model.train()
        train_loss_epoch = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['N_EPOCHS']}", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(config['DEVICE'])
                else: batch[k] = {sk: sv.to(config['DEVICE']) for sk, sv in v.items()}

            with torch.amp.autocast(device_type=config['DEVICE'], dtype=torch.float16, enabled=(config['DEVICE']=="cuda")):
                preds_log_dev = model(batch)
                loss = torch.mean((preds_log_dev - batch["target_log_deviation"]).float().pow(2))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_epoch += loss.item()

        val_mse, val_mape = evaluate_model(model, val_loader, config['DEVICE'])
        train_rmse, val_rmse = np.sqrt(train_loss_epoch / len(train_loader)), np.sqrt(val_mse)
        elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

        if val_mape < best_val_mape - config['EARLY_STOPPING_MIN_DELTA']:
            best_val_mape, patience_counter = val_mape, 0
            best_model_state = model.state_dict()
            # The "INFO" print statement that was here has been removed.
        else:
            patience_counter += 1

        print(f"{epoch+1:>5} | {elapsed_time:>8} | {train_rmse:>12.4f} | {val_rmse:>10.4f} | {val_mape*100:>12.2f} | {patience_counter:>8}")
        history.append({'epoch': epoch, 'train_rmse': train_rmse, 'val_rmse': val_rmse, 'val_mape': val_mape})
        scheduler.step(val_mape)

        if patience_counter >= config['EARLY_STOPPING_PATIENCE']:
            print(f"--- Early Stopping Triggered (MAPE did not improve for {patience_counter} epochs) ---"); break

    print("\n--- Training Complete ---")
    if best_model_state:
        print(f"Loading best model state with Val MAPE: {best_val_mape*100:.2f}%")
        model.load_state_dict(best_model_state)
    return model, pd.DataFrame(history)


def save_artifacts(model, processor, config: dict) -> str:
    """
    Saves the trained model, feature processor, and config to a file.

    This function creates a single .pt file containing all the necessary objects
    to run inference or continue training later.

    Args:
        model (AdditiveAxisModel): The trained PyTorch model.
        processor (FeatureProcessor): The fitted FeatureProcessor instance.
        config (dict): The global configuration dictionary.

    Returns:
        str: The file path where the artifacts were saved.
    """
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{config['CITY']}_model_artifacts_{timestamp}.pt"
    save_path = os.path.join(config['DRIVE_SAVE_PATH'], filename)
    os.makedirs(config['DRIVE_SAVE_PATH'], exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_processor': processor,
        'config': config
    }, save_path)
    
    print(f"\nArtifacts successfully saved to: {save_path}")
    return save_path