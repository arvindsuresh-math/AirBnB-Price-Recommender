"""
Contains functions for training and evaluating the deep learning models.
"""

import time
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple
import os

def evaluate_model(model, data_loader, device: str) -> Tuple[float, float]:
    """Calculates validation loss (MSE) and Mean Absolute Percentage Error (MAPE)."""
    model.eval()
    total_loss, total_mape = 0.0, 0.0
    with torch.no_grad():
        for batch in data_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                else: batch[k] = {sk: sv.to(device) for sk, sv in v.items()}

            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                preds_log_dev = model(batch)
                loss = torch.mean((preds_log_dev - batch['target_log_deviation']).pow(2))
                price_preds = torch.expm1(preds_log_dev + batch['neighborhood_log_mean'])
                mape = (torch.abs(price_preds - batch['target_price']) / (batch['target_price'] + 1e-6)).mean()

            total_loss += loss.item()
            total_mape += mape.item()
    return total_loss / len(data_loader), total_mape / len(data_loader)

def train_model(model, train_loader, val_loader, optimizer, scheduler, config: dict):
    """
    Main function to train a model with a custom early stopping criterion.

    The "best" model is defined by two conditions:
    1. Constraint: The MAPE Gap (Val MAPE - Train MAPE) must be less than 4%.
    2. Optimization: Among all models satisfying the constraint, find the one
       with the lowest Train MAPE.

    Patience-based early stopping only begins once the 4% gap threshold is breached.
    """
    history = []
    # --- CHANGE 1: Track the best valid train MAPE and its corresponding gap ---
    best_valid_train_mape = float('inf')
    best_model_gap = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler(enabled=(config['DEVICE'] == "cuda"))
    start_time = time.time()
    temp_model_path = "temp_best_model.pt"

    print("\n--- Starting Model Training (Optimizing for Min Train MAPE with <4% Gap) ---")
    header = f"{'Epoch':>5} | {'Time':>8} | {'Train RMSE':>12} | {'Train MAPE (%)':>14} | {'Val RMSE':>10} | {'Val MAPE (%)':>12} | {'MAPE Gap (%)':>12} | {'Patience':>8}"
    print(header); print("-" * len(header))

    for epoch in range(config['N_EPOCHS']):
        model.train()
        train_loss_epoch, train_mape_epoch = 0.0, 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['N_EPOCHS']}", leave=False)
        for batch in progress_bar:
            # ... (forward pass logic is the same) ...
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(config['DEVICE'])
                else: batch[k] = {sk: sv.to(config['DEVICE']) for sk, sv in v.items()}
            with torch.amp.autocast(device_type=config['DEVICE'], dtype=torch.float16, enabled=(config['DEVICE']=="cuda")):
                preds_log_dev = model(batch)
                loss = torch.mean((preds_log_dev - batch["target_log_deviation"]).pow(2))
                price_preds = torch.expm1(preds_log_dev + batch['neighborhood_log_mean'])
                mape = (torch.abs(price_preds - batch['target_price']) / (batch['target_price'] + 1e-6)).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_epoch += loss.item()
            train_mape_epoch += mape.item()

        val_mse, val_mape = evaluate_model(model, val_loader, config['DEVICE'])
        train_rmse = np.sqrt(train_loss_epoch / len(train_loader))
        train_mape = train_mape_epoch / len(train_loader)
        val_rmse = np.sqrt(val_mse)
        elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        mape_gap = val_mape - train_mape
        
        # --- CHANGE 2: Implement the new two-part model selection logic ---
        # Condition 1: Check if the model is valid (gap is below threshold)
        if mape_gap < 0.04:
            # If valid, reset patience
            patience_counter = 0
            
            # Condition 2: Check if this valid model is the best one so far
            if train_mape < best_valid_train_mape:
                best_valid_train_mape = train_mape
                best_model_gap = mape_gap
                torch.save(model.state_dict(), temp_model_path)
        else:
            # If the model is invalid (gap is too high), start or increment patience
            patience_counter += 1

        print(f"{epoch+1:>5} | {elapsed_time:>8} | {train_rmse:>12.4f} | {train_mape*100:>14.2f} | {val_rmse:>10.4f} | {val_mape*100:>12.2f} | {mape_gap*100:>12.2f} | {patience_counter:>8}")
        
        history.append({
            'epoch': epoch, 'train_rmse': train_rmse, 'train_mape': train_mape,
            'val_rmse': val_rmse, 'val_mape': val_mape, 'mape_gap': mape_gap
        })
        
        scheduler.step(val_mape)

        # Early stopping is now based only on patience
        if patience_counter >= config['EARLY_STOPPING_PATIENCE']:
            print(f"--- Early Stopping Triggered (MAPE Gap exceeded 4% for {patience_counter} epochs) ---")
            break

    print("\n--- Training Complete ---")
    if os.path.exists(temp_model_path):
        # --- CHANGE 3: The final printout now reports the metrics of the best valid model ---
        print(f"Loading best model state from file with Train MAPE: {best_valid_train_mape*100:.2f}% (and MAPE Gap: {best_model_gap*100:.2f}%)")
        model.load_state_dict(torch.load(temp_model_path))
        os.remove(temp_model_path)
    else:
        print("Warning: No model was saved. The MAPE gap may have exceeded 4% on the first epoch.")
        
    return model, pd.DataFrame(history)


def run_ablation_experiment(exclude_axes: list, config: dict, processor, train_loader, val_loader):
    """High-level wrapper to run a single ablation study experiment."""
    from model import AblationAdditiveModel
    
    print("\n" + "="*70)
    print(f"  STARTING ABLATION EXPERIMENT: EXCLUDING {exclude_axes}")
    print("="*70)

    model = AblationAdditiveModel(processor, config, exclude_axes=exclude_axes)
    model.to(config['DEVICE'])
    
    transformer_params = model.text_transformer.parameters()
    other_params = [p for n, p in model.named_parameters() if 'text_transformer' not in n]
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': config['LEARNING_RATE'], 'weight_decay': config['WEIGHT_DECAY']},
        {'params': transformer_params, 'lr': config['TRANSFORMER_LEARNING_RATE']}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['SCHEDULER_FACTOR'], patience=config['SCHEDULER_PATIENCE']
    )

    trained_model, history = train_model(model, train_loader, val_loader, optimizer, scheduler, config)
    
    # Use the final history record to get metrics, which is more efficient
    final_metrics = history.iloc[-1].to_dict()
    
    return {
        "excluded_axes": str(exclude_axes or "['None (Baseline)']"),
        "train_rmse": final_metrics['train_rmse'], 
        "val_rmse": final_metrics['val_rmse'],
        "train_mape": final_metrics['train_mape'], 
        "val_mape": final_metrics['val_mape']
    }