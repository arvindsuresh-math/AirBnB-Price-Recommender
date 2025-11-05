"""
Contains functions for generating predictions from trained models.
"""
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def run_inference(model, data_loader, device: str) -> pd.DataFrame:
    """Runs inference for a simple model and returns predictions."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Inference"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                else: batch[k] = {sk: sv.to(device) for sk, sv in v.items()}
            
            preds_log_dev = model(batch)
            price_preds = torch.expm1(preds_log_dev + batch['neighborhood_log_mean'])
            all_preds.append(price_preds.cpu().numpy())
    
    return pd.DataFrame({'predicted_price': np.concatenate(all_preds)})


def run_inference_with_details(model, data_loader, device: str) -> dict:
    """Runs inference for the AdditiveModel, returning all detailed outputs."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Detailed Inference"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                else: batch[k] = {sk: sv.to(device) for sk, sv in v.items()}

            batch_outputs = model(batch, return_details=True)
            batch_outputs['neighborhood_log_mean'] = batch['neighborhood_log_mean']
            outputs.append({k: v.cpu() for k, v in batch_outputs.items()})

    final_outputs = {key: torch.cat([o[key] for o in outputs]).numpy() for key in outputs[0].keys()}
    
    predicted_log = final_outputs['predicted_log_deviation'] + final_outputs['neighborhood_log_mean']
    final_outputs['predicted_price'] = np.expm1(predicted_log)
    return final_outputs