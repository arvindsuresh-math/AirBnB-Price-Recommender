# build_app_dataset.py

"""
Builds the final, self-contained dataset for the Streamlit application.

This script loads a trained model, creates a full panel of all listings for all
months, runs inference to get predictions and all intermediate model outputs,
calculates multiplicative contributions, and saves everything to a single
Parquet file.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from app.src.data_processing import create_full_panel_dataset, FeatureProcessor
from app.src.model import AirbnbPriceDataset
from app.src.inference import run_inference_with_details
from typing import Dict, Any


def build_dataset(artifacts: Dict[str, Any]):
    """
    Orchestrates the creation of the final application dataset.

    Args:
        artifacts (dict): A dictionary loaded from the training pipeline,
            containing the trained 'model', 'processor', 'train_ids', and 'val_ids'.
    """
    model = artifacts['model']
    processor = artifacts['processor']
    config = processor.config  # Config is stored inside processor
    train_ids = artifacts['train_ids']
    val_ids = artifacts['val_ids']

    # 1. Load the raw dataset to be augmented
    dataset_filename = f"{config['CITY']}_dataset_oct_20.parquet"
    raw_df = pd.read_parquet(f"./{dataset_filename}")
    raw_df = raw_df[raw_df["price"] > 0].copy()

    # 2. Create the augmented panel dataframe
    panel_df = create_full_panel_dataset(raw_df, train_ids, val_ids)

    # 3. Process features for the entire panel
    # The neighborhood means were learned on the original train set and are part of the fitted processor
    neighborhood_log_means = np.log1p(raw_df[raw_df.id.isin(train_ids)].groupby('neighbourhood_cleansed')['price'].mean()).to_dict()
    panel_features = processor.transform(panel_df, neighborhood_log_means)

    # 4. Create a DataLoader for the panel
    tokenizer = AutoTokenizer.from_pretrained(config['TEXT_MODEL_NAME'])
    panel_dataset = AirbnbPriceDataset(panel_features, tokenizer)
    panel_loader = DataLoader(
        panel_dataset,
        batch_size=config['VALIDATION_BATCH_SIZE'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 5. Run inference to get all predictions and hidden states
    print(f"Running inference on {len(panel_df):,} augmented listings...")
    details_outputs = run_inference_with_details(model, panel_loader, config['DEVICE'])

    # 6. Combine raw data with the new detailed outputs
    app_database_df = panel_df.copy()
    for key, value in details_outputs.items():
        if len(value) == len(app_database_df):
            app_database_df[key] = list(value)
        else:
            print(f"Warning: Length mismatch for key '{key}'. Expected {len(app_database_df)}, got {len(value)}. Skipping.")
    
    # 7. Calculate multiplicative contributions
    p_cols = [col for col in app_database_df.columns if col.startswith('p_') and col != 'predicted_price']
    for col in p_cols:
        multiplicative_col_name = col.replace('p_', 'pm_')
        # The multiplicative factor is simply e^(log-deviation)
        app_database_df[multiplicative_col_name] = np.exp(app_database_df[col])

    # 8. Finalize columns and save
    app_database_df.rename(columns={'neighborhood_log_mean': 'p_base_log'}, inplace=True)
    output_path = os.path.join(config['DRIVE_SAVE_PATH'], 'app_data')
    os.makedirs(output_path, exist_ok=True)
    output_filename = os.path.join(output_path, f"{config['CITY']}_app_database.parquet")
    
    app_database_df.to_parquet(output_filename, index=False)
    print(f"\nSuccessfully created application database for {config['CITY'].upper()}!")
    print(f"Saved to: {output_filename}")