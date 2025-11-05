"""
Builds the final, self-contained dataset for the Streamlit application.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data_processing import FeatureProcessor
from model import AdditiveModel, AirbnbPriceDataset
from inference import run_inference_with_details

def create_full_panel_dataset(df: pd.DataFrame, train_ids: set, val_ids: set) -> pd.DataFrame:
    """Augments the dataset to create a full panel for every listing and every month."""
    static_cols = [
        'id', 'host_id', 'name', 'description', 'host_is_superhost',
        'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type',
        'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'amenities', 'total_reviews'
    ]
    unique_listings = df.drop_duplicates(subset=['id'], keep='first')[static_cols]
    
    months_df = pd.DataFrame({'month': range(1, 13)})
    panel_df = unique_listings.merge(months_df, how='cross')
    panel_df = panel_df.merge(df[['id', 'month', 'price']], on=['id', 'month'], how='left')
    panel_df['split'] = panel_df['id'].apply(lambda x: 'train' if x in train_ids else 'val')
    
    review_cols = [col for col in df.columns if 'review_scores' in col]
    for col in review_cols:
        mean_score = df[col].mean()
        panel_df[col] = panel_df['id'].map(df.groupby('id')[col].first()).fillna(mean_score)

    return panel_df

def build_dataset(model: AdditiveModel, processor: FeatureProcessor, config: dict, train_ids: set, val_ids: set):
    """Orchestrates the creation of the final application dataset."""
    # 1. Load raw data and create the full panel
    raw_df = pd.read_parquet(f"./{config['CITY']}_dataset_oct_20.parquet")
    panel_df = create_full_panel_dataset(raw_df, train_ids, val_ids)
    
    # 2. Process features for the entire panel
    train_df_for_means = raw_df[raw_df['id'].isin(train_ids)]
    neighborhood_log_means = np.log1p(train_df_for_means.groupby('neighbourhood_cleansed')['price'].mean()).to_dict()
    panel_features = processor.transform(panel_df, neighborhood_log_means)

    # 3. Create a DataLoader for the panel
    tokenizer = AutoTokenizer.from_pretrained(config['TEXT_MODEL_NAME'])
    panel_dataset = AirbnbPriceDataset(panel_features, tokenizer)
    panel_loader = DataLoader(
        panel_dataset, batch_size=config['VALIDATION_BATCH_SIZE'],
        shuffle=False, num_workers=2, pin_memory=True
    )

    # 4. Run inference to get all predictions and hidden states
    details_outputs = run_inference_with_details(model, panel_loader, config['DEVICE'])
    
    # 5. Combine raw data with the new detailed outputs
    app_df = panel_df.copy()
    for key, value in details_outputs.items():
        if len(value) == len(app_df):
            app_df[key] = list(value)
    
    # 6. Calculate multiplicative contributions
    p_cols = [col for col in app_df.columns if col.startswith('p_') and col != 'predicted_price']
    for col in p_cols:
        app_df[col.replace('p_', 'pm_')] = np.exp(app_df[col])
    
    # 7. Finalize columns and save
    app_df.rename(columns={'neighborhood_log_mean': 'p_base_log'}, inplace=True)
    output_path = os.path.join(config['DRIVE_SAVE_PATH'], 'app_data')
    os.makedirs(output_path, exist_ok=True)
    output_filename = os.path.join(output_path, f"{config['CITY']}_app_database.parquet")
    
    app_df.to_parquet(output_filename, index=False)
    print(f"\nSuccessfully created application database at: {output_filename}")