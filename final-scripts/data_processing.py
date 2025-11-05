"""
Handles all data loading, splitting, and feature engineering for the training pipeline.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import Tuple, Dict, Set

from model import AirbnbPriceDataset

def load_and_split_data(config: dict, n_strata_bins: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Set[int], Set[int]]:
    """
    Loads the dataset and performs a stratified, group-based train/validation split.

    This function ensures that all records for a given listing_id are in the same
    set (train or val). Stratification is based on the average log-price deviation
    for each listing to ensure representative validation set.
    """
    dataset_filename = f"{config['CITY']}_dataset_oct_20.parquet"
    dataset_path = f"./{dataset_filename}"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"'{dataset_filename}' not found. Please provide the file.")

    df = pd.read_parquet(dataset_path)
    df = df[df["price"] > 0].copy()

    # --- Stratification Logic ---
    temp_neighborhood_means = np.log1p(df.groupby('neighbourhood_cleansed')['price'].mean()).to_dict()
    df['temp_log_mean'] = df['neighbourhood_cleansed'].map(temp_neighborhood_means)
    df['log_price_deviation'] = np.log1p(df['price']) - df['temp_log_mean']
    listing_avg_deviation = df.groupby('id')['log_price_deviation'].mean().to_frame()

    listing_avg_deviation['strata_bin'] = pd.qcut(
        listing_avg_deviation['log_price_deviation'],
        q=n_strata_bins, labels=False, duplicates='drop'
    )

    listing_ids = listing_avg_deviation.index
    strata = listing_avg_deviation['strata_bin']
    train_ids, val_ids = train_test_split(
        listing_ids, test_size=config['VAL_SIZE'], random_state=config['SEED'], stratify=strata
    )
    train_ids_set, val_ids_set = set(train_ids), set(val_ids)

    train_df = df[df['id'].isin(train_ids_set)].drop(columns=['temp_log_mean', 'log_price_deviation']).reset_index(drop=True)
    val_df = df[df['id'].isin(val_ids_set)].drop(columns=['temp_log_mean', 'log_price_deviation']).reset_index(drop=True)

    final_neighborhood_log_means = np.log1p(train_df.groupby('neighbourhood_cleansed')['price'].mean()).to_dict()

    print(f"Data split: {len(train_df):,} train records, {len(val_df):,} validation records.")
    return train_df, val_df, final_neighborhood_log_means, train_ids_set, val_ids_set


class FeatureProcessor:
    """
    Prepares raw DataFrame columns into numerical features for the model.
    """
    def __init__(self, config: dict):
        """Initializes the processor and its configuration."""
        self.config = config
        self.vocabs, self.scalers = {}, {}
        self.categorical_cols = ["property_type", "room_type"]
        self.numerical_cols = [
            "accommodates", "review_scores_rating", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication",
            "review_scores_location", "review_scores_value",
            "bedrooms", "beds", "bathrooms"
        ]
        self.log_transform_cols = ["total_reviews"]

    def fit(self, df: pd.DataFrame):
        """Fits scalers and vocabularies based on the training data."""
        for col in self.categorical_cols:
            self.vocabs[col] = {val: i for i, val in enumerate(["<UNK>"] + sorted(df[col].unique()))}
        for col in self.numerical_cols + self.log_transform_cols:
            vals = df[col].astype(float)
            vals = np.log1p(vals) if col in self.log_transform_cols else vals
            self.scalers[col] = {'mean': vals.mean(), 'std': vals.std()}

    def transform(self, df: pd.DataFrame, neighborhood_log_means: dict) -> dict:
        """Transforms a DataFrame into a dictionary of feature tensors."""
        df = df.copy()
        df['neighborhood_log_mean'] = df['neighbourhood_cleansed'].map(neighborhood_log_means)
        global_mean = sum(neighborhood_log_means.values()) / len(neighborhood_log_means)
        df['neighborhood_log_mean'] = df['neighborhood_log_mean'].fillna(global_mean)

        target_log_deviation = (np.log1p(df["price"]) - df['neighborhood_log_mean']).to_numpy(dtype=np.float32)

        def positional_encoding(arr, max_val, d):
            pos = (arr / max_val) * 10000.0
            idx = np.arange(0, d, 2, dtype=np.float32)
            div = np.exp(-(np.log(10000.0) / d) * idx)
            s, c = np.sin(pos[:, None] * div[None, :]), np.cos(pos[:, None] * div[None, :])
            out = np.empty((arr.shape[0], d), dtype=np.float32)
            out[:, 0::2], out[:, 1::2] = s, c
            return out

        half_dim = self.config['GEO_EMBEDDING_DIM'] // 2
        lat, lon = df["latitude"].to_numpy(np.float32), df["longitude"].to_numpy(np.float32)
        geo_position = np.hstack([positional_encoding(lat, 90.0, half_dim), positional_encoding(lon, 180.0, half_dim)])

        size_features, quality_features = {}, {}
        size_num_cols = ["accommodates", "bedrooms", "beds", "bathrooms"]
        for col in self.categorical_cols:
            size_features[col] = df[col].map(self.vocabs[col]).fillna(0).astype(np.int64)
        for col in size_num_cols:
            size_features[col] = ((df[col] - self.scalers[col]["mean"]) / self.scalers[col]["std"]).astype(np.float32)
        
        quality_num_cols = set(self.numerical_cols) - set(size_num_cols)
        for col in quality_num_cols:
            quality_features[col] = ((df[col] - self.scalers[col]["mean"]) / self.scalers[col]["std"]).astype(np.float32)

        tr_log = np.log1p(df["total_reviews"].astype(float))
        quality_features["total_reviews"] = ((tr_log - self.scalers["total_reviews"]["mean"]) / self.scalers["total_reviews"]["std"]).astype(np.float32)
        quality_features["host_is_superhost"] = df["host_is_superhost"].astype(np.float32)
        
        month = df["month"].to_numpy(np.float32)
        season_cyc = np.stack([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)], axis=1)

        return {
            "location": {"geo_position": geo_position},
            "size_capacity": {k: v.to_numpy() for k, v in size_features.items()},
            "quality": {k: v.to_numpy() for k, v in quality_features.items()},
            "amenities_text": df["amenities"].tolist(),
            "description_text": df["description"].tolist(),
            "seasonality": {"cyclical": season_cyc},
            "target_price": df["price"].to_numpy(dtype=np.float32),
            "target_log_deviation": target_log_deviation,
            "neighborhood_log_mean": df['neighborhood_log_mean'].to_numpy(dtype=np.float32),
        }

def create_dataloaders(train_features: dict, val_features: dict, config: dict) -> Tuple[DataLoader, DataLoader]:
    """Initializes and returns the training and validation DataLoaders."""
    tokenizer = AutoTokenizer.from_pretrained(config['TEXT_MODEL_NAME'], use_fast=True)
    train_dataset = AirbnbPriceDataset(train_features, tokenizer)
    val_dataset = AirbnbPriceDataset(val_features, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['VALIDATION_BATCH_SIZE'], shuffle=False,
        num_workers=2, pin_memory=True
    )
    return train_loader, val_loader