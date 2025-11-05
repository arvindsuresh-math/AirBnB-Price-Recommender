# data_processing.py

"""
Handles all data loading, splitting, and feature engineering for the project.

This module contains:
- load_and_split_data: Loads the dataset and performs a group-based split,
  ensuring all records for a given listing_id are in the same set.
- FeatureProcessor: A class to transform raw data into numerical tensors.
- create_full_panel_dataset: Augments the data to create a complete
  listing-month panel for comprehensive inference.
- create_dataloaders: A function to prepare PyTorch DataLoaders.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from model import AirbnbPriceDataset
from typing import Tuple, Dict, List, Set


def load_and_split_data(config: dict, n_strata_bins: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Set[int], Set[int]]:
    """
    Loads the dataset and performs a stratified, group-based train/validation split.

    This function ensures that all records for a given listing_id are in the same
    set (train or val). The stratification is performed on the average log-price
    deviation for each listing, ensuring that the validation set has a
    representative distribution of listings that are cheaper or more expensive
    than their neighborhood average.

    Args:
        config (dict): The global configuration dictionary.
        n_strata_bins (int): The number of bins to create for stratification.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The training data.
            - pd.DataFrame: The validation data.
            - dict: A mapping from neighborhood to its mean log-price,
                    calculated ONLY from the final training set.
            - set: A set of unique listing IDs in the training set.
            - set: A set of unique listing IDs in the validation set.
    """
    dataset_filename = f"{config['CITY']}_dataset_oct_20.parquet"
    dataset_path = f"./{dataset_filename}"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"'{dataset_filename}' not found. Please upload the file.")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df = df[df["price"] > 0].copy()

    # --- Stratification Logic ---
    # 1. Temporarily calculate neighborhood means on ALL data to find deviation for each listing.
    #    This is *only* for creating the stratification key and will be re-calculated properly later.
    temp_neighborhood_means = np.log1p(df.groupby('neighbourhood_cleansed')['price'].mean()).to_dict()
    df['temp_log_mean'] = df['neighbourhood_cleansed'].map(temp_neighborhood_means)
    df['log_price_deviation'] = np.log1p(df['price']) - df['temp_log_mean']

    # 2. Calculate the mean deviation for each unique listing ID.
    listing_avg_deviation = df.groupby('id')['log_price_deviation'].mean().to_frame()

    # 3. Create stratification bins from these average deviations.
    listing_avg_deviation['strata_bin'] = pd.qcut(
        listing_avg_deviation['log_price_deviation'],
        q=n_strata_bins,
        labels=False,
        duplicates='drop'
    )

    # 4. Perform the stratified split on the listing IDs.
    listing_ids = listing_avg_deviation.index
    strata = listing_avg_deviation['strata_bin']
    train_ids, val_ids = train_test_split(
        listing_ids,
        test_size=config['VAL_SIZE'],
        random_state=config['SEED'],
        stratify=strata
    )
    train_ids_set, val_ids_set = set(train_ids), set(val_ids)

    # 5. Create final dataframes and drop temporary columns.
    train_df = df[df['id'].isin(train_ids_set)].drop(columns=['temp_log_mean', 'log_price_deviation']).reset_index(drop=True)
    val_df = df[df['id'].isin(val_ids_set)].drop(columns=['temp_log_mean', 'log_price_deviation']).reset_index(drop=True)

    # 6. CRUCIAL: Re-calculate neighborhood means using ONLY the training data to prevent leakage.
    final_neighborhood_log_means = np.log1p(train_df.groupby('neighbourhood_cleansed')['price'].mean()).to_dict()

    print(f"Stratified split complete. Listings in Train: {len(train_ids_set):,}, Val: {len(val_ids_set):,}")
    print(f"Total records in Train: {len(train_df):,}, Val: {len(val_df):,}")

    return train_df, val_df, final_neighborhood_log_means, train_ids_set, val_ids_set


class FeatureProcessor:
    """
    Prepares raw DataFrame columns into numerical features for the model.
    The fit/transform pattern prevents data leakage from the validation set.
    """
    def __init__(self, config: dict):
        """Initializes the processor and its configuration."""
        self.vocabs, self.scalers = {}, {}
        self.embedding_dim_geo = config['GEO_EMBEDDING_DIM']
        self.categorical_cols = ["property_type", "room_type"]
        self.numerical_cols = [
            "accommodates", "review_scores_rating", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication",
            "review_scores_location", "review_scores_value",
            "bedrooms", "beds", "bathrooms"
        ]
        self.log_transform_cols = ["total_reviews"]

    def fit(self, df: pd.DataFrame):
        """
        Fits scalers and vocabularies based on the training data.

        Args:
            df (pd.DataFrame): The training dataframe.
        """
        print("Fitting FeatureProcessor...")
        for col in self.categorical_cols:
            self.vocabs[col] = {val: i for i, val in enumerate(["<UNK>"] + sorted(df[col].unique()))}

        for col in self.numerical_cols + self.log_transform_cols:
            vals = df[col].astype(float)
            vals = np.log1p(vals) if col in self.log_transform_cols else vals
            self.scalers[col] = {'mean': vals.mean(), 'std': vals.std()}
        print("Fit complete.")

    def transform(self, df: pd.DataFrame, neighborhood_log_means: dict) -> dict:
        """
        Transforms a DataFrame into a dictionary of feature tensors.

        Args:
            df (pd.DataFrame): The dataframe to transform.
            neighborhood_log_means (dict): A pre-computed dictionary mapping
                neighborhoods to their average log-price.

        Returns:
            dict: A dictionary of features ready for the PyTorch Dataset.
        """
        df = df.copy()
        df['neighborhood_log_mean'] = df['neighbourhood_cleansed'].map(neighborhood_log_means)
        global_mean = sum(neighborhood_log_means.values()) / len(neighborhood_log_means)
        df['neighborhood_log_mean'].fillna(global_mean, inplace=True)

        target_log_deviation = (np.log1p(df["price"]) - df['neighborhood_log_mean']).to_numpy(dtype=np.float32)

        def positional_encoding(arr, max_val, d):
            pos = (arr / max_val) * 10000.0
            idx = np.arange(0, d, 2, dtype=np.float32)
            div = np.exp(-(np.log(10000.0) / d) * idx)
            s, c = np.sin(pos[:, None] * div[None, :]), np.cos(pos[:, None] * div[None, :])
            out = np.empty((arr.shape[0], d), dtype=np.float32)
            out[:, 0::2], out[:, 1::2] = s, c
            return out

        half_dim = self.embedding_dim_geo // 2
        lat = df["latitude"].to_numpy(dtype=np.float32)
        lon = df["longitude"].to_numpy(dtype=np.float32)
        geo_position = np.hstack([positional_encoding(lat, 90.0, half_dim), positional_encoding(lon, 180.0, half_dim)])

        size_features, quality_features = {}, {}
        size_num_cols = ["accommodates", "bedrooms", "beds", "bathrooms"]
        for col in self.categorical_cols:
            size_features[col] = df[col].map(self.vocabs[col]).fillna(0).astype(np.int64)
        for col in size_num_cols:
            x = df[col].astype(float)
            size_features[col] = ((x - self.scalers[col]["mean"]) / self.scalers[col]["std"]).astype(np.float32)

        quality_num_cols = set(self.numerical_cols) - set(size_num_cols) - set(self.categorical_cols)
        for col in quality_num_cols:
            x = df[col].astype(float)
            quality_features[col] = ((x - self.scalers[col]["mean"]) / self.scalers[col]["std"]).astype(np.float32)

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

def create_full_panel_dataset(df: pd.DataFrame, train_ids: Set[int], val_ids: Set[int]) -> pd.DataFrame:
    """
    Augments the dataset to create a full panel for every listing and every month.

    This is used to generate predictions for months where no real data exists,
    enabling seasonal analysis in the final application.

    Args:
        df (pd.DataFrame): The original, complete listings dataframe.
        train_ids (Set[int]): The set of listing IDs used for training.
        val_ids (Set[int]): The set of listing IDs used for validation.

    Returns:
        pd.DataFrame: An augmented dataframe with one row per listing per month.
    """
    print("Creating augmented panel dataset for all months...")
    static_cols = [
        'id', 'host_id', 'name', 'description', 'host_is_superhost',
        'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type',
        'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'amenities', 'total_reviews'
    ]
    # Keep only the first occurrence of each listing for its static features
    unique_listings = df.drop_duplicates(subset=['id'], keep='first')[static_cols]
    
    # Create a Cartesian product of unique listings and all 12 months
    months_df = pd.DataFrame({'month': range(1, 13)})
    panel_df = unique_listings.merge(months_df, how='cross')

    # Merge the original price data back onto the full panel
    panel_df = panel_df.merge(df[['id', 'month', 'price']], on=['id', 'month'], how='left')

    # Add the 'split' column for traceability
    panel_df['split'] = panel_df['id'].apply(lambda x: 'train' if x in train_ids else 'val')
    
    # Fill review scores with the mean from the original data as a reasonable default
    review_cols = [col for col in df.columns if 'review_scores' in col]
    for col in review_cols:
        mean_score = df[col].mean()
        panel_df[col] = panel_df['id'].map(df.groupby('id')[col].first()).fillna(mean_score)

    print(f"Augmented panel created with {len(panel_df):,} rows.")
    return panel_df

def create_dataloaders(train_features: dict, val_features: dict, config: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Initializes and returns the training and validation DataLoaders.

    Args:
        train_features (dict): The dictionary of processed training features.
        val_features (dict): The dictionary of processed validation features.
        config (dict): The global configuration dictionary.

    Returns:
        tuple: A tuple containing the training DataLoader and validation DataLoader.
    """
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

    print("DataLoaders created.")
    return train_loader, val_loader