# utils.py

"""
Utility functions for the Airbnb price prediction project.

This module contains helper functions for tasks such as setting random seeds for
reproducibility and various distance calculations required for the nearest
neighbor similarity search.
"""

import random
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

def set_seed(seed: int):
    """
    Sets random seeds for numpy, torch, and Python's random module to ensure
    reproducible results across runs.

    Args:
        seed (int): The integer value to use for all random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These settings are needed for full determinism with CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}.")


def haversine_distance(latlon1_rad: np.ndarray, latlon2_rad: np.ndarray) -> np.ndarray:
    """
    Calculates the Haversine distance in miles between a single point and an
    array of other points.

    Args:
        latlon1_rad (np.ndarray): A 1D array of shape (2,) containing the
            latitude and longitude of the query point in radians.
        latlon2_rad (np.ndarray): A 2D array of shape (N, 2) containing the
            latitudes and longitudes of N comparison points in radians.

    Returns:
        np.ndarray: A 1D array of shape (N,) containing the calculated
            distances in miles.
    """
    dlon = latlon2_rad[:, 1] - latlon1_rad[1]
    dlat = latlon2_rad[:, 0] - latlon1_rad[0]
    a = np.sin(dlat / 2.0)**2 + np.cos(latlon1_rad[0]) * np.cos(latlon2_rad[:, 0]) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3959
    return earth_radius_miles * c


def calculate_axis_importances(p_contributions_single_listing: dict, exclude_axes: list = None) -> dict:
    """
    Calculates the normalized importance weights for each model axis based on
    the absolute magnitude of its price contribution.

    Args:
        p_contributions_single_listing (dict): A dictionary where keys are axis
            names (e.g., 'size_capacity') and values are their float price
            contributions for a single listing.
        exclude_axes (list, optional): A list of axis names to exclude from
            the importance calculation. Defaults to None.

    Returns:
        dict: A dictionary with the same keys as the input (minus excluded
            axes) and values representing the normalized importance (summing to 1).
    """
    exclude_axes = exclude_axes or []
    filtered_contributions = {k: v for k, v in p_contributions_single_listing.items() if k not in exclude_axes}

    abs_contributions = {k: abs(v) for k, v in filtered_contributions.items()}
    total_abs_contribution = sum(abs_contributions.values())

    if total_abs_contribution == 0:
        num_axes = len(abs_contributions)
        return {k: 1.0 / num_axes for k in abs_contributions} if num_axes > 0 else {}

    return {k: v / total_abs_contribution for k, v in abs_contributions.items()}


def euclidean_distance(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance from a single vector to every row vector
    in a matrix.

    Args:
        vector (np.ndarray): A 1D array of shape (D,).
        matrix (np.ndarray): A 2D array of shape (N, D).

    Returns:
        np.ndarray: A 1D array of shape (N,) containing the distances.
    """
    return cdist(vector.reshape(1, -1), matrix, 'euclidean').flatten()


def cosine_distance(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the Cosine distance from a single vector to every row vector
    in a matrix.

    Args:
        vector (np.ndarray): A 1D array of shape (D,).
        matrix (np.ndarray): A 2D array of shape (N, D).

    Returns:
        np.ndarray: A 1D array of shape (N,) containing the distances.
    """
    return cdist(vector.reshape(1, -1), matrix, 'cosine').flatten()

def plot_target_distributions(train_df: pd.DataFrame, val_df: pd.DataFrame, neighborhood_log_means: Dict[str, float]):
    """
    Visualizes and compares the distributions of key target variables for the
    training and validation sets.

    This function generates a 2x2 grid of plots showing:
    1. Raw price distribution.
    2. Log-transformed price distribution.
    3. Log-price deviation from neighborhood mean distribution.
    4. Distribution of the neighborhood mean log-prices themselves.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        val_df (pd.DataFrame): The validation dataframe.
        neighborhood_log_means (dict): A dictionary mapping neighborhoods to
            their average log-price calculated from the training set.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution of Target and Derived Variables', fontsize=16)

    # --- Prepare dataframes with deviation column ---
    train_vis_df = train_df.copy()
    val_vis_df = val_df.copy()

    # Map neighborhood means and calculate deviation for plotting
    global_mean = np.mean(list(neighborhood_log_means.values()))
    train_vis_df['log_mean'] = train_vis_df['neighbourhood_cleansed'].map(neighborhood_log_means).fillna(global_mean)
    train_vis_df['log_deviation'] = np.log1p(train_vis_df['price']) - train_vis_df['log_mean']
    val_vis_df['log_mean'] = val_vis_df['neighbourhood_cleansed'].map(neighborhood_log_means).fillna(global_mean)
    val_vis_df['log_deviation'] = np.log1p(val_vis_df['price']) - val_vis_df['log_mean']

    # --- Plot 1: Raw Price ---
    sns.histplot(data=train_vis_df, x='price', ax=axes[0, 0], kde=True, label='Train', color='skyblue', bins=50)
    sns.histplot(data=val_vis_df, x='price', ax=axes[0, 0], kde=True, label='Val', color='salmon', bins=50)
    axes[0, 0].set_title('1. Raw Price Distribution')
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].legend()

    # --- Plot 2: Log Price ---
    train_vis_df['log_price'] = np.log1p(train_vis_df['price'])
    val_vis_df['log_price'] = np.log1p(val_vis_df['price'])
    sns.histplot(data=train_vis_df, x='log_price', ax=axes[0, 1], kde=True, label='Train', color='skyblue')
    sns.histplot(data=val_vis_df, x='log_price', ax=axes[0, 1], kde=True, label='Val', color='salmon')
    axes[0, 1].set_title('2. Log-Transformed Price Distribution')
    axes[0, 1].set_xlabel('Log(1 + Price)')
    axes[0, 1].legend()

    # --- Plot 3: Log Price Deviation ---
    sns.histplot(data=train_vis_df, x='log_deviation', ax=axes[1, 0], kde=True, label='Train', color='skyblue')
    sns.histplot(data=val_vis_df, x='log_deviation', ax=axes[1, 0], kde=True, label='Val', color='salmon')
    axes[1, 0].set_title('3. Distribution of Target Variable (Log Deviation)')
    axes[1, 0].set_xlabel('Log(Price) - Neighborhood_Log_Mean')
    axes[1, 0].legend()

    # --- Plot 4: Neighborhood Log Means ---
    sns.histplot(x=list(neighborhood_log_means.values()), ax=axes[1, 1], kde=True, color='purple')
    axes[1, 1].set_title('4. Distribution of Neighborhood Mean Log-Prices')
    axes[1, 1].set_xlabel('Mean Log-Price per Neighborhood')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_training_history(history_df: pd.DataFrame):
    """
    Visualizes the training and validation loss curves alongside the validation MAPE.

    This function creates a dual-axis plot:
    - The left y-axis shows the Root Mean Squared Error (RMSE) for both the
      training and validation sets.
    - The right y-axis shows the Mean Absolute Percentage Error (MAPE) for the
      validation set.

    Args:
        history_df (pd.DataFrame): A DataFrame containing the training history,
            with columns 'epoch', 'train_rmse', 'val_rmse', and 'val_mape'.
    """
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot RMSE on the primary y-axis
    ax1.plot(history_df['epoch'], history_df['train_rmse'], 'b-', label='Train RMSE')
    ax1.plot(history_df['epoch'], history_df['val_rmse'], 'c-', label='Validation RMSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE (Log Deviation)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')

    # Create a secondary y-axis for MAPE
    ax2 = ax1.twinx()
    ax2.plot(history_df['epoch'], history_df['val_mape'] * 100, 'r--', label='Validation MAPE')
    ax2.set_ylabel('Validation MAPE (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Training History: Loss and MAPE')
    plt.show()


def plot_predictions_vs_actual(train_results_df: pd.DataFrame, val_results_df: pd.DataFrame, max_sample_size: int = 2000):
    """
    Generates scatter plots of true vs. predicted prices for training and validation sets.

    To prevent overplotting with large datasets, this function will plot a random
    sample of the data if the number of records exceeds max_sample_size. A diagonal
    line (y=x) is included to represent a perfect prediction.

    Args:
        train_results_df (pd.DataFrame): A DataFrame with at least 'price' and
            'predicted_price' columns for the training set.
        val_results_df (pd.DataFrame): A DataFrame with at least 'price' and
            'predicted_price' columns for the validation set.
        max_sample_size (int): The maximum number of points to plot to avoid
            overcrowding. Defaults to 2000.
    """
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Training Set Plot ---
    if len(train_results_df) > max_sample_size:
        train_sample = train_results_df.sample(n=max_sample_size, random_state=42)
    else:
        train_sample = train_results_df

    sns.scatterplot(x='price', y='predicted_price', data=train_sample, ax=ax1, alpha=0.5, s=20)
    ax1.plot([0, train_sample['price'].max()], [0, train_sample['price'].max()], 'r--', lw=2)
    ax1.set_title(f'Train Set: True vs. Predicted Prices (Sampled {len(train_sample)} points)')
    ax1.set_xlabel('True Price ($)')
    ax1.set_ylabel('Predicted Price ($)')
    ax1.grid(True)

    # --- Validation Set Plot ---
    if len(val_results_df) > max_sample_size:
        val_sample = val_results_df.sample(n=max_sample_size, random_state=42)
    else:
        val_sample = val_results_df

    sns.scatterplot(x='price', y='predicted_price', data=val_sample, ax=ax2, alpha=0.5, s=20, color='orange')
    ax2.plot([0, val_sample['price'].max()], [0, val_sample['price'].max()], 'r--', lw=2)
    ax2.set_title(f'Validation Set: True vs. Predicted Prices (Sampled {len(val_sample)} points)')
    ax2.set_xlabel('True Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()