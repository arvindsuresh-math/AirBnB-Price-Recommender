"""
Contains all visualization functions for the results and analysis notebook.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predictions_vs_actual(results_df: pd.DataFrame, model_name: str, ax):
    """Generates a scatter plot of true vs. predicted prices on a given axis."""
    sample_df = results_df.sample(n=min(2000, len(results_df)), random_state=42)
    sns.scatterplot(x='price', y='predicted_price', data=sample_df, ax=ax, alpha=0.6, s=20)
    max_val = max(sample_df['price'].max(), sample_df['predicted_price'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2)
    ax.set_title(f'{model_name}: True vs. Predicted Prices')
    ax.set_xlabel('True Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.grid(True)

def plot_mape_distribution(results_df: pd.DataFrame, model_name: str, ax):
    """Generates a histogram of the Mean Absolute Percentage Error."""
    results_df['mape'] = np.abs(results_df['price'] - results_df['predicted_price']) / results_df['price']
    sns.histplot(results_df['mape'], ax=ax, kde=True, bins=50, stat="percent")
    ax.set_title(f'{model_name}: MAPE Distribution')
    ax.set_xlabel('Mean Absolute Percentage Error')
    ax.set_xlim(0, 1.0) # Cap x-axis at 100% error for readability

def plot_ablation_results(ablation_df: pd.DataFrame):
    """Creates a bar chart visualizing the drop in performance from ablation."""
    baseline_mape = ablation_df[ablation_df['excluded_axes'].str.contains("None")]['val_mape'].iloc[0]
    plot_df = ablation_df[~ablation_df['excluded_axes'].str.contains("None")].copy()
    plot_df['mape_increase'] = (plot_df['val_mape'] - baseline_mape) * 100
    plot_df['axis'] = plot_df['excluded_axes'].str.extract(r"\['(.*?)'\]")

    plt.figure(figsize=(12, 7))
    barplot = sns.barplot(x='mape_increase', y='axis', data=plot_df.sort_values('mape_increase', ascending=False), palette='viridis')
    plt.title('Impact of Removing Each Axis on Validation MAPE', fontsize=16)
    plt.xlabel('Increase in Validation MAPE (Percentage Points)')
    plt.ylabel('Axis Removed')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width + 0.01, p.get_y() + p.get_height()/2, f'{width:.3f} pp', va='center')
    plt.show()

def plot_additive_contributions(listing_data: pd.Series):
    """Creates a waterfall chart showing the price breakdown for a single listing."""
    contribs = {
        'Location': listing_data['p_location'],
        'Size/Capacity': listing_data['p_size_capacity'],
        'Quality': listing_data['p_quality'],
        'Amenities': listing_data['p_amenities'],
        'Description': listing_data['p_description'],
        'Seasonality': listing_data['p_seasonality']
    }
    
    base_price = np.expm1(listing_data['p_base_log'])
    predicted_price = listing_data['predicted_price']
    
    data = {'Category': [], 'Contribution ($)': []}
    current_price = base_price
    
    for name, p_log in sorted(contribs.items(), key=lambda item: item[1]):
        price_with_contrib = np.expm1(listing_data['p_base_log'] + p_log)
        price_delta = np.expm1(listing_data['p_base_log'] + p_log) - base_price
        # This is a simplified approximation; a true waterfall is more complex
        data['Category'].append(name)
        data['Contribution ($)'].append(price_delta)

    plot_df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Contribution ($)', y='Category', data=plot_df.sort_values('Contribution ($)'), palette='coolwarm')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title(f'Price Breakdown for Listing (Base Price: ${base_price:.2f})', fontsize=16)
    plt.xlabel('Price Contribution (Positive or Negative) vs. Neighborhood Average')
    plt.ylabel('Feature Axis')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()