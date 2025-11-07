# Airbnb Price Prediction with an Interpretable Deep Learning Model

This project delivers a comprehensive deep learning solution for predicting Airbnb listing prices, featuring an interpretable neural network that decomposes price drivers into understandable components. The end-to-end pipeline covers data processing, model training, performance evaluation, and a Streamlit web application for interactive price exploration and competitor analysis.

## 🏗️ Project Overview

The primary goal is to predict Airbnb prices accurately while providing clear, explainable insights into *why* a listing has a certain price. To achieve this, the project develops and compares three distinct modeling approaches for datasets from New York City and Toronto:

1. **Random Forest Baseline**: A robust, traditional machine learning model that serves as a strong performance benchmark.
2. **Deep Learning Baseline**: A fully-connected neural network that processes all features together to establish a baseline for deep learning performance.
3. **Interpretable Additive Model**: The flagship model of this project. It features a modular architecture where specialized sub-networks learn the influence of distinct feature groups (e.g., location, quality, amenities). This design allows the final price prediction to be broken down into its constituent parts, offering unparalleled explainability.

The final additive model powers an interactive Streamlit application that not only predicts prices but also allows users to understand the impact of each feature axis and find comparable listings based on multi-faceted similarity.

## 📊 Data Pipeline

The project utilizes detailed monthly snapshots of Airbnb listings and reviews from the Inside Airbnb project. A rigorous ETL pipeline, detailed in `notebooks/data_cleaning.ipynb`, transforms this raw data into a model-ready panel dataset.

Key steps include:

- **Loading & Concatenation**: Combining 12 monthly snapshots to capture seasonal trends.
- **Feature Engineering**: Creating new features like `price_per_person` and performing log transformations.
- **Cleaning & Sanitization**: Parsing complex text fields like `amenities`, cleaning HTML tags, and handling missing values.
- **Outlier Removal**: Filtering the top and bottom 5% of listings based on price-per-person to stabilize the dataset.
- **Consistent Data Splitting**: A custom **stratified group split** is used to ensure that all records for a single listing belong to either the training or validation set. This prevents data leakage and provides a reliable evaluation of model performance.

## 🧠 Model Architecture

### Interpretable Additive Model

The core innovation of this project is the Additive Model, which is designed for explainability without sacrificing performance. Instead of predicting the price directly, the model predicts the **logarithmic deviation from the average price of a listing's neighborhood**. This approach has a powerful consequence: it transforms the model into a system of multiplicative factors.

The architecture consists of six independent sub-networks, each focused on a specific aspect of an Airbnb listing:

```
┌───────────────────┐  ┌─────────────────────┐  ┌─────────────────┐
│ Location Network  │  │ Size/Capacity Network├─►│ Quality Network │
└───────────────────┘  └─────────────────────┘  └─────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
        p_loc                p_size_capacity          p_quality
          │                      │                      │
          └───────────► p_base + Σ(p_i) ◄───────────┘
          ┌───────────────────────▲───────────────────────┐
          │                       │                       │
      p_amenities            p_description         p_seasonality
          ▲                       ▲                       ▲
          │                      │                      │
┌───────────────────┐  ┌─────────────────────┐  ┌───────────────────┐
│ Amenities Network │◄─┤ Description Network │  │ Seasonality Network│
└───────────────────┘  └─────────────────────┘  └───────────────────┘
```

#### From Additive Logs to Multiplicative Prices

Each sub-network outputs an additive adjustment (`p_location`, `p_size_capacity`, etc.) in log-price space. The model's total predicted log-price deviation is the sum of these individual components.

1. **Additive in Log-Space**: The complete prediction in log-space is calculated as:

    ```
    Predicted_Log_Price = p_base_log + p_location + p_size_capacity + p_quality + p_amenities + p_description + p_seasonality
    ```

    where `p_base_log` is the mean log-price of the listing's neighborhood.

2. **Multiplicative in Price-Space**: To convert this back to a dollar amount, we take the exponential (`exp()`). Due to the properties of exponents, this additive relationship becomes multiplicative:

    ```
    Predicted_Price = exp(Predicted_Log_Price)
                    = exp(p_base_log) * exp(p_location) * exp(p_size_capacity) * ...
    ```

    This results in a final, intuitive formula:

    ```
    Predicted_Price = (Neighborhood_Base_Price) * Factor_location * Factor_size * ...
    ```

Each "Factor" (`Factor_x = exp(p_x)`) represents the percentage premium or discount contributed by that feature axis. For example, a `Factor_location` of **1.15** means the listing's specific location commands a **15% premium** over the neighborhood average, while a factor of **0.90** indicates a **10% discount**. This design makes the model's reasoning transparent and easy to communicate.

## 🚀 Interactive Streamlit Application

The project culminates in a powerful Streamlit application that brings the interpretable model to life. It serves as both a market explorer and a price recommendation tool for hosts. The application has two main pages:

### 1. Main Map View

This page provides a geographic interface to explore existing Airbnb listings in the selected city.

- **Interactive Map**: Displays all listings, with neighborhood boundaries overlaid. Users can pan, zoom, and click on listings to see a quick summary.
- **Advanced Filtering**: A sidebar allows users to filter the displayed listings by city, room type, neighborhood, price range, and capacity.
- **Detailed Analysis**: When a user selects a listing from the map or a table, a detailed analysis panel appears, showing:
  - **Price Contribution Breakdown**: A table itemizing the multiplicative factor for each of the six feature axes (Location, Quality, etc.), showing exactly what drives the listing's price.
  - **Price Trends Over Time**: A line chart displaying the model's predicted price for the selected listing for all 12 months of the year, visualizing its seasonality.
  - **Similar Properties Nearby**: A table of the top 5 most similar listings within a 2-mile radius, allowing for direct competitor analysis.
  - **Summary Statistics**: Key metrics including the average, lowest, and highest predicted price for the listing throughout the year.

### 2. Price Recommender

This page empowers hosts to get a price estimate for a new or hypothetical listing.

- **Intuitive Input Form**: A comprehensive form allows users to input all relevant details of a property, including its address, size, amenities, and perceived quality (via review score sliders).
- **Instant Price Recommendation**: Upon submission, the model provides:
  - **Recommended Price**: A clear price estimate for the specified month.
  - **Neighborhood Context**: The recommended price is compared directly against the average price for that neighborhood.
  - **Price Adjustment Waterfall**: A highly intuitive waterfall chart that visually breaks down the price. It starts with the neighborhood base price and shows how each feature axis (Location, Size, Quality, etc.) adds to or subtracts from the final price in dollar amounts.
  - **Find Similar Listings**: The app also runs the similarity search to find the 5 most comparable existing properties, giving hosts a real-world sanity check for their pricing strategy.

### Finding Similar Listings: A Weighted Approach

The "Similar Properties Nearby" feature is powered by a sophisticated algorithm that goes beyond simple distance metrics. It finds listings that are not just geographically close but are *similar in the ways that matter most* for determining price.

1. **Geospatial Filtering**: First, the search is narrowed to candidate listings within a **2-mile radius** of the query listing using the Haversine distance formula. This ensures local relevance and computational efficiency.
2. **Axis-Importance Weighting**: The algorithm then calculates importance weights for each of the model's feature axes based on the *query listing's own price breakdown*. For example, if a luxury penthouse's price is 70% determined by its `Quality` and `Description`, the similarity search will be weighted to prioritize finding other listings with similar quality scores and descriptions. This is achieved by normalizing the absolute log-price contributions (`|p_quality|`, `|p_description|`, etc.) from the model.
3. **Calculating Weighted Distances**: For each candidate listing, the algorithm calculates the distance between its internal vector representation (the hidden state, e.g., `h_quality`) and the query's vector.
    - **Cosine Distance** is used for semantic text features (`Amenities`, `Description`).
    - **Euclidean Distance** is used for all other numerical axes (`Size`, `Quality`, etc.).
    The final similarity score is a **weighted sum** of these normalized distances, using the axis-importance weights from the previous step.
4. **Ranking and Display**: The candidate listings are ranked by this final composite similarity score, and the top 5 are presented to the user. This dynamic, context-aware approach provides highly relevant and actionable competitor insights.

## 📈 Results and Analysis

All models were trained and evaluated on a consistent validation set, comprising 5% of the total listings. The neural network models were trained with a custom early stopping rule to prevent overfitting, which halts training if the gap between validation and training MAPE exceeds 4% for 3 consecutive epochs.

The tables below summarize the best validation performance achieved by each model for both cities.

### New York City (NYC)

| Model | Validation MAPE (%) | Validation RMSE |
| :--- | :---: | :---: |
| Random Forest Baseline | 29.40% | N/A |
| Deep Learning Baseline | 27.89% | 0.3305 |
| **Interpretable Additive Model** | **27.30%** | **0.3449** |

### Toronto

| Model | Validation MAPE (%) | Validation RMSE |
| :--- | :---: | :---: |
| Random Forest Baseline | 31.08% | N/A |
| **Deep Learning Baseline** | **26.71%** | **0.3319** |
| Interpretable Additive Model | 28.91% | 0.3440 |

#### Key Findings

- The deep learning models consistently outperform the Random Forest baseline on MAPE, demonstrating their ability to capture complex, non-linear patterns in the data.
- The Interpretable Additive Model achieves performance very close to the standard (black-box) deep learning baseline, confirming that explainability can be achieved with a minimal trade-off in accuracy.
- The multiplicative factors derived from the additive model provide clear, actionable insights into the key drivers of price for any given listing.

## 🏛️ Project Structure

```
├── app/                    # Streamlit web application source code
├── data/                   # Raw and processed datasets for NYC and Toronto
├── notebooks/              # Jupyter notebooks for data cleaning, training, and analysis
├── src/                    # Core Python modules for the modeling pipeline
│   ├── config.py           # Central configuration for hyperparameters
│   ├── data_processing.py  # Data loading and feature engineering
│   ├── model.py            # PyTorch model architectures
│   ├── train.py            # Model training and evaluation loops
│   ├── inference.py        # Prediction generation
│   ├── build_app_dataset.py# Script to generate the final app dataset
│   ├── similarity.py       # Functions for similarity search
│   └── ...
└── README.md               # Project summary
```

## 🔧 Configuration and Hyperparameters

All key settings and hyperparameters are centralized in `src/config.py`. This allows for easy modification and experimentation with:

- **Model Architecture**: Hidden layer dimensions, embedding sizes, dropout rates.
- **Training Parameters**: Learning rates (including a separate, lower rate for the transformer), batch sizes, and early stopping criteria.
- **Environment**: City selection (`nyc` or `toronto`), random seed, and file paths.

## 📚 Key Dependencies

- **PyTorch**: The primary deep learning framework.
- **Transformers / Sentence-Transformers**: For leveraging pre-trained language models.
- **Streamlit**: For building the interactive web application.
- **Scikit-learn**: For the Random Forest baseline and data processing utilities.
- **Pandas / PyArrow**: For efficient data manipulation.

## 🙏 Acknowledgments

- This project utilizes data from [Inside Airbnb](http://insideairbnb.com), which is made available under the Creative Commons CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.
- The neural network models were built using PyTorch and the Hugging Face ecosystem.
