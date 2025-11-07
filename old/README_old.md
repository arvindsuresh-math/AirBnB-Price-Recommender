# Airbnb Price Prediction with Deep Learning

A comprehensive deep learning project for predicting Airbnb listing prices using interpretable neural network models. The project includes data processing, model training, evaluation, and a Streamlit web application for interactive price exploration.

## 🏗️ Project Overview

This project develops and compares multiple machine learning approaches for Airbnb price prediction:

1. **Random Forest** - Traditional tree-based model for baseline performance
2. **Baseline Neural Network** - Fully-connected deep learning model
3. **Additive Neural Network** - Interpretable deep learning model with modular architecture

The **Additive Neural Network** is the flagship model, designed with explainability in mind. It decomposes price predictions into interpretable components (location, size/capacity, quality, amenities, description, and seasonality), making it ideal for a price recommendation tool.

## 📊 Data

The project uses Airbnb listing data for two cities:
- **New York City** (`nyc/`)
- **Toronto** (`toronto/`)

### Key Features
- **Location**: Latitude, longitude, neighborhood information
- **Size/Capacity**: Accommodates, bedrooms, beds, bathrooms, property/room type
- **Quality**: Review scores, superhost status, total reviews
- **Amenities**: Parsed text features of available amenities
- **Description**: Natural language processing of listing descriptions
- **Seasonality**: Monthly price variations

## 🧠 Model Architecture

### Additive Neural Network
The interpretable model consists of specialized sub-networks:

```
Location Network → Size/Capacity Network → Quality Network → Amenities Network → Description Network → Seasonality Network
                                                                 ↓
                                                          Final Prediction
```

Each sub-network processes its domain-specific features and contributes an additive component to the final log-price prediction. This modular design enables:
- **Explainability**: See exactly how each factor affects the price
- **Feature Importance**: Understand which aspects drive pricing decisions
- **Debugging**: Isolate and improve individual components

## 📈 Results and Analysis

The project includes comprehensive evaluation notebooks:

- `notebooks/nn_models_nyc.ipynb` - Baseline neural network training
- `notebooks/nn_models_toronto.ipynb` - Toronto-specific model training
- `notebooks/results_and_analysis.ipynb` - Model comparison and interpretability analysis

### Performance Metrics
- **Mean Absolute Percentage Error (MAPE)**
- **Root Mean Square Error (RMSE)**
- **R² Score**

### Key Findings
- The Additive Neural Network achieves competitive performance while providing full interpretability
- Location and size/capacity are the strongest predictors
- Seasonal effects show clear patterns (summer peaks, winter lows)
- Text features (amenities, descriptions) add significant predictive power

## 🏛️ Project Structure

```
├── app/                    # Streamlit web application
│   ├── src/
│   │   ├── base_map_app.py         # Main app with interactive map
│   │   ├── streamlit_app.py        # Alternative app entry point
│   │   └── page_modules/           # Additional app modules and data
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── compose.yaml
│   ├── README.md
│   └── README.Docker.md
├── data/                   # Raw and processed datasets
│   ├── data-description/   # Data dictionaries and schemas
│   │   ├── data-desc-listings.csv
│   │   ├── data-desc-reviews.csv
│   │   └── inside-airbnb-data-dictionary.csv
│   ├── nyc/               # New York City data
│   │   ├── nyc_sample_listings_oct_20.csv
│   │   ├── nyc_dataset_oct_20.parquet
│   │   └── nyc-neighbourhoods.geojson
│   └── toronto/           # Toronto data
│       ├── toronto_sample_listings_oct_20.csv
│       ├── toronto_dataset_oct_20.parquet
│       └── toronto-neighbourhoods.geojson
├── notebooks/             # Jupyter notebooks for analysis
│   ├── data_cleaning.ipynb
│   ├── data_cleaning_files/    # Supporting files for data cleaning
│   ├── nn_models_nyc.ipynb
│   ├── nn_models_toronto.ipynb
│   ├── random_forest_nyc.ipynb
│   ├── random_forest_toronto.ipynb
│   └── results_and_analysis.ipynb
├── src/                   # Core Python modules
│   ├── build_app_dataset.py   # Data preparation for app
│   ├── config.py              # Model hyperparameters
│   ├── data_dictionaries.md   # Data documentation
│   ├── data_processing.py     # Data preprocessing utilities
│   ├── inference.py           # Model inference utilities
│   ├── model.py               # PyTorch model architectures
│   ├── plotting.py            # Visualization utilities
│   ├── similarity.py          # Similarity search for recommendations
│   ├── train.py               # Training and evaluation functions
│   └── requirements.txt
└── old/                   # Legacy code and experiments
```

## 🔧 Configuration

Model hyperparameters and settings are centralized in `src/config.py`:

- **Architecture**: Hidden layer dimensions, dropout rates
- **Training**: Learning rates, batch sizes, early stopping
- **Data**: City selection, validation split, random seeds

## 📚 Key Dependencies

- **PyTorch** - Deep learning framework
- **Transformers** - Pre-trained language models for text processing
- **Sentence Transformers** - Text embeddings
- **Streamlit** - Web application framework
- **Folium** - Interactive maps
- **Scikit-learn** - Traditional ML models
- **Pandas/PyArrow** - Data processing

## 🙏 Acknowledgments

- Airbnb data provided through Inside Airbnb project
- Built with PyTorch and the Hugging Face transformers library
- Inspired by interpretable machine learning research