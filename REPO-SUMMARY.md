# Repository Summary: Fall-2025-Team-Big-Data

## Project Overview

This repository contains a deep learning project for building an explainable Airbnb pricing tool. The project aims to create a transparent, evidence-based pricing recommendation system that decomposes price contributions into interpretable components and supports recommendations with comparable listings.

## Full Repository Tree

```text
Fall-2025-Team-Big-Data/
├── .DS_Store                                    # macOS system file (ignored)
├── .env                                         # Environment variables for secrets (ignored)
├── .gitattributes                               # Git file handling configuration
├── .gitignore                                   # Git ignore patterns for Python, Jupyter, AWS, data files
├── README.md                                    # Main project overview and team information
├── SETUP.md                                     # Comprehensive setup guide for Conda, Java/PySpark, Git workflow
├── environment.yml                              # Conda environment specification with Python 3.12, PySpark, ML libs
├── APP.md                                       # Technical specification for AWS deployment (SageMaker + Streamlit)
├── COMP-SET.md                                  # Methodology for generating market comparable listings via 2-stage retrieval
├── DATASET-SCHEMA.md                            # Schema definition for final modeling dataset (listing-month grain)
├── EMBEDDINGS.md                                # Feature representation strategy for 5-axis neural network
├── FEATURE-AXES.md                              # Additive price decomposition into 5 interpretable axes
├── MODELING.md                                  # Multi-phase model architecture from baseline to advanced
├── PRICING-TOOL-IDEA.md                         # Original project proposal for explainable pricing tool
├── SUGGESTED-REFINEMENTS.md                     # AI-suggested improvements for model and architecture
├── TARGET-PRICE.md                              # Target variable engineering using market-accepted prices
├── Naive_NN_for_AirBNB_NYC.ipynb               # Initial neural network experiment with fastai tabular models
├── data/                                        # Data storage directory
│   ├── data-description/                       # Data dictionary and schema files
│   │   ├── data-desc-calendar.csv             # Calendar dataset schema description
│   │   ├── data-desc-listings.csv             # Listings dataset schema description
│   │   ├── data-desc-reviews.csv              # Reviews dataset schema description
│   │   ├── inside-airbnb-data-dictionary.csv  # InsideAirbnb official data dictionary
│   │   └── airroi/                             # AirROI data source descriptions
│   │       ├── data-desc-listings.tsv         # AirROI listings schema (TSV format)
│   │       ├── data-desc-past-calender-rates.tsv # AirROI calendar rates schema
│   │       └── data-desc-reviews.tsv          # AirROI reviews schema
│   ├── nyc/                                    # New York City dataset files
│   │   ├── nyc-listings-detailed-insideairbnb.csv
│   │   ├── airroi/
│   │   │   ├── nyc-listings-airroi.csv
│   │   │   ├── nyc-past-calendar-rates-airroi.csv
│   │   │   └── nyc-reviews-airroi.csv
│   │   └── insideairbnb-samples/
│   │       ├── nyc-calendar-insideairbnb-sample.csv
│   │       ├── nyc-listings-detailed-insideairbnb-sample.csv
│   │       └── nyc-reviews-detailed-insideairbnb-sample.csv
│   └── toronto/                                # Toronto dataset files
│       ├── toronto-listings.csv
│       ├── toronto-past-calender-rates.csv
│       └── toronto-reviews.csv
├── notebooks/                                  # Jupyter notebooks directory
│   └── .gitkeep                               # Placeholder to maintain empty directory in Git
├── scripts/                                    # Python scripts directory
│   └── .gitkeep                               # Placeholder to maintain empty directory in Git
├── sql/                                        # SQL queries directory
│   └── .gitkeep                               # Placeholder to maintain empty directory in Git
├── src/                                        # Source code directory
│   └── .gitkeep                               # Placeholder to maintain empty directory in Git
└── tinkering/                                  # Experimental notebooks directory
    ├── calendar-exploration.ipynb             # Exploratory analysis of calendar/availability data
    └── exploration.ipynb                      # General data exploration notebook
```

## Key File Descriptions

### Documentation Files

- **README.md**: Project overview, team info, objectives for portfolio-grade ML pipeline on AWS
- **SETUP.md**: Complete development environment setup including Conda, Java/PySpark, Git workflow
- **APP.md**: AWS deployment architecture using SageMaker endpoints and Streamlit on App Runner
- **COMP-SET.md**: Two-stage retrieval system for finding comparable listings using model-learned representations
- **DATASET-SCHEMA.md**: Final modeling dataset schema with listing-month grain and 5-axis feature organization
- **EMBEDDINGS.md**: Feature embedding strategies tailored to data types (learned, one-hot, positional encoding)
- **FEATURE-AXES.md**: Additive price model decomposing price into 5 interpretable axes (location, size, amenities, quality, seasonality)
- **MODELING.md**: Phased model development from simple additive baseline to advanced interactive architecture
- **PRICING-TOOL-IDEA.md**: Original project proposal for explainable short-term rental pricing tool
- **TARGET-PRICE.md**: Target variable engineering using review activity as proxy for market acceptance
- **SUGGESTED-REFINEMENTS.md**: AI-generated improvements for comparable set generation, model architecture, and production deployment

### Code Files

- **Naive_NN_for_AirBNB_NYC.ipynb**: Initial tabular neural network experiments using fastai with Random Forest comparison
- **environment.yml**: Conda environment with Python 3.12, PySpark, PyTorch-ready ML stack, and AWS tools

### Data Structure

- **data/**: Contains NYC and Toronto Airbnb datasets in both CSV and Parquet formats, plus data dictionaries
- **data/data-description/**: Schema documentation for all datasets from multiple sources (InsideAirbnb, AirROI)
- **tinkering/**: Exploratory Jupyter notebooks for initial data analysis and calendar exploration

### Project Organization

- **notebooks/**, **scripts/**, **sql/**, **src/**: Organized directories for different code types (currently empty, ready for development)

## Architecture Summary

The project implements an explainable pricing system with:

1. **5-Axis Price Decomposition**: Location, Size/Capacity, Amenities, Quality/Reputation, Seasonality
2. **Two-Stage Comparable Retrieval**: Multi-index ANN search + dynamic weighted re-ranking
3. **Market-Based Target Engineering**: Review activity as occupancy proxy for revenue-maximizing prices
4. **AWS Cloud Deployment**: SageMaker for ML serving, Streamlit on App Runner for UI
5. **Transparent Explanations**: Price breakdowns and evidence-based comparable listings

This repository serves as a comprehensive foundation for building and deploying an interpretable machine learning system for short-term rental pricing optimization.
