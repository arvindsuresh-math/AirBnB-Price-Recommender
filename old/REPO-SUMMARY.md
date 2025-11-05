# Repository Summary: Fall-2025-Team-Big-Data

## Project Overview

This repository contains a deep learning project for building an explainable Airbnb pricing tool. The project aims to create a transparent, evidence-based pricing recommendation system that decomposes price contributions into interpretable components and supports recommendations with comparable listings.

## Full Repository Tree

```text
Fall-2025-Team-Big-Data/
├── .gitignore
├── README.md
├── APP.md
├── COMP-SET.md
├── DATASET-SCHEMA.md
├── EMBEDDINGS.md
├── FEATURE-AXES.md
├── MODELING.md
├── PRICING-TOOL-IDEA.md
├── TARGET-PRICE.md
├── environment.yml
├── instructions/
│   ├── 01_ETL_INSTRUCTIONS.md
│   ├── 02_PYTORCH_DATASET_INSTRUCTIONS.md
│   ├── 03_PYTORCH_MODEL_INSTRUCTIONS.md
│   ├── 04_MODEL_TRAINING_INSTRUCTIONS.md
│   ├── 05_SAGEMAKER_DEPLOYMENT_INSTRUCTIONS.md
│   └── 06_STREAMLIT_APP_INSTRUCTIONS.md
├── Naive_NN_for_AirBNB_NYC.ipynb                # Initial neural network experiment with fastai tabular models
├── data/                                        # Data storage directory
│   ├── data-description/                        # Data dictionary and schema files
│   │   ├── data-desc-calendar.csv               # Calendar dataset schema description
│   │   ├── data-desc-listings.csv               # Listings dataset schema description
│   │   ├── data-desc-reviews.csv                # Reviews dataset schema description
│   │   ├── inside-airbnb-data-dictionary.csv    # InsideAirbnb official data dictionary
│   │   └── airroi/                              # AirROI data source descriptions
│   │       ├── data-desc-listings.tsv           # AirROI listings schema (TSV format)
│   │       ├── data-desc-past-calender-rates.tsv # AirROI calendar rates schema
│   │       └── data-desc-reviews.tsv            # AirROI reviews schema
│   ├── nyc/                                     # New York City dataset files
│   │   ├── nyc-listings-detailed-insideairbnb.csv
│   │   ├── airroi/
│   │   │   ├── nyc-listings-airroi.csv
│   │   │   ├── nyc-past-calendar-rates-airroi.csv
│   │   │   └── nyc-reviews-airroi.csv
│   │   └── insideairbnb-samples/
│   │       ├── nyc-calendar-insideairbnb-sample.csv
│   │       ├── nyc-listings-detailed-insideairbnb-sample.csv
│   │       └── nyc-reviews-detailed-insideairbnb-sample.csv
│   └── toronto/                                 # Toronto dataset files
│       ├── toronto-listings.csv
│       ├── toronto-past-calender-rates.csv
│       └── toronto-reviews.csv
├── notebooks/                                   # Jupyter notebooks directory
│   └── .gitkeep                                 # Placeholder to maintain empty directory in Git
├── scripts/                                     # Python scripts directory
│   └── .gitkeep                                 # Placeholder to maintain empty directory in Git
├── sql/                                         # SQL queries directory
│   └── .gitkeep                                 # Placeholder to maintain empty directory in Git
├── src/                                         # Source code directory
│   └── .gitkeep                                 # Placeholder to maintain empty directory in Git
└── tinkering/                                   # Experimental notebooks directory
    ├── calendar-exploration.ipynb               # Exploratory analysis of calendar/availability data
    └── exploration.ipynb                        # General data exploration notebook
```

## Key File Descriptions

### Design Documentation Files

- **APP.md**: Technical specification for the AWS deployment architecture (SageMaker + Streamlit).
- **COMP-SET.md**: Methodology for the two-stage retrieval system for finding comparable listings.
- **DATASET-SCHEMA.md**: Final schema for the modeling dataset at the `listing-month` grain.
- **EMBEDDINGS.md**: Feature representation and embedding strategies for the 5-axis neural network.
- **FEATURE-AXES.md**: Defines the additive price decomposition into 5 interpretable axes.
- **MODELING.md**: Phased model development from a simple additive baseline to more advanced architectures.
- **PRICING-TOOL-IDEA.md**: **(Updated)** High-level project proposal reflecting the final design decisions.
- **TARGET-PRICE.md**: The definitive guide to engineering the target variable using review velocity as a proxy for market acceptance and as a sample weight.

### LLM Agent Instruction Files (`instructions/`)

These files are information-dense blueprints designed to be fed to an LLM agent to generate the project's codebase. Each includes specifications for implementation, dependencies, and unit tests.

- **01_ETL_INSTRUCTIONS.md**: Generates the PySpark script to build the final modeling dataset from raw data.
- **02_PYTORCH_DATASET_INSTRUCTIONS.md**: Generates the PyTorch `Dataset` and `FeatureProcessor` classes for data transformation and loading.
- **03_PYTORCH_MODEL_INSTRUCTIONS.md**: Generates the PyTorch `nn.Module` for the additive 5-axis model architecture.
- **04_MODEL_TRAINING_INSTRUCTIONS.md**: Generates the main script for training, validation, and saving model artifacts.
- **05_SAGEMAKER_DEPLOYMENT_INSTRUCTIONS.md**: Generates the inference script and a deployment utility for AWS SageMaker.
- **06_STREAMLIT_APP_INSTRUCTIONS.md**: Generates the Streamlit web application for user interaction.

### Code Files

- **Naive_NN_for_AirBNB_NYC.ipynb**: Initial tabular neural network experiments using fastai.
- **environment.yml**: Conda environment with Python 3.12, PySpark, PyTorch, and AWS tools.

### Data Structure

- **data/**: Contains NYC and Toronto Airbnb datasets, plus data dictionaries.
- **tinkering/**: Exploratory Jupyter notebooks for initial data analysis.

### Project Organization

- **notebooks/**, **scripts/**, **sql/**, **src/**: Organized directories for different code types, ready for development.

## Architecture Summary

The project implements an explainable pricing system with:

1. **5-Axis Price Decomposition**: Location, Size/Capacity, Amenities, Quality/Reputation, Seasonality
2. **Two-Stage Comparable Retrieval**: Multi-index ANN search + dynamic weighted re-ranking
3. **Market-Based Target Engineering**: Review activity as occupancy proxy for sample weighting
4. **AWS Cloud Deployment**: SageMaker for ML serving, Streamlit on App Runner for UI
5. **Transparent Explanations**: Price breakdowns and evidence-based comparable listings

This repository serves as a comprehensive foundation for building and deploying an interpretable machine learning system for short-term rental pricing optimization.
