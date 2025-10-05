# PYTORCH-DATASET-INSTRUCTIONS.md

## 1. Objective

This specification details the creation of a Python script that defines a custom PyTorch `Dataset` class and associated `DataLoader` functions. The purpose of this module is to load the pre-processed data and artifacts created by `01_create_modeling_dataset.py`, perform the final on-the-fly tensor transformations, and serve batches of model-ready data for training, validation, and testing.

## 2. File and Path Specifications

### 2.1. Script Location and Name

* **Location:** `/Fall-2025-Team-Big-Data/src/`
* **Name:** `data_loader.py`

### 2.2. Input Artifacts

This script directly depends on the outputs from the data cleaning script.

* **Modeling Data:** `/Fall-2025-Team-Big-Data/data/processed/nyc/nyc-modeling-data-YYYY-MM.parquet`
* **Feature Artifacts:** `/Fall-2025-Team-Big-Data/data/processed/nyc/nyc-feature-artifacts-YYYY-MM.json`

### 2.3. Core Components to Implement

The `data_loader.py` script must contain two main components:

1. A class named `AirbnbNightlyPriceDataset(torch.utils.data.Dataset)`.
2. A function named `create_dataloaders(data_path, artifacts_path, batch_size)`.

## 3. `AirbnbNightlyPriceDataset` Class Implementation

This class is the core of the script. It will handle loading, indexing, and transforming the data.

### 3.1. `__init__(self, data_path, artifacts_path)` Method

* **Purpose:** To load all necessary data and artifacts into memory.
* **Implementation Steps:**
    1. Load the Parquet file from `data_path` into a Pandas DataFrame and store it as `self.data`.
    2. Load the JSON file from `artifacts_path` into a dictionary and store it as `self.artifacts`.
    3. For convenience, create separate instance attributes for the vocabularies and imputation values:
        * `self.vocabs = self.artifacts['categorical_vocabularies']`
        * `self.imputation_values = self.artifacts['imputation_values']`

### 3.2. `__len__(self)` Method

* **Purpose:** To return the total number of samples in the dataset.
* **Implementation:** `return len(self.data)`

### 3.3. `__getitem__(self, idx)` Method

* **Purpose:** To retrieve a single sample from the dataset at index `idx`, perform all necessary transformations, and return a dictionary of tensors. This is the most critical method.
* **Implementation Steps:**
    1. Retrieve the data row at index `idx`: `row = self.data.iloc[idx]`.
    2. Create an empty dictionary to hold the final tensors: `model_inputs = {}`.
    3. **Process each axis according to the `EMBEDDINGS.md` specification:**

        * **Location Axis:**
            * `coords = [row['latitude'], row['longitude']]`
            * Apply **Positional (Cyclical) Encoding** to `coords` with `L=8` to create a 32-dim tensor.
            * `neighbourhood_idx = self.vocabs['neighbourhood_cleansed'][row['neighbourhood_cleansed']]` (Look up the integer index).
            * Concatenate them: `model_inputs['location'] = torch.cat([...])` -> `(48,)` tensor.

        * **Size & Capacity Axis:**
            * Look up integer indexes for `property_type`, `bedrooms`, `beds`, `bathrooms_numeric` using `self.vocabs`.
            * Perform **One-Hot Encoding** for `room_type` and `bathrooms_type`.
            * Apply **Standardization** to `accommodates` (NOTE: The scaler must be fitted on the training set and saved as part of the artifacts in the previous step. For simplicity now, we can omit this and address it during the training script implementation).
            * Concatenate all resulting vectors: `model_inputs['size_capacity'] = torch.cat([...])` -> `(27,)` tensor.

        * **Quality & Reputation Axis:**
            * Standardize the 8 rating/rate features.
            * Apply `log(1 + x)` and then standardize `number_of_reviews`.
            * One-hot encode the 3 boolean features.
            * Concatenate all: `model_inputs['quality_reputation'] = torch.cat([...])` -> `(15,)` tensor.

        * **Amenities Axis:**
            * `amenities_text = row['amenities']` (Clean the string by removing brackets, quotes, etc.).
            * This text **must not** be tokenized here. The raw cleaned text should be stored. The `DataLoader`'s `collate_fn` is the correct place for batch tokenization. Store it as `model_inputs['amenities_text'] = amenities_text`.

        * **Seasonality Axis:**
            * `month = row['month']`
            * Apply **Cyclical Encoding** (`sin`, `cos`) to create a 2-dim tensor.
            * `model_inputs['seasonality'] = ...` -> `(2,)` tensor.

    4. **Add Target and Weight:**
        * `target = torch.tensor(row['target_price'], dtype=torch.float32)`
        * `weight = torch.tensor(row['estimated_occupancy_rate'], dtype=torch.float32)`

    5. **Return:** A single dictionary containing the model inputs, the target, and the weight: `{'inputs': model_inputs, 'target': target, 'weight': weight}`.

## 4. `create_dataloaders` Function

* **Purpose:** To instantiate the `Dataset`, split it into training, validation, and test sets, and create `DataLoader` objects for each.
* **Function Signature:** `create_dataloaders(data_path, artifacts_path, batch_size, train_split=0.8, val_split=0.1)`
* **Implementation Steps:**
    1. Instantiate the full dataset: `full_dataset = AirbnbNightlyPriceDataset(data_path, artifacts_path)`.
    2. Perform a standard train-validation-test split on the dataset indices.
    3. Create `torch.utils.data.Subset` objects for train, val, and test sets using the split indices.
    4. Instantiate `DataLoader` for each subset.
    5. **Crucially, the `DataLoader` must use a custom `collate_fn`.** This function will receive a list of dictionary samples from `__getitem__`. It is responsible for:
        * Batching all the pre-computed tensors for location, size, etc.
        * Taking the list of `amenities_text` strings, passing them to a pre-trained tokenizer (e.g., from `BAAI/bge-small-en-v1.5`), and creating a batched tensor of `input_ids` and `attention_mask`. This is the standard and most efficient way to handle text tokenization.
    6. The function should return the three `DataLoader` objects: `(train_loader, val_loader, test_loader)`.
