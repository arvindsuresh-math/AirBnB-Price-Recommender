# `02_PYTORCH_DATASET_INSTRUCTIONS.md`

### LLM Agent Pre-computation Instructions

To enrich your context, first read and fully comprehend the following project design documents in order:

1. `REPO-SUMMARY.md`: For the overall project structure and goals.
2. `DATASET-SCHEMA.md`: For the exact schema of the pandas DataFrame that will be input to the classes you will generate.
3. `EMBEDDINGS.md`: For the specific feature representation strategy (e.g., which columns are categorical, which are continuous, what transformations to apply). This is your primary source for implementation logic.
4. `MODELING.md`: To understand the model architecture that will consume the data produced by your classes.

### Primary Objective

Your task is to generate a single Python script located at `src/dataset.py`. This script will contain two primary classes:

1. **`FeatureProcessor`**: A class to learn transformations from the training data (e.g., vocabularies, scalers) and apply them consistently to any dataset partition (train, validation, inference).
2. **`AirbnbDataset`**: A standard PyTorch `Dataset` class that uses a fitted `FeatureProcessor` and a sentence transformer to convert raw DataFrame rows into tensors ready for the model.

---

## 1. `FeatureProcessor` Class Specification

This class encapsulates all stateful feature engineering logic.

### 1.1. Class Structure and Methods

Generate the class `FeatureProcessor` with the following methods and attributes.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class FeatureProcessor:
    def __init__(self):
        # Dictionaries to hold fitted objects
        self.vocabs = {}
        self.scalers = {}

        # Pre-defined column groups based on EMBEDDINGS.md
        self.high_card_categorical_cols = [
            'neighbourhood_cleansed', 'property_type', 'bedrooms', 'beds', 'bathrooms_numeric'
        ]
        self.low_card_categorical_cols = ['room_type', 'bathrooms_type'] # To be One-Hot Encoded later
        self.boolean_cols = ['host_is_superhost', 'host_identity_verified', 'instant_bookable']
        self.numerical_cols_to_standardize = [
            'accommodates', 'review_scores_rating', 'review_scores_cleanliness',
            'review_scores_checkin', 'review_scores_communication',
            'review_scores_location', 'review_scores_value',
            'host_response_rate', 'host_acceptance_rate'
        ]
        self.numerical_col_to_log_standardize = 'number_of_reviews_ltm'
        # Passthrough columns
        self.passthrough_cols = ['amenities', 'target_price', 'estimated_occupancy_rate']

    def fit(self, dataframe: pd.DataFrame):
        # Logic to learn vocabs and scalers
        ...

    def transform(self, dataframe: pd.DataFrame) -> dict:
        # Logic to apply transformations row-by-row
        ...

    def save(self, filepath: str):
        # Logic to save the fitted state
        ...

    @classmethod
    def load(cls, filepath: str):
        # Logic to load a fitted processor
        ...
```

### 1.2. Method Implementation Details

**`fit(self, dataframe)`:**

* This method learns from the **training** DataFrame and populates `self.vocabs` and `self.scalers`.
* **For `self.high_card_categorical_cols`**:
  * Iterate through each column name in the list.
  * Get the unique values from the DataFrame column.
  * Create a sorted list of unique values.
  * Create a vocabulary dictionary mapping each value to an integer index. The mapping must reserve index `0` for an unknown token `"<UNK>"`.
  * Store this dictionary in `self.vocabs` with the column name as the key.
* **For `self.numerical_cols_to_standardize`**:
  * Iterate through each column name in the list.
  * Instantiate a `sklearn.preprocessing.StandardScaler`.
  * Fit the scaler on the column's data (reshaped via `.values.reshape(-1, 1)`).
  * Store the fitted scaler object in `self.scalers` with the column name as the key.
* **For `self.numerical_col_to_log_standardize`**:
  * Apply a log1p transform to the column (`np.log1p(dataframe[self.numerical_col_to_log_standardize])`).
  * Fit a `StandardScaler` on this transformed data.
  * Store the fitted scaler in `self.scalers` with the column name as the key.

**`transform(self, dataframe)`:**

* This method applies the learned transformations to a DataFrame.
* It must return a dictionary of lists, where each key is a feature name and the value is a list of the processed feature values for all rows.
* Initialize an empty dictionary `processed_data` with keys for all feature columns.
* Iterate through each row of the input `dataframe` (`for index, row in dataframe.iterrows()`).
* **For `self.high_card_categorical_cols`**: Use the learned vocab from `self.vocabs` to map the string to an integer. Use `.get(value, 0)` to default to the `"<UNK>"` index if the value is not in the vocabulary.
* **For `self.low_card_categorical_cols` and `self.boolean_cols`**: Pass the raw values through. They will be handled by the `Dataset` class.
* **For `self.numerical_cols_to_standardize`**: Apply the corresponding fitted scaler from `self.scalers` using the `.transform()` method.
* **For `self.numerical_col_to_log_standardize`**: First apply `np.log1p`, then apply the fitted scaler's `.transform()` method.
* **Cyclical Features (`month`, `latitude`, `longitude`):**
  * `month`: `sin = np.sin(2 * np.pi * row['month'] / 12)`, `cos = np.cos(2 * np.pi * row['month'] / 12)`. Append both to the processed data.
  * `latitude`/`longitude`: Use a period of 180 for latitude and 360 for longitude. Apply the same sin/cos transformation.
* **Passthrough Features**: Append the raw values for `amenities`, `target_price`, and `estimated_occupancy_rate`.

**`save(self, filepath)` and `load(cls, filepath)`:**

* Implement saving and loading the entire `FeatureProcessor` instance using `joblib.dump` and `joblib.load`. This will preserve the fitted vocabs and scalers.

---

## 2. `AirbnbDataset` Class Specification

This class will interface between the pre-processed data and the PyTorch model.

### 2.1. Class Structure and Methods

Generate the class `AirbnbDataset` inheriting from `torch.utils.data.Dataset`.

```python
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import numpy as np

class AirbnbDataset(Dataset):
    def __init__(self, processed_data: dict, amenities_transformer_name: str):
        # Store processed data and initialize sentence transformer
        ...

    def __len__(self):
        # Return the total number of samples
        ...

    def __getitem__(self, idx: int) -> dict:
        # Retrieve, process, and return one sample
        ...
```

### 2.2. Method Implementation Details

**`__init__(self, processed_data, amenities_transformer_name)`:**

* Store the `processed_data` dictionary (the output from `FeatureProcessor.transform`).
* Load the sentence transformer model: `self.amenities_model = SentenceTransformer(amenities_transformer_name)`.
* Store the total number of samples for `__len__`.
* Convert the low-cardinality categorical features into one-hot encodings and store them. Use `pd.get_dummies` for this.

**`__getitem__(self, idx)`:**

* This method must return a dictionary of tensors.
* Create an empty dictionary `sample_tensors`.
* **High-Cardinality Categorical**: Retrieve the integer index and convert to a `torch.LongTensor`.
* **One-Hot Encoded Categorical/Boolean**: Retrieve the pre-computed one-hot encoded vector for the sample at `idx` and convert to a `torch.FloatTensor`.
* **Numerical**: Retrieve the scaled value and convert to a `torch.FloatTensor`.
* **Cyclical**: Retrieve the sin/cos pairs, stack them into a single vector, and convert to a `torch.FloatTensor`.
* **Amenities**:
    1. Retrieve the raw amenities string for the sample at `idx`.
    2. Convert the string representation of a list into an actual Python list (e.g., using `ast.literal_eval`).
    3. Format it into the descriptive sentence: `"This listing has the following amenities: " + ", ".join(amenities_list)`.
    4. Use `self.amenities_model.encode()` with `convert_to_tensor=True` on this sentence to get the embedding tensor.
* **Target and Weight**: Retrieve `target_price` and `estimated_occupancy_rate` and convert them to `torch.FloatTensor`.
* Return the final `sample_tensors` dictionary. The keys must be descriptive (e.g., `'neighbourhood_cleansed'`, `'location_coords'`, `'amenities_embedding'`).

---

## 3. Verification and Unit Tests

Generate a test script at `tests/test_dataset.py` using the `pytest` framework.

### Test Setup (`pytest.fixture`)

* Create a fixture `sample_dataframe()` that returns a small pandas DataFrame (3-4 rows) with representative data for all columns used by the `FeatureProcessor`. Include a known categorical value, an unknown one, and NaNs in a numerical column.

### Test Cases for `FeatureProcessor`

1. **`test_fit(sample_dataframe)`**:
    * Instantiate `FeatureProcessor` and fit it on the sample data.
    * Assert that `processor.vocabs['neighbourhood_cleansed']` contains the known value and `"<UNK>"`.
    * Assert that `processor.scalers['accommodates']` is a fitted `StandardScaler` instance.
2. **`test_transform(sample_dataframe)`**:
    * Get a fitted processor.
    * Transform the data.
    * Assert the output is a dictionary of lists.
    * Check a specific value: assert the integer for the known neighborhood is correct. Assert the integer for the unknown neighborhood is `0`. Assert that a numerical value has been scaled (i.e., is not equal to its original value).
3. **`test_save_load(sample_dataframe)`**:
    * Fit a processor, save it to a temporary file, and load it back using the class method.
    * Assert that the loaded processor's `vocabs` and `scalers` are identical to the original's.

### Test Cases for `AirbnbDataset`

1. **`test_getitem(sample_dataframe)`**:
    * Fit a `FeatureProcessor` and transform the sample data.
    * Instantiate `AirbnbDataset` with the processed data and `'all-MiniLM-L6-v2'` as the transformer name.
    * Get the first item: `sample = dataset[0]`.
    * Assert that `sample` is a dictionary.
    * Assert that `sample['amenities_embedding']` is a `torch.Tensor` with shape `(384,)`.
    * Assert that `sample['neighbourhood_cleansed']` is a scalar `torch.LongTensor`.
    * Assert that `sample['accommodates']` is a scalar `torch.FloatTensor`.
