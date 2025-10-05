# `03_PYTORCH_MODEL_INSTRUCTIONS.md`

### LLM Agent Pre-computation Instructions

To enrich your context, first read and fully comprehend the following project design documents in order:

1. `REPO-SUMMARY.md`: For the overall project structure.
2. `FEATURE-AXES.md`: For the logical grouping of features into the 5 axes which your model must implement.
3. `EMBEDDINGS.md`: For the **exact input dimensions** of each of the 5 sub-networks, found in the "Final Sub-Network Inputs" table.
4. `MODELING.md`: For the **exact layer architecture** (layer types, dimensions, activation functions) of the "Phase 1: The Simple Additive Baseline Model". This is your primary architectural reference.
5. `02_PYTORCH_DATASET_INSTRUCTIONS.md`: To understand the structure of the batch dictionary that will be the input to your model's `forward` method.

### Primary Objective

Your task is to generate a single Python script at `src/model.py`. This script will contain a single PyTorch `nn.Module` subclass named `AdditiveAxisModel`. This class will implement the complete Phase 1 additive baseline model, including all embedding layers, sub-networks, and the final aggregation logic.

---

## 1. Model Architecture Specification (Phase 1)

The model's core principle is additive decomposition. It consists of 5 independent sub-networks (one for each feature axis) whose scalar outputs are summed together with a global bias term to produce the final price prediction.

### 1.1. Class Structure

Generate the class `AdditiveAxisModel` inheriting from `torch.nn.Module`.

```python
import torch
import torch.nn as nn

class AdditiveAxisModel(nn.Module):
    def __init__(self, vocab_sizes: dict, embedding_dims: dict):
        super().__init__()
        # ... Embedding layers, sub-networks, and bias will be defined here ...

    def forward(self, batch: dict) -> dict:
        # ... Logic to process the batch and return predictions ...
```

### 1.2. `__init__(self, vocab_sizes: dict, embedding_dims: dict)` Method

The constructor will receive two dictionaries:

* `vocab_sizes`: Maps high-cardinality categorical feature names to their vocabulary size (e.g., `{'neighbourhood_cleansed': 251}`).
* `embedding_dims`: Maps high-cardinality categorical feature names to their desired embedding dimension (e.g., `{'neighbourhood_cleansed': 16}`).

**Define the following layers precisely:**

1. **Global Bias:**
    * `self.global_bias = nn.Parameter(torch.zeros(1))`

2. **Embedding Layers (for High-Cardinality Categorical Features):**
    * `self.embedding_neighbourhood = nn.Embedding(vocab_sizes['neighbourhood_cleansed'], embedding_dims['neighbourhood_cleansed'])`
    * `self.embedding_property_type = nn.Embedding(vocab_sizes['property_type'], embedding_dims['property_type'])`
    * `self.embedding_bedrooms = nn.Embedding(vocab_sizes['bedrooms'], embedding_dims['bedrooms'])`
    * `self.embedding_beds = nn.Embedding(vocab_sizes['beds'], embedding_dims['beds'])`
    * `self.embedding_bathrooms_numeric = nn.Embedding(vocab_sizes['bathrooms_numeric'], embedding_dims['bathrooms_numeric'])`

3. **MLP Sub-Networks (using `nn.Sequential`):**
    * **Location Sub-network:**
        * Input Dimension: **48** (`16` from neighborhood embedding + `32` from cyclical lat/lon).
        * `self.location_mlp = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))`
    * **Size & Capacity Sub-network:**
        * Input Dimension: **27** (`8`+`4`+`4`+`4` from embeddings + `4`+`2` from OHE + `1` from numerical).
        * `self.size_capacity_mlp = nn.Sequential(nn.Linear(27, 32), nn.ReLU(), nn.Linear(32, 1))`
    * **Quality & Reputation Sub-network:**
        * Input Dimension: **15** (`8`+`1` from numerical + `2`+`2`+`2` from OHE).
        * `self.quality_reputation_mlp = nn.Sequential(nn.Linear(15, 32), nn.ReLU(), nn.Linear(32, 1))`
    * **Amenities Sub-network:**
        * Input Dimension: **384** (from the pre-trained sentence transformer).
        * `self.amenities_mlp = nn.Sequential(nn.Linear(384, 1))`
    * **Seasonality Sub-network:**
        * Input Dimension: **2** (from cyclical month sin/cos).
        * `self.seasonality_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))`

### 1.3. `forward(self, batch: dict) -> dict` Method

This method orchestrates the data flow through the model.

* **Input Contract:** The `batch` argument is a dictionary of tensors produced by `AirbnbDataset`.
* **Output Contract:** The method **must** return a dictionary with two keys:
  * `"predicted_price"`: A tensor of shape `(batch_size, 1)` containing the final summed price.
  * `"breakdown"`: A dictionary containing the scalar price contributions from each of the 5 axes and the global bias. E.g., `{'location': tensor, 'size_capacity': tensor, ...}`.

**Implement the following step-by-step logic:**

1. **Assemble Axis 1 (Location) Input:**
    * Get the `neighbourhood_cleansed` tensor from the batch and pass it through `self.embedding_neighbourhood`.
    * Concatenate this embedding with the `location_coords_cyclical` tensor from the batch along `dim=1`. The result is `location_input_tensor` of shape `(batch_size, 48)`.

2. **Assemble Axis 2 (Size & Capacity) Input:**
    * Get and embed `property_type`, `bedrooms`, `beds`, and `bathrooms_numeric`.
    * Get the one-hot encoded tensors `room_type_ohe`, `bathrooms_type_ohe`, and the numerical `accommodates` tensor from the batch.
    * Concatenate all resulting tensors along `dim=1` to form `size_capacity_input_tensor` of shape `(batch_size, 27)`.

3. **Assemble Axis 3 (Quality & Reputation) Input:**
    * Get all 8 numerical review/host-rate tensors and the 3 boolean one-hot encoded tensors from the batch.
    * Concatenate them along `dim=1` to form `quality_reputation_input_tensor` of shape `(batch_size, 15)`.

4. **Assemble Axis 4 (Amenities) Input:**
    * The `amenities_embedding` tensor from the batch is the complete input. No further processing is needed. Shape is `(batch_size, 384)`.

5. **Assemble Axis 5 (Seasonality) Input:**
    * The `month_cyclical` tensor from the batch is the complete input. No further processing is needed. Shape is `(batch_size, 2)`.

6. **Calculate Individual Contributions:**
    * `p_location = self.location_mlp(location_input_tensor)`
    * `p_size_capacity = self.size_capacity_mlp(size_capacity_input_tensor)`
    * `p_quality_reputation = self.quality_reputation_mlp(quality_reputation_input_tensor)`
    * `p_amenities = self.amenities_mlp(amenities_embedding_tensor)`
    * `p_seasonality = self.seasonality_mlp(month_cyclical_tensor)`

7. **Aggregate Final Price:**
    * `predicted_price = self.global_bias + p_location + p_size_capacity + p_quality_reputation + p_amenities + p_seasonality`

8. **Construct and Return Output Dictionary:**
    * Create the `breakdown` dictionary.
    * Create the final output dictionary and return it.

---

## 2. Verification and Unit Tests

Generate a test script at `tests/test_model.py` using the `pytest` framework.

### Test Setup (`pytest.fixture`)

* Create a fixture `dummy_model_input_batch()` that returns a dictionary of tensors mimicking a single batch from the `AirbnbDataset`.
  * Set `batch_size = 4`.
  * For each expected key in the batch (e.g., `neighbourhood_cleansed`, `location_coords_cyclical`, etc.), create a `torch.Tensor` with the correct shape and `dtype`.
  * Example: `'neighbourhood_cleansed': torch.randint(0, 251, (batch_size,))`, `'location_coords_cyclical': torch.randn(batch_size, 32)`.
  * Ensure all keys required by the `forward` method are present.

### Test Cases

1. **`test_model_instantiation()`**:
    * Define sample `vocab_sizes` and `embedding_dims` dictionaries.
    * Instantiate `AdditiveAxisModel(vocab_sizes, embedding_dims)`.
    * Assert that the model is created without error.

2. **`test_forward_pass(dummy_model_input_batch)`**:
    * Instantiate the model as in the previous test.
    * Pass the dummy batch fixture to `model.forward()`.
    * **Assert 1 (Execution):** The primary assertion is that the forward pass completes without any runtime or shape-mismatch errors.
    * **Assert 2 (Output Keys):** Assert that the returned dictionary contains the keys `"predicted_price"` and `"breakdown"`.
    * **Assert 3 (Output Shapes):**
        * Assert that `output['predicted_price'].shape` is `(batch_size, 1)`.
        * Assert that `output['breakdown']` is a dictionary and that each of its 5 value tensors also has the shape `(batch_size, 1)`.
    * **Assert 4 (Gradient Flow):** Assert that `output['predicted_price'].requires_grad` is `True`, confirming that the computation graph is connected for backpropagation.
