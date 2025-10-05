# PYTORCH-MODEL-INSTRUCTIONS.md

## 1. Objective

This specification details the creation of a Python script that defines the complete PyTorch model architecture. The script will implement the **Phase 1 Simple Additive Baseline Model** as a primary `nn.Module` class. It will also contain commented-out code and clear instructions for enabling the **Phase 2 Enhancements** (fine-tuning and the Transformer Interaction Layer) at a later stage.

## 2. File and Path Specifications

### 2.1. Script Location and Name

* **Location:** `/Fall--2025-Team-Big-Data/src/`
* **Name:** `model.py`

### 2.2. Input Artifacts

This script depends on the vocabulary sizes from the feature artifacts file to correctly size the embedding layers.

* **Feature Artifacts:** `/Fall-2025-Team-Big-Data/data/processed/nyc/nyc-feature-artifacts-YYYY-MM.json`

### 2.3. Core Components to Implement

The `model.py` script must contain one main class:

* `ExplainablePriceModel(nn.Module)`

## 3. `ExplainablePriceModel` Class Implementation

This class will encapsulate the entire architecture, from embeddings to the final prediction.

### 3.1. `__init__(self, artifacts_path)` Method

* **Purpose:** To define and initialize all learnable layers of the network.
* **Implementation Steps:**
    1. Load the `artifacts_path` JSON file to get vocabulary sizes for all categorical features.
    2. **Instantiate Embedding Layers (From Scratch):**
        * `self.embed_neighbourhood = nn.Embedding(num_embeddings=vocab_size['neighbourhood_cleansed'], embedding_dim=16)`
        * `self.embed_property_type = nn.Embedding(num_embeddings=vocab_size['property_type'], embedding_dim=8)`
        * `self.embed_bedrooms = nn.Embedding(num_embeddings=vocab_size['bedrooms'], embedding_dim=4)`
        * `self.embed_beds = nn.Embedding(num_embeddings=vocab_size['beds'], embedding_dim=4)`
        * `self.embed_bathrooms_numeric = nn.Embedding(num_embeddings=vocab_size['bathrooms_numeric'], embedding_dim=4)`
    3. **Instantiate Amenities Transformer (Pre-trained):**
        * Load the `bge-small-en-v1.5` model from the `transformers` library (e.g., using `AutoModel.from_pretrained(...)`). Store it as `self.amenities_transformer`.
        * **CRITICAL (Phase 1):** Iterate through all parameters of `self.amenities_transformer` and set `param.requires_grad = False` to **freeze** the model.
    4. **Instantiate MLP Sub-Networks:**
        * Define each of the 5 sub-networks as `nn.Sequential` modules, following the exact architecture from `MODELING.md`.
            * `self.location_subnet = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))`
            * `self.size_capacity_subnet = nn.Sequential(...)`
            * `self.quality_reputation_subnet = nn.Sequential(...)`
            * `self.amenities_subnet = nn.Sequential(nn.Linear(384, 1))`
            * `self.seasonality_subnet = nn.Sequential(...)`
    5. **Instantiate Global Bias:**
        * `self.global_bias = nn.Parameter(torch.zeros(1))`

### 3.2. `forward(self, batch)` Method

* **Purpose:** To define the computation performed at every call of the model.
* **Input:** The `batch` dictionary produced by the `DataLoader`'s `collate_fn`. It will contain a nested dictionary `batch['inputs']` with tensors for each axis.
* **Implementation Steps:**

    1. **Prepare Input Tensors for Each Axis:**
        * This step mirrors the logic in `PYTORCH-DATASET-INSTRUCTIONS.md`, but it is performed on the batched tensors from the `collate_fn`.
        * **Location:**
            * Look up the neighborhood embedding: `emb_hood = self.embed_neighbourhood(batch['inputs']['neighbourhood_idx'])`.
            * Concatenate with the pre-computed positional encoding for coordinates: `location_input_tensor = torch.cat([batch['inputs']['coords_encoded'], emb_hood], dim=1)`.
        * **Size & Capacity:**
            * Look up all required embeddings.
            * Concatenate with the one-hot encoded and numerical features: `size_input_tensor = torch.cat([...], dim=1)`.
        * **Quality & Reputation:**
            * `quality_input_tensor = batch['inputs']['quality_reputation']` (This tensor is already fully formed).
        * **Amenities:**
            * Pass the tokenized batch through the frozen transformer: `amenities_embedding = self.amenities_transformer(input_ids=batch['inputs']['amenities_input_ids'], attention_mask=batch['inputs']['amenities_attention_mask']).last_hidden_state[:, 0]`. (Use the `[CLS]` token embedding).
        * **Seasonality:**
            * `seasonality_input_tensor = batch['inputs']['seasonality']` (Already fully formed).

    2. **Calculate Per-Axis Contributions:**
        * `p_loc = self.location_subnet(location_input_tensor)`
        * `p_size = self.size_capacity_subnet(size_input_tensor)`
        * `p_qual = self.quality_reputation_subnet(quality_input_tensor)`
        * `p_amen = self.amenities_subnet(amenities_embedding)`
        * `p_seas = self.seasonality_subnet(seasonality_input_tensor)`

    3. **Aggregate for Final Prediction:**
        * `predicted_price = self.global_bias + p_loc + p_size + p_qual + p_amen + p_seas`

    4. **Return:** A dictionary containing the final prediction and all intermediate contributions for explainability.

        ```python
        return {
            "predicted_price": predicted_price.squeeze(-1),
            "contributions": {
                "global_bias": self.global_bias.expand(predicted_price.size(0)),
                "location": p_loc.squeeze(-1),
                "size_capacity": p_size.squeeze(-1),
                "quality_reputation": p_qual.squeeze(-1),
                "amenities": p_amen.squeeze(-1),
                "seasonality": p_seas.squeeze(-1)
            }
        }
        ```

## 4. Phase 2 Enhancements (To Be Implemented Later)

The script must include clear comments indicating where to make changes for the advanced architectures.

* **For Fine-Tuning (Enhancement 2.i):**
  * Add a comment in the `__init__` method: `# PHASE 2.i: To enable fine-tuning, comment out the following loop that freezes the transformer weights.`

* **For Transformer Interaction Layer (Enhancement 2.ii):**
  * Add a commented-out section in `__init__` to define the projection layers and the `nn.TransformerEncoderLayer`.
  * Add a commented-out block in the `forward` pass showing the alternative data flow: `Embed -> Project -> Stack -> Transformer -> Unstack -> Prediction Heads -> Aggregate`. This will make it easy to switch between architectures.

## 5. Testing and Verification

To verify the model's architectural integrity and learning capability, a dedicated unit test script must be created.

1. **Test Script:**
    * **Location:** `/Fall-2025-Team-Big-Data/tests/`
    * **Name:** `test_model.py`
2. **Implementation:** The test script should contain at least two test functions:

    * **`test_forward_pass_shapes()`:**
        1. Instantiate the `ExplainablePriceModel`.
        2. Create a "dummy batch" of tensors with the exact shapes and dtypes produced by the `DataLoader`.
        3. Pass the dummy batch through `model.forward()`.
        4. Assert that the call completes without raising an error.
        5. Assert that the output dictionary contains the `predicted_price` key.
        6. Assert that the `predicted_price` tensor has the correct shape: `(batch_size,)`.

    * **`test_model_can_learn()` (Overfit-One-Batch Test):**
        1. This is a critical sanity check to prove the model is trainable.
        2. Instantiate the `Dataset` and `DataLoader`. Get a single, real batch of data.
        3. Instantiate the model and an optimizer (`torch.optim.AdamW`).
        4. Run a small training loop for 50-100 iterations, training *only on this single batch*.
        5. In each step, calculate the loss and perform backpropagation.
        6. After the loop, assert that the final loss is significantly lower than the initial loss and has converged to a value very close to zero (e.g., `assert final_loss < 1e-3`).
