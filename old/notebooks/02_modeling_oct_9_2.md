#### **0. Setup and Installations**


```python
# --- Hugging Face Authentication (using Colab Secrets) ---
from google.colab import userdata
from huggingface_hub import login
print("Attempting Hugging Face login...")
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Could not log in. Please ensure 'HF_TOKEN' is a valid secret. Error: {e}")
```

    Attempting Hugging Face login...
    Hugging Face login successful.



```python
# --- Mount Google Drive ---
from google.colab import drive
print("Mounting Google Drive...")
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except Exception as e:
    print(f"Could not mount Google Drive. Error: {e}")
```

    Mounting Google Drive...
    Mounted at /content/drive
    Google Drive mounted successfully.



```python
# --- Install Dependencies ---
!pip install pandas
!pip install pyarrow
!pip install sentence-transformers
!pip install scikit-learn
!pip install torch
!pip install tqdm
```

#### **2. Configuration and Helper Functions**

This section contains all hyperparameters and the new functions for data loading and splitting.


```python
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import json
import time
```


```python
# --- Seeding function for reproducibility ---
def set_seed(seed: int):
    """Sets the seed for all relevant RNGs to ensure reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        # These are crucial for reproducibility on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}.")

class Config:
    # --- Data and Environment ---
    CITY: str = "nyc"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    DRIVE_SAVE_PATH: str = "/content/drive/MyDrive/Colab_Notebooks/Airbnb_Project/"

    # --- Data Pre-processing ---
    VAL_SIZE: float = 0.2

    # --- Reproducibility ---
    SEED: int = 42 # Master seed for the entire experiment

    # --- Model Training ---
    BATCH_SIZE: int = 1024
    LEARNING_RATE: float = 1e-3
    N_EPOCHS: int = 30

    # --- Early Stopping ---
    EARLY_STOPPING_PATIENCE: int = 5
    EARLY_STOPPING_MIN_DELTA: float = 1e-4 # For RW-MSLE, the accuracy metric

    # --- Logging ---
    LOG_EVERY_N_STEPS: int = 10
```

### **2. Data Loading and Splitting**

This function handles loading, outlier removal, and the 3-way stratified split.


```python
def load_and_split_data(config: Config):
    """
    Loads data, removes price outliers, and performs a 3-way stratified split.
    """
    dataset_filename = f"{config.CITY}_final_modeling_dataset.parquet"
    dataset_path = f"./{dataset_filename}" # Assumes file in root Colab runtime

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"'{dataset_filename}' not found. Please upload the file to the Colab Runtime.")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_parquet(dataset_path)

    # Remove price outliers (top/bottom 1%)
    price_q01 = df['target_price'].quantile(0.01)
    price_q99 = df['target_price'].quantile(0.99)
    df = df[(df['target_price'] >= price_q01) & (df['target_price'] <= price_q99)].copy()
    print(f"Removed price outliers. New size: {len(df):,} records.")

    # Create bins for stratifying continuous price
    df['price_bin'] = pd.cut(df['target_price'], bins=10, labels=False)

    # Create a combined key for 3-way stratification
    stratify_key = (
        df['neighbourhood_cleansed'].astype(str) + '_' +
        df['month'].astype(str) + '_' +
        df['price_bin'].astype(str)
    )

    # Handle small strata (<2 members)
    strata_counts = stratify_key.value_counts()
    valid_strata = strata_counts[strata_counts >= 2].index
    df_filtered = df[stratify_key.isin(valid_strata)].copy()
    print(f"Removed small strata. New size: {len(df_filtered):,} records.")

    # Perform stratified split
    train_indices, val_indices = train_test_split(
        df_filtered.index,
        test_size=config.VAL_SIZE,
        random_state=config.SEED,
        stratify=stratify_key[df_filtered.index]
    )

    train_df = df_filtered.loc[train_indices].copy().reset_index(drop=True)
    val_df = df_filtered.loc[val_indices].copy().reset_index(drop=True)

    print(f"Split complete. Training: {len(train_df):,}, Validation: {len(val_df):,}")

    print("\n--- Sample Record from Training Data ---")
    # Pretty-print the first record by transposing it
    print(train_df.head(1).T)

    return train_df, val_df
```

### **3. Feature Processor**

This class encapsulates the feature engineering logic as defined in `EMBEDDINGS.md`. It learns transformations (like vocabularies and scaling parameters) from the training data via the `.fit()` method. The `.transform()` method then consistently applies these learned transformations to any dataset, preventing data leakage. This is a crucial step for creating model-ready tensors from raw dataframes.


```python
class FeatureProcessor:
    def __init__(self, embedding_dim_geo: int = 32):
        self.vocabs, self.scalers = {}, {}
        self.embedding_dim_geo = embedding_dim_geo
        self.categorical_cols = [
            "neighbourhood_cleansed",
            "property_type",
            "room_type",
            "bathrooms_type",
            "bedrooms",
            "beds",
            "bathrooms_numeric"
            ]
        self.numerical_cols = [
            "accommodates",
            "review_scores_rating",
            "review_scores_cleanliness",
            "review_scores_checkin",
            "review_scores_communication",
            "review_scores_location",
            "review_scores_value",
            "host_response_rate",
            "host_acceptance_rate"
            ]
        self.log_transform_cols = ["number_of_reviews_ltm"]
        self.boolean_cols = [
            "host_is_superhost",
            "host_identity_verified",
            "instant_bookable"
            ]

    def _create_positional_encoding(self, value, max_val):
        d = self.embedding_dim_geo
        if d % 2 != 0: raise ValueError("embedding_dim_geo must be even.")
        pe = np.zeros(d)
        position = (value / max_val) * 10000
        div_term = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
        pe[0::2] = np.sin(position * div_term)
        pe[1::2] = np.cos(position * div_term)
        return pe

    def fit(self, df: pd.DataFrame):
        for col in self.categorical_cols:
            valid_uniques = df[col].dropna().unique().tolist()
            self.vocabs[col] = {val: i for i, val in enumerate(["<UNK>"] + sorted(valid_uniques))}
        for col in self.numerical_cols + self.log_transform_cols:
            vals = np.log1p(df[col]) if col in self.log_transform_cols else df[col]
            self.scalers[col] = {'mean': vals.mean(), 'std': vals.std()}

    def transform(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        half_dim = self.embedding_dim_geo // 2
        lat_enc = df['latitude'].apply(lambda x: self._create_positional_encoding(x, 90)[:half_dim])
        lon_enc = df['longitude'].apply(lambda x: self._create_positional_encoding(x, 180)[:half_dim])

        # --- Axis 1: Location ---
        half_dim = self.embedding_dim_geo // 2
        lat_enc = df['latitude'].apply(lambda x: self._create_positional_encoding(x, 90)[:half_dim])
        lon_enc = df['longitude'].apply(lambda x: self._create_positional_encoding(x, 180)[:half_dim])
        geo_position = np.hstack([np.stack(lat_enc), np.stack(lon_enc)])
        neighbourhood = df["neighbourhood_cleansed"].apply(lambda x: self.vocabs["neighbourhood_cleansed"].get(x, 0)).values
        location_features = {"geo_position": geo_position, "neighbourhood": neighbourhood}

        # --- Axis 2: Size & Capacity ---
        size_features = {}
        for col in ["property_type", "room_type", "bathrooms_type", "bedrooms", "beds", "bathrooms_numeric"]:
            size_features[col] = df[col].apply(lambda x: self.vocabs[col].get(x, 0) if pd.notna(x) else 0).values
        size_features["accommodates"] = ((df["accommodates"] - self.scalers["accommodates"]["mean"]) / self.scalers["accommodates"]["std"]).values

        # --- Axis 3: Quality & Reputation ---
        quality_features = {}
        for col in self.numerical_cols:
            if col != "accommodates":
                quality_features[col] = ((df[col] - self.scalers[col]["mean"]) / self.scalers[col]["std"]).values
        quality_features["number_of_reviews_ltm"] = ((np.log1p(df["number_of_reviews_ltm"]) - self.scalers["number_of_reviews_ltm"]["mean"]) / self.scalers["number_of_reviews_ltm"]["std"]).values
        for col in self.boolean_cols:
            quality_features[col] = df[col].astype(float).values

        # --- Axis 5: Seasonality ---
        month_sin = np.sin(2 * np.pi * df["month"] / 12)
        month_cos = np.cos(2 * np.pi * df["month"] / 12)
        seasonality_features = {"cyclical": np.vstack([month_sin, month_cos]).T}

        return {
            "location": location_features,
            "size_capacity": size_features,
            "quality": quality_features,
            "amenities": {"text": df["amenities"].tolist()},
            "seasonality": seasonality_features,
            "target_price": df["target_price"].values, # try without log to see if training is stable
            "sample_weight": df["estimated_occupancy_rate"].values
        }
```

### **4. AirbnbDataset Class**

The PyTorch `Dataset` class, which defines how to retrieve a single item from our processed feature dictionary.


```python
class AirbnbPriceDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features
        self.n_samples = len(features['sample_weight'])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        item = {}
        # Location
        item['loc_geo_position'] = self.features['location']['geo_position'][index]
        item['loc_neighbourhood'] = self.features['location']['neighbourhood'][index]

        # Size & Capacity
        for k, v in self.features['size_capacity'].items():
            item[f'size_{k}'] = v[index]

        # Quality
        for k, v in self.features['quality'].items():
            item[f'qual_{k}'] = v[index]

        # Amenities & Seasonality
        item['amenities_text'] = self.features['amenities']['text'][index]
        item['season_cyclical'] = self.features['seasonality']['cyclical'][index]

        # Target & Weight
        item['target'] = self.features['target_price'][index]
        item['sample_weight'] = self.features['sample_weight'][index]

        return item
```

### **5. Dataloader Creation**

A function to create the `DataLoader` instances, including the custom collate function for batch tokenization.


```python
# (This cell replaces the 'preprocess_and_tensorize' and 'create_dataloaders' functions)

def preprocess_and_tensorize_CPU(processor, df):
    """
    Applies the feature processor and converts data to CPU tensors with correct dtypes.
    """
    features_cpu = processor.transform(df)
    features_tensor = {}

    for key, value in features_cpu.items():
        if key == 'amenities':
            features_tensor[key] = value # Keep raw text
        elif isinstance(value, dict):
            features_tensor[key] = {}
            for sub_key, sub_val in value.items():
                # --- CORRECTED DTYPE LOGIC ---
                # Use the lists from the processor to determine the correct dtype.
                # Categorical features need to be 'long' for embedding layers.
                # All other features (numerical, boolean, cyclical, positional) must be 'float'.
                if sub_key in processor.categorical_cols or sub_key == 'neighbourhood':
                    dtype = torch.long
                else:
                    dtype = torch.float32

                features_tensor[key][sub_key] = torch.from_numpy(sub_val).to(dtype=dtype)
        else: # Handles top-level items like target_log_price and sample_weight
            features_tensor[key] = torch.from_numpy(value).to(dtype=torch.float32)

    return features_tensor

def create_dataloaders(train_features_cpu, val_features_cpu, config: Config):
    """Creates high-performance, reproducible PyTorch DataLoaders."""

    # Tokenizer can still be on the GPU for speed, as it's used in the main process
    tokenizer_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=config.DEVICE)

    def custom_collate_fn(batch: list) -> dict:
        amenities_texts = [item.pop('amenities_text') for item in batch]
        collated_batch = {key: torch.stack([d[key] for d in batch]) for key in batch[0].keys()}
        # The tokenizer is called, but its output remains on CPU by default
        tokenized = tokenizer_model.tokenizer(
            amenities_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128
        )
        collated_batch['amenities_tokens'] = tokenized
        return collated_batch

    train_dataset = AirbnbPriceDataset(train_features_cpu)
    val_dataset = AirbnbPriceDataset(val_features_cpu)

    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        generator=g,
        pin_memory=True,
        num_workers=2
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        num_workers=2
    )
    print(f"DataLoaders created with pin_memory=True and num_workers=2.")
    return train_loader, val_loader
```

### **6. Model Architecture**

This is the `AdditiveAxisModel`, our core neural network. As detailed in `MODELING.md`, it's a multi-headed architecture where each "head" or sub-network is responsible for a distinct feature axis (Location, Size, etc.). The final price is the sum of contributions from each axis plus a global bias. This design makes the model's predictions inherently explainable.


```python
class HybridExplainableModel(nn.Module):
    def __init__(self, processor: FeatureProcessor, device: str):
        super().__init__()
        self.vocabs, self.device = processor.vocabs, device

        # --- Embedding Layers (unchanged) ---
        self.embed_neighbourhood = nn.Embedding(len(self.vocabs['neighbourhood_cleansed']), 16)
        self.embed_property_type = nn.Embedding(len(self.vocabs['property_type']), 8)
        self.embed_room_type = nn.Embedding(len(self.vocabs['room_type']), 4)
        self.embed_bathrooms_type = nn.Embedding(len(self.vocabs['bathrooms_type']), 2)
        self.embed_bedrooms = nn.Embedding(len(self.vocabs['bedrooms']), 4)
        self.embed_beds = nn.Embedding(len(self.vocabs['beds']), 4)
        self.embed_bathrooms_numeric = nn.Embedding(len(self.vocabs['bathrooms_numeric']), 4)
        self.amenities_transformer = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        for param in self.amenities_transformer.parameters(): param.requires_grad = False

        # --- 1. Attribution Heads (formerly subnets) ---
        # These predict the additive dollar contributions
        self.loc_head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))
        self.size_head = nn.Sequential(nn.Linear(27, 32), nn.ReLU(), nn.Linear(32, 1))
        self.qual_head = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 1))
        self.amenities_head = nn.Linear(384, 1)
        self.season_head = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.global_bias_head = nn.Parameter(torch.randn(1))

        # --- 2. Backbone (for accurate prediction) ---
        # It takes all features concatenated together
        backbone_input_dim = 48 + 27 + 12 + 384 + 2 # loc+size+qual+amen+season
        self.backbone = nn.Sequential(
            nn.Linear(backbone_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.to(self.device)

    def forward(self, batch: dict) -> dict:
        # --- Create feature embeddings (same as before) ---
        loc_geo = batch['loc_geo_position']
        loc_hood_embed = self.embed_neighbourhood(batch['loc_neighbourhood'])
        loc_input = torch.cat([loc_geo, loc_hood_embed], dim=1)
        size_embeds = [
            self.embed_property_type(batch['size_property_type']), self.embed_room_type(batch['size_room_type']),
            self.embed_bathrooms_type(batch['size_bathrooms_type']), self.embed_beds(batch['size_beds']),
            self.embed_bedrooms(batch['size_bedrooms']), self.embed_bathrooms_numeric(batch['size_bathrooms_numeric']),
            batch['size_accommodates'].unsqueeze(1)
        ]
        size_input = torch.cat(size_embeds, dim=1)
        qual_inputs = [
            batch[f'qual_{col}'].unsqueeze(1) for col in [
                "review_scores_rating", "review_scores_cleanliness", "review_scores_checkin",
                "review_scores_communication", "review_scores_location", "review_scores_value",
                "host_response_rate", "host_acceptance_rate", "number_of_reviews_ltm",
                "host_is_superhost", "host_identity_verified", "instant_bookable"
            ]
        ]
        qual_input = torch.cat(qual_inputs, dim=1)
        amenities_tokens = batch['amenities_tokens'].to(self.device)
        amenities_input = self.amenities_transformer(amenities_tokens)['sentence_embedding']
        season_input = batch['season_cyclical']

        # --- Backbone Prediction (for accuracy) ---
        backbone_full_input = torch.cat([
            loc_input, size_input, qual_input, amenities_input, season_input
        ], dim=1)
        q_pred = self.backbone(backbone_full_input).squeeze(-1)

        # --- Attribution Head Predictions (for explanation) ---
        p_loc = self.loc_head(loc_input)
        p_size = self.size_head(size_input)
        p_qual = self.qual_head(qual_input)
        p_amenities = self.amenities_head(amenities_input)
        p_season = self.season_head(season_input)

        return {
            "q_pred": q_pred, # Predicted log_price
            "p_loc": p_loc.squeeze(-1),
            "p_size": p_size.squeeze(-1),
            "p_qual": p_qual.squeeze(-1),
            "p_amenities": p_amenities.squeeze(-1),
            "p_season": p_season.squeeze(-1),
            "p_bias": self.global_bias_head.expand(q_pred.shape[0])
        }
```

### **7. Training Function**

This function orchestrates the training and validation loops for a given number of epochs.



```python
def evaluate_model(model, data_loader, device, config):
    """Runs a full evaluation pass for the hybrid model."""
    model.eval()
    total_backbone_loss = 0.0
    total_attribution_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor): batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items(): batch[key][sub_key] = sub_value.to(device, non_blocking=True)

            targets = batch['target']
            weights = batch['sample_weight']

            # Get all predictions
            preds = model(batch)
            q_pred = preds['q_pred']

            # --- Calculate Backbone Loss (Accuracy) ---
            target_log_price = torch.log1p(targets)
            backbone_loss = (weights * (q_pred - target_log_price)**2).mean().item()
            total_backbone_loss += backbone_loss

            # --- Calculate Attribution Loss (Explanation) ---
            p_pred = torch.exp(q_pred)
            p_sum = preds['p_bias'] + preds['p_loc'] + preds['p_size'] + preds['p_qual'] + preds['p_amenities'] + preds['p_season']
            attribution_loss = (weights * (p_sum - p_pred)**2).mean().item()
            total_attribution_loss += attribution_loss

    return {
        "backbone_loss": total_backbone_loss / len(data_loader),
        "attribution_loss": total_attribution_loss / len(data_loader)
    }

def train_model(model, train_loader, val_loader, optimizer, config):
    """
    Trains the hybrid model, logging both training and validation losses for each component.
    """
    print("\n--- Starting Model Training (Hybrid Architecture) ---")
    start_time = time.time()
    history = []
    global_step_count = 0
    patience_counter = 0
    early_stop_flag = False

    print("Performing Step 0 evaluation...")
    # Get initial validation losses
    val_losses_0 = evaluate_model(model, val_loader, config.DEVICE, config)
    # Get initial training losses (on first batch for speed)
    first_train_batch = next(iter(train_loader))
    train_losses_0 = evaluate_model(model, [first_train_batch], config.DEVICE, config)

    best_val_loss = val_losses_0['backbone_loss'] # Early stopping is based on accuracy
    best_model_state = model.state_dict()
    print("Step 0 evaluation complete.\n")

    # --- Updated Header ---
    header = (
        f"{'Steps':>5} | {'Epoch':>5} | {'Train RW-MSLE':>13} | {'Val RW-MSLE':>11} | "
        f"{'Train Attr RMSE':>15} | {'Val Attr RMSE':>13} | {'Patience':>8} | {'Elapsed Time'}"
    )
    print(header)
    print("-" * len(header))

    # --- Log Step 0 ---
    elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    step_stats_0 = {
        'Steps': 0, 'Epoch': 0.00,
        'Train RW-MSLE': np.sqrt(train_losses_0['backbone_loss']),
        'Val RW-MSLE': np.sqrt(val_losses_0['backbone_loss']),
        'Train Attr RMSE': np.sqrt(train_losses_0['attribution_loss']),
        'Val Attr RMSE': np.sqrt(val_losses_0['attribution_loss']),
        'Patience': 0, 'Elapsed Time': elapsed_time_str
    }
    history.append(step_stats_0)

    log_line_0 = (
        f"{0:>5d} | {0:>5.2f} | {step_stats_0['Train RW-MSLE']:>13.4f} | {step_stats_0['Val RW-MSLE']:>11.4f} | "
        f"{step_stats_0['Train Attr RMSE']:>15.2f} | {step_stats_0['Val Attr RMSE']:>13.2f} | "
        f"{0:>8d} | {elapsed_time_str}"
    )
    print(log_line_0)

    for epoch in range(config.N_EPOCHS):
        if early_stop_flag: break
        model.train()

        for i, batch in enumerate(train_loader):
            if early_stop_flag: break

            for key, value in batch.items():
                if isinstance(value, torch.Tensor): batch[key] = value.to(config.DEVICE, non_blocking=True)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items(): batch[key][sub_key] = sub_value.to(config.DEVICE, non_blocking=True)

            targets = batch['target']
            weights = batch['sample_weight']
            preds = model(batch)
            q_pred = preds['q_pred']

            target_log_price = torch.log1p(targets)
            loss_backbone = (weights * (q_pred - target_log_price)**2).mean()

            p_pred = torch.exp(q_pred).detach()
            p_sum = preds['p_bias'] + preds['p_loc'] + preds['p_size'] + preds['p_qual'] + preds['p_amenities'] + preds['p_season']
            loss_attribution = (weights * (p_sum - p_pred)**2).mean()

            total_loss = loss_backbone + loss_attribution

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step_count += 1

            if (global_step_count % config.LOG_EVERY_N_STEPS == 0):
                val_losses = evaluate_model(model, val_loader, config.DEVICE, config)
                current_val_loss = val_losses['backbone_loss']

                if current_val_loss < best_val_loss - config.EARLY_STOPPING_MIN_DELTA:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    early_stop_flag = True

                elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

                step_stats = {
                    'Steps': global_step_count,
                    'Epoch': epoch + (i + 1) / len(train_loader),
                    'Train RW-MSLE': np.sqrt(loss_backbone.item()), # Current batch train loss
                    'Val RW-MSLE': np.sqrt(current_val_loss),
                    'Train Attr RMSE': np.sqrt(loss_attribution.item()), # Current batch train loss
                    'Val Attr RMSE': np.sqrt(val_losses['attribution_loss']),
                    'Patience': patience_counter,
                    'Elapsed Time': elapsed_time_str
                }
                history.append(step_stats)

                log_line = (
                    f"{global_step_count:>5d} | {step_stats['Epoch']:>5.2f} | "
                    f"{step_stats['Train RW-MSLE']:>13.4f} | {step_stats['Val RW-MSLE']:>11.4f} | "
                    f"{step_stats['Train Attr RMSE']:>15.2f} | {step_stats['Val Attr RMSE']:>13.2f} | "
                    f"{patience_counter:>8d} | {elapsed_time_str}"
                )
                print(log_line)

                if early_stop_flag:
                    print(f"\n--- Early Stopping Triggered at Step {global_step_count} ---")

    print("\n--- Training Complete ---")
    if best_model_state is not None:
        print(f"Loading best model state (Val RW-MSLE: {np.sqrt(best_val_loss):.4f})")
        model.load_state_dict(best_model_state)

    return model, pd.DataFrame(history)
```

### **8. Main Execution Function**

This single cell runs the entire pipeline from start to finish using the settings defined in the `Config` class.


```python
def save_artifacts(artifacts: dict, config: Config):
    """Saves the essential training artifacts to a single file."""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{config.CITY}_artifacts_{timestamp}.pt"

    # Define paths for local runtime and Google Drive
    runtime_path = f"./{filename}"
    drive_path = os.path.join(config.DRIVE_SAVE_PATH, filename)

    # Ensure Google Drive directory exists
    os.makedirs(config.DRIVE_SAVE_PATH, exist_ok=True)

    print(f"\nSaving artifacts to {runtime_path} and {drive_path}...")
    torch.save(artifacts, runtime_path)
    torch.save(artifacts, drive_path)
    print("Artifacts saved successfully.")

def main(config: Config):
    """Runs the end-to-end training pipeline for the Hybrid Explainable Model."""
    # 1. Load and split data
    train_df, val_df = load_and_split_data(config)

    # 2. Process features
    processor = FeatureProcessor()
    processor.fit(train_df)
    train_features_cpu = preprocess_and_tensorize_CPU(processor, train_df)
    val_features_cpu = preprocess_and_tensorize_CPU(processor, val_df)

    # 3. Create DataLoaders
    train_loader, val_loader = create_dataloaders(train_features_cpu, val_features_cpu, config)

    # 4. Initialize the new HybridExplainableModel and optimizer
    model = HybridExplainableModel(processor, device=config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 5. Run training
    trained_model, training_history = train_model(model, train_loader, val_loader, optimizer, config)

    # 6. Collate all artifacts into a single dictionary
    artifacts = {
        "config": config,
        "processor": processor,
        "model_state_dict": trained_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": training_history
    }

    # 7. Save artifacts and display final results
    save_artifacts(artifacts, config)

    print("\n--- Final Training History ---")
    if not training_history.empty:
        display_format = {
            'Epoch': '{:.2f}'.format,
            'Train RW-MSLE': '{:.4f}'.format,
            'Val RW-MSLE': '{:.4f}'.format,
            'Train Attr RMSE': '{:.2f}'.format,
            'Val Attr RMSE': '{:.2f}'.format,
        }
        display(training_history.set_index('Steps').style.format(display_format))

    return artifacts
```

#### **9. Final execution cell**

Requires two steps-- First, instantiate a Config object (`config`, say), changing any attributes from the default as needed. Next, simply run `main(config)`


```python
# Instantiate the configuration
config = Config()
set_seed(config.SEED)

print(f"Configuration loaded:")
print(f"Device: {config.DEVICE}")
print(f"City: {config.CITY}")
print(f"Seed: {config.SEED}")
print(f"Batch Size: {config.BATCH_SIZE}")
print(f"Learning Rate: {config.LEARNING_RATE}")
print(f"Number of Epochs: {config.N_EPOCHS}")
print(f"Logging Interval (steps): {config.LOG_EVERY_N_STEPS}")
print("="*50)
```

    All random seeds set to 42.
    Configuration loaded:
    Device: cuda
    City: nyc
    Seed: 42
    Batch Size: 1024
    Learning Rate: 0.001
    Number of Epochs: 30
    Logging Interval (steps): 10
    ==================================================



```python
# Run the end-to-end training pipeline
training_artifacts = main(config)
```

    Loading dataset from: ./nyc_final_modeling_dataset.parquet
    Removed price outliers. New size: 81,643 records.
    Removed small strata. New size: 79,485 records.
    Split complete. Training: 63,588, Validation: 15,897
    
    --- Sample Record from Training Data ---
                                                                                 0
    listing_id                                                  779010937952266773
    year_month                                                             2024-11
    target_price                                                              90.0
    estimated_occupancy_rate                                              0.066667
    latitude                                                              40.63478
    longitude                                                             -73.9501
    neighbourhood_cleansed                                                Flatbush
    property_type                                               Entire rental unit
    room_type                                                      Entire home/apt
    accommodates                                                                 2
    bedrooms                                                                   1.0
    beds                                                                       1.0
    bathrooms_numeric                                                          1.0
    bathrooms_type                                                         private
    amenities                    ["Fire extinguisher", "Smoke alarm", "Gas stov...
    review_scores_rating                                                       4.8
    review_scores_cleanliness                                                  4.7
    review_scores_checkin                                                      4.8
    review_scores_communication                                                4.9
    review_scores_location                                                     4.6
    review_scores_value                                                        4.7
    number_of_reviews_ltm                                                        3
    host_is_superhost                                                         True
    host_response_rate                                                         1.0
    host_acceptance_rate                                                      0.89
    host_identity_verified                                                    True
    instant_bookable                                                         False
    month                                                                       11
    price_bin                                                                    0
    DataLoaders created with pin_memory=True and num_workers=2.
    
    --- Starting Model Training (Hybrid Architecture) ---
    Performing Step 0 evaluation...
    Step 0 evaluation complete.
    
    Steps | Epoch | Train RW-MSLE | Val RW-MSLE | Train Attr RMSE | Val Attr RMSE | Patience | Elapsed Time
    -------------------------------------------------------------------------------------------------------
        0 |  0.00 |        2.7764 |      2.6963 |            1.01 |          1.00 |        0 | 00:00:09
       10 |  0.16 |        1.5087 |      1.2651 |            4.59 |          9.08 |        0 | 00:00:21
       20 |  0.32 |        1.4296 |      1.6772 |         4343.85 |       6029.62 |        1 | 00:00:32
       30 |  0.48 |        3.0298 |      3.2462 |       318091.07 |     335721.46 |        2 | 00:00:43
       40 |  0.63 |        3.9874 |      3.9649 |      2105049.38 |    2181718.28 |        3 | 00:00:53
       50 |  0.79 |        4.1736 |      4.2779 |      4118208.72 |    4941870.04 |        4 | 00:01:04
       60 |  0.95 |        4.2446 |      4.4244 |      7200002.58 |    7232145.59 |        5 | 00:01:15
    
    --- Early Stopping Triggered at Step 60 ---
    
    --- Training Complete ---
    Loading best model state (Val RW-MSLE: 1.2651)
    
    Saving artifacts to ./nyc_artifacts_20251009_155058.pt and /content/drive/MyDrive/Colab_Notebooks/Airbnb_Project/nyc_artifacts_20251009_155058.pt...
    Artifacts saved successfully.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def analyze_hybrid_model(model, data_loader, df, device):
    """
    Analyzes the performance and explanations of the hybrid model for a given dataset.
    """
    model.eval()

    # Store all outputs from the model
    q_preds, targets = [], []
    contributions = { "p_loc": [], "p_size": [], "p_qual": [], "p_amenities": [], "p_season": [], "p_bias": [] }

    with torch.no_grad():
        for batch in data_loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor): batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items(): batch[key][sub_key] = sub_value.to(device, non_blocking=True)

            preds = model(batch)

            q_preds.extend(preds['q_pred'].cpu().numpy())
            targets.extend(batch['target'].cpu().numpy())

            for key in contributions.keys():
                contributions[key].extend(preds[key].cpu().numpy())

    # Create a detailed results dataframe
    results_df = df.iloc[:len(q_preds)].copy()
    results_df['target_price'] = targets
    results_df['predicted_log_price'] = q_preds

    # Use np.expm1 to correctly invert np.log1p
    results_df['predicted_price'] = np.expm1(results_df['predicted_log_price'])

    for key, value in contributions.items():
        results_df[key] = value

    results_df['p_sum'] = results_df[list(contributions.keys())].sum(axis=1)

    # --- Perform Analyses ---

    # 1. MAPE by Price Bracket
    results_df['percentage_error'] = ((results_df['predicted_price'] - results_df['target_price']).abs() / results_df['target_price']) * 100
    price_bins = [0, 75, 150, 250, 400, results_df['target_price'].max() + 1]
    results_df['price_bin'] = pd.cut(results_df['target_price'], bins=price_bins, right=False)
    performance_summary = results_df.groupby('price_bin').agg(mape=('percentage_error', 'mean')).reset_index()

    return performance_summary, results_df
```


```python
 --- Main Analysis Execution ---

# 1. Unpack the artifacts from your training run
print("Unpacking artifacts...")
processor = training_artifacts['processor']
config = training_artifacts['config']
model_state_dict = training_artifacts['model_state_dict']

# 2. Recreate the data splits and data loaders
print("Recreating data splits and data loaders...")
train_df, val_df = load_and_split_data(config)
train_features_cpu = preprocess_and_tensorize_CPU(processor, train_df)
val_features_cpu = preprocess_and_tensorize_CPU(processor, val_df)
train_loader, val_loader = create_dataloaders(train_features_cpu, val_features_cpu, config)
print("Data loaders recreated.")

# 3. Instantiate a new model and load the trained (best) weights
print("Loading trained model...")
hybrid_model = HybridExplainableModel(processor, config.DEVICE)
hybrid_model.load_state_dict(model_state_dict)
print("Model loaded.")

# 4. Analyze performance on both TRAINING and VALIDATION sets
print("\n--- Analyzing performance on TRAINING set ---")
train_summary, train_results_df = analyze_hybrid_model(hybrid_model, train_loader, train_df, config.DEVICE)
print("\n--- Analyzing performance on VALIDATION set ---")
val_summary, val_results_df = analyze_hybrid_model(hybrid_model, val_loader, val_df, config.DEVICE)

# --- 5. Print Key Statistics ---

# Sanity Check
train_consistency_error = (train_results_df['p_sum'] - train_results_df['predicted_price']).abs().mean()
val_consistency_error = (val_results_df['p_sum'] - val_results_df['predicted_price']).abs().mean()
print("\n--- Sanity Check: Does Σ(contributions) ≈ predicted price? ---")
print(f"Mean Absolute Difference (Train): ${train_consistency_error:.2f}")
print(f"Mean Absolute Difference (Validation): ${val_consistency_error:.2f}")

# MAPE Summaries
print("\n--- MAPE by Price Bracket (Validation Set) ---")
print(val_summary.to_string(index=False, float_format="%.2f"))

# Average Contributions
print("\n--- Average Dollar Contributions (Validation Set) ---")
val_avg_contrib = val_results_df[['p_bias', 'p_loc', 'p_size', 'p_qual', 'p_amenities', 'p_season']].mean()
print(val_avg_contrib)

# --- 6. Generate Visualizations ---

print("\nGenerating plots...")

# Figure 1: MAPE by Price Bracket
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
fig.suptitle('Model Accuracy Analysis: Hybrid Model', fontsize=20)

sns.barplot(ax=axes[0], data=train_summary, x='price_bin', y='mape', palette='plasma')
axes[0].set_title('Training Set: MAPE by Price Bracket', fontsize=16)
axes[0].set_xlabel('True Price Range ($)', fontsize=12)
axes[0].set_ylabel('Mean Absolute Percentage Error (MAPE %)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

sns.barplot(ax=axes[1], data=val_summary, x='price_bin', y='mape', palette='plasma')
axes[1].set_title('Validation Set: MAPE by Price Bracket', fontsize=16)
axes[1].set_xlabel('True Price Range ($)', fontsize=12)
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Figure 2: Average Dollar Contributions
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
fig.suptitle('Model Explanation Analysis: Average Dollar Contributions', fontsize=20)

train_avg_contrib = train_results_df[['p_bias', 'p_loc', 'p_size', 'p_qual', 'p_amenities', 'p_season']].mean()
train_avg_contrib.plot(kind='bar', ax=axes[0], color=sns.color_palette('viridis', 6))
axes[0].set_title('Training Set', fontsize=16)
axes[0].set_ylabel('Average Contribution ($)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

val_avg_contrib.plot(kind='bar', ax=axes[1], color=sns.color_palette('viridis', 6))
axes[1].set_title('Validation Set', fontsize=16)
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```
