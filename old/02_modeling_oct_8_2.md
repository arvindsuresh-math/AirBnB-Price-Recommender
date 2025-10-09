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
from tqdm import tqdm
from IPython.display import display, clear_output
import json
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
    N_EPOCHS: int = 20

    # --- Logging ---
    LOG_EVERY_N_STEPS: int = 10 # Log and validate every N training steps
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
    display(train_df.head(1).T)

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
            "target_log_price": np.log1p(df["target_price"].values),
            "sample_weight": df["estimated_occupancy_rate"].values
        }
```

### **4. AirbnbDataset Class**

The PyTorch `Dataset` class, which defines how to retrieve a single item from our processed feature dictionary.


```python
# class AirbnbPriceDataset(Dataset):
#     def __init__(self, features: dict):
#         self.features = features
#         self.n_samples = len(features['sample_weight'])

#     def __len__(self):
#         return self.n_samples

#     def __getitem__(self, index: int) -> dict:
#         item = {}
#         # Location
#         item['loc_geo_position'] = torch.tensor(self.features['location']['geo_position'][index], dtype=torch.float32)
#         item['loc_neighbourhood'] = torch.tensor(self.features['location']['neighbourhood'][index], dtype=torch.long)

#         # Size & Capacity
#         for k, v in self.features['size_capacity'].items():
#             dtype = torch.float32 if k == 'accommodates' else torch.long
#             item[f'size_{k}'] = torch.tensor(v[index], dtype=dtype)

#         # Quality
#         for k, v in self.features['quality'].items():
#             item[f'qual_{k}'] = torch.tensor(v[index], dtype=torch.float32)

#         # Amenities & Seasonality
#         item['amenities_text'] = self.features['amenities']['text'][index]
#         item['season_cyclical'] = torch.tensor(self.features['seasonality']['cyclical'][index], dtype=torch.float32)

#         # Target & Weight
#         item['target'] = torch.tensor(self.features['target_log_price'][index], dtype=torch.float32)
#         item['sample_weight'] = torch.tensor(self.features['sample_weight'][index], dtype=torch.float32)

#         return item

# (This cell replaces your existing AirbnbPriceDataset class definition)

# (This cell replaces the AirbnbPriceDataset class)

class AirbnbPriceDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features
        self.n_samples = len(features['sample_weight'])

    def __len__(self): return self.n_samples

    def __getitem__(self, index: int) -> dict:
        # Reverting to torch.tensor() calls as input is now CPU numpy/tensors
        item = {}
        item['loc_geo_position'] = torch.tensor(self.features['location']['geo_position'][index], dtype=torch.float32)
        item['loc_neighbourhood'] = torch.tensor(self.features['location']['neighbourhood'][index], dtype=torch.long)
        for k, v in self.features['size_capacity'].items():
            dtype = torch.float32 if k == 'accommodates' else torch.long
            item[f'size_{k}'] = torch.tensor(v[index], dtype=dtype)
        for k, v in self.features['quality'].items():
            item[f'qual_{k}'] = torch.tensor(v[index], dtype=torch.float32)
        item['amenities_text'] = self.features['amenities']['text'][index]
        item['season_cyclical'] = torch.tensor(self.features['seasonality']['cyclical'][index], dtype=torch.float32)
        item['target'] = torch.tensor(self.features['target_log_price'][index], dtype=torch.float32)
        item['sample_weight'] = torch.tensor(self.features['sample_weight'][index], dtype=torch.float32)
        return item
```

### **5. Dataloader Creation**

A function to create the `DataLoader` instances, including the custom collate function for batch tokenization.


```python
# (This cell replaces the 'preprocess_and_tensorize' and 'create_dataloaders' functions)

def preprocess_and_tensorize_CPU(processor, df):
    """
    Applies the feature processor and converts data to CPU tensors.
    """
    features_cpu = processor.transform(df)

    # Create a new dictionary for CPU tensors
    features_tensor = {}

    for key, value in features_cpu.items():
        if key == 'amenities':
            features_tensor[key] = value # Keep raw text
        elif isinstance(value, dict):
            features_tensor[key] = {}
            for sub_key, sub_val in value.items():
                dtype = torch.long if sub_key not in ['accommodates', 'geo_position'] else torch.float32
                features_tensor[key][sub_key] = torch.from_numpy(sub_val).to(dtype=dtype)
        else:
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
class AdditiveAxisModel(nn.Module):
    def __init__(self, processor: FeatureProcessor, device: str):
        super().__init__()
        self.vocabs, self.device = processor.vocabs, device
        self.embed_neighbourhood = nn.Embedding(len(self.vocabs['neighbourhood_cleansed']), 16)
        self.embed_property_type = nn.Embedding(len(self.vocabs['property_type']), 8)
        self.embed_room_type = nn.Embedding(len(self.vocabs['room_type']), 4)
        self.embed_bathrooms_type = nn.Embedding(len(self.vocabs['bathrooms_type']), 2)
        self.embed_bedrooms = nn.Embedding(len(self.vocabs['bedrooms']), 4)
        self.embed_beds = nn.Embedding(len(self.vocabs['beds']), 4)
        self.embed_bathrooms_numeric = nn.Embedding(len(self.vocabs['bathrooms_numeric']), 4)
        self.amenities_transformer = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        for param in self.amenities_transformer.parameters(): param.requires_grad = False
        self.loc_subnet = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))
        self.size_subnet = nn.Sequential(nn.Linear(27, 32), nn.ReLU(), nn.Linear(32, 1))
        self.qual_subnet = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 1))
        self.amenities_subnet = nn.Linear(384, 1)
        self.season_subnet = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.global_bias = nn.Parameter(torch.randn(1))
        self.to(self.device)

    def forward(self, batch: dict) -> torch.Tensor:
        # Location
        loc_geo = batch['loc_geo_position']
        loc_hood_embed = self.embed_neighbourhood(batch['loc_neighbourhood'])
        loc_input = torch.cat([loc_geo, loc_hood_embed], dim=1)

        # Size
        size_embeds = [
            self.embed_property_type(batch['size_property_type']),
            self.embed_room_type(batch['size_room_type']),
            self.embed_bathrooms_type(batch['size_bathrooms_type']),
            self.embed_beds(batch['size_beds']),
            self.embed_bedrooms(batch['size_bedrooms']),
            self.embed_bathrooms_numeric(batch['size_bathrooms_numeric']),
            batch['size_accommodates'].unsqueeze(1)
            ]
        size_input = torch.cat(size_embeds, dim=1)

        # Quality
        qual_inputs = [
            batch['qual_review_scores_rating'].unsqueeze(1),
            batch['qual_review_scores_cleanliness'].unsqueeze(1),
            batch['qual_review_scores_checkin'].unsqueeze(1),
            batch['qual_review_scores_communication'].unsqueeze(1),
            batch['qual_review_scores_location'].unsqueeze(1),
            batch['qual_review_scores_value'].unsqueeze(1),
            batch['qual_host_response_rate'].unsqueeze(1),
            batch['qual_host_acceptance_rate'].unsqueeze(1),
            batch['qual_number_of_reviews_ltm'].unsqueeze(1),
            batch['qual_host_is_superhost'].unsqueeze(1),
            batch['qual_host_identity_verified'].unsqueeze(1),
            batch['qual_instant_bookable'].unsqueeze(1)
            ]
        qual_input = torch.cat(qual_inputs, dim=1)

        # Amenities
        amenities_tokens = batch['amenities_tokens']
        amenities_embed = self.amenities_transformer(amenities_tokens)['sentence_embedding']

        # Get price contributions
        p_loc = self.loc_subnet(loc_input)
        p_size = self.size_subnet(size_input)
        p_qual = self.qual_subnet(qual_input)
        p_amenities = self.amenities_subnet(amenities_embed)
        p_season = self.season_subnet(batch['season_cyclical'])

        return (
            self.global_bias
            + p_loc
            + p_size
            + p_qual
            + p_amenities
            + p_season
          ).squeeze(-1)
```

### **7. Training Function**

This function orchestrates the training and validation loops for a given number of epochs.



```python
# def train_model(
#     model,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     optimizer: optim.Optimizer,
#     config: Config
#     ):
#     """Trains the model with step-based logging and validation."""
#     print("\n--- Starting Model Training ---")

#     history = []
#     total_steps = len(train_loader) * config.N_EPOCHS
#     pbar = tqdm(total=total_steps, desc="Overall Training Progress")

#     global_step_count = 0

#     for epoch in range(config.N_EPOCHS):
#         model.train()

#         for i, batch in enumerate(train_loader):
#             # --- Training Step ---
#             predictions = model(batch)
#             targets = batch['target']
#             weights = batch['sample_weight']
#             loss = (weights * (predictions - targets)**2).mean()

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             global_step_count += 1
#             pbar.update(1)
#             pbar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})

#             # --- Periodic Validation and Logging ---
#             # Check if it's a logging step or the very last step of training
#             if (global_step_count % config.LOG_EVERY_N_STEPS == 0) or (global_step_count == total_steps):
#                 model.eval()
#                 val_loss = 0.0
#                 with torch.no_grad():
#                     for val_batch in val_loader:
#                         val_preds = model(val_batch)
#                         val_targets = val_batch['target'].to(config.DEVICE)
#                         val_weights = val_batch['sample_weight'].to(config.DEVICE)
#                         val_batch_loss = (val_weights * (val_preds - val_targets)**2).mean()
#                         val_loss += val_batch_loss.item()

#                 avg_val_loss = val_loss / len(val_loader)

#                 # --- Update UI ---
#                 epoch_float = epoch + (i + 1) / len(train_loader)
#                 step_stats = {
#                     'Steps': global_step_count,
#                     'Epoch': epoch_float,
#                     'Train RW-MSLE': np.sqrt(loss.item()), # Current train batch loss
#                     'Val RW-MSLE': np.sqrt(avg_val_loss)
#                 }
#                 history.append(step_stats)

#                 clear_output(wait=True)
#                 history_df = pd.DataFrame(history).set_index('Steps')
#                 display(history_df.style.format({'Epoch': '{:.2f}', 'Train RW-MSLE': '{:.4f}', 'Val RW-MSLE': '{:.4f}'}))

#                 model.train() # Switch back to training mode

#     pbar.close()
#     print("\n--- Training Complete ---")
#     return model, pd.DataFrame(history)

def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Config
    ):
    """Trains the model with step-based logging and validation."""
    print("\n--- Starting Model Training ---")

    history = []
    total_steps = len(train_loader) * config.N_EPOCHS
    pbar = tqdm(total=total_steps, desc="Overall Training Progress")

    global_step_count = 0

    for epoch in range(config.N_EPOCHS):
        model.train()
        for i, batch in enumerate(train_loader):
            # --- Re-introduce the efficient data transfer step ---
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(config.DEVICE, non_blocking=True)
                elif isinstance(value, dict): # For amenities_tokens
                    for sub_key, sub_value in value.items():
                        batch[key][sub_key] = sub_value.to(config.DEVICE, non_blocking=True)

            targets = batch['target']
            weights = batch['sample_weight']

            # --- Training Step ---
            predictions = model(batch)
            loss = (weights * (predictions - targets)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step_count += 1
            pbar.update(1)
            pbar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})

            # --- Periodic Validation and Logging ---
            # Check if it's a logging step or the very last step of training
            if (global_step_count % config.LOG_EVERY_N_STEPS == 0) or (global_step_count == total_steps):
                model.eval()
                with torch.no_grad():
                    for val_batch in val_loader:
                        for key, value in val_batch.items():
                            if isinstance(value, torch.Tensor):
                              val_batch[key] = value.to(config.DEVICE, non_blocking=True)
                            elif isinstance(value, dict):
                              for sub_key, sub_value in value.items():
                                  val_batch[key][sub_key] = sub_value.to(config.DEVICE, non_blocking=True)
                        val_targets = val_batch['target']
                        val_weights = val_batch['sample_weight']

                        val_preds = model(val_batch)
                        val_batch_loss = (val_weights * (val_preds - val_targets)**2).mean()
                        val_loss += val_batch_loss.item()

                avg_val_loss = val_loss / len(val_loader)

                # --- Update UI ---
                epoch_float = epoch + (i + 1) / len(train_loader)
                step_stats = {
                    'Steps': global_step_count,
                    'Epoch': epoch_float,
                    'Train RW-MSLE': np.sqrt(loss.item()), # Current train batch loss
                    'Val RW-MSLE': np.sqrt(avg_val_loss)
                }
                history.append(step_stats)

                clear_output(wait=True)
                history_df = pd.DataFrame(history).set_index('Steps')
                display(history_df.style.format({'Epoch': '{:.2f}', 'Train RW-MSLE': '{:.4f}', 'Val RW-MSLE': '{:.4f}'}))

                model.train() # Switch back to training mode

    pbar.close()
    print("\n--- Training Complete ---")
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
    """Runs the end-to-end training pipeline."""
    # 1. Load and split data
    train_df, val_df = load_and_split_data(config)

    # 2. Process features and move to GPU
    processor = FeatureProcessor()
    processor.fit(train_df)
    train_features_cpu = preprocess_and_tensorize_CPU(processor, train_df)
    val_features_cpu = preprocess_and_tensorize_CPU(processor, val_df)

    # 3. Create DataLoaders (pass CPU features)
    train_loader, val_loader = create_dataloaders(train_features_cpu, val_features_cpu, config)

    # 4. Initialize model and optimizer
    model = AdditiveAxisModel(processor, device=config.DEVICE)
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

    print("\n--- Final Training Results ---")
    display(training_history.style.format('{:.4f}').set_index('Steps'))

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
print(f"Logging Interval: {config.LOG_EVERY_N_STEPS}")
print("="*50)

# Run the end-to-end training pipeline
training_artifacts = main(config)
```

    All random seeds set to 42.
    Configuration loaded:
    Device: cuda
    City: nyc
    Seed: 42
    Batch Size: 1024
    Learning Rate: 0.001
    Number of Epochs: 20
    Logging Interval: 10
    ==================================================
    Loading dataset from: ./nyc_final_modeling_dataset.parquet
    Removed price outliers. New size: 81,643 records.
    Removed small strata. New size: 79,485 records.
    Split complete. Training: 63,588, Validation: 15,897
    
    --- Sample Record from Training Data ---




  <div id="df-4b56441e-8e26-4330-9c19-1b498a35f4e0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>listing_id</th>
      <td>779010937952266773</td>
    </tr>
    <tr>
      <th>year_month</th>
      <td>2024-11</td>
    </tr>
    <tr>
      <th>target_price</th>
      <td>90.0</td>
    </tr>
    <tr>
      <th>estimated_occupancy_rate</th>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>40.63478</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-73.9501</td>
    </tr>
    <tr>
      <th>neighbourhood_cleansed</th>
      <td>Flatbush</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>Entire rental unit</td>
    </tr>
    <tr>
      <th>room_type</th>
      <td>Entire home/apt</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>2</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bathrooms_numeric</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bathrooms_type</th>
      <td>private</td>
    </tr>
    <tr>
      <th>amenities</th>
      <td>["Fire extinguisher", "Smoke alarm", "Gas stov...</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>4.8</td>
    </tr>
    <tr>
      <th>review_scores_cleanliness</th>
      <td>4.7</td>
    </tr>
    <tr>
      <th>review_scores_checkin</th>
      <td>4.8</td>
    </tr>
    <tr>
      <th>review_scores_communication</th>
      <td>4.9</td>
    </tr>
    <tr>
      <th>review_scores_location</th>
      <td>4.6</td>
    </tr>
    <tr>
      <th>review_scores_value</th>
      <td>4.7</td>
    </tr>
    <tr>
      <th>number_of_reviews_ltm</th>
      <td>3</td>
    </tr>
    <tr>
      <th>host_is_superhost</th>
      <td>True</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>0.89</td>
    </tr>
    <tr>
      <th>host_identity_verified</th>
      <td>True</td>
    </tr>
    <tr>
      <th>instant_bookable</th>
      <td>False</td>
    </tr>
    <tr>
      <th>month</th>
      <td>11</td>
    </tr>
    <tr>
      <th>price_bin</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4b56441e-8e26-4330-9c19-1b498a35f4e0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4b56441e-8e26-4330-9c19-1b498a35f4e0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4b56441e-8e26-4330-9c19-1b498a35f4e0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-252fb58a-5c8f-47ba-862c-f09e0657a110">
      <button class="colab-df-quickchart" onclick="quickchart('df-252fb58a-5c8f-47ba-862c-f09e0657a110')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-252fb58a-5c8f-47ba-862c-f09e0657a110 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    DataLoaders created with pin_memory=True and num_workers=2.
    
    --- Starting Model Training ---


    
    
    
    
    Overall Training Progress:   0%|          | 0/1260 [00:00<?, ?it/s][A[A[A[A/tmp/ipython-input-1781483063.py:48: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['loc_geo_position'] = torch.tensor(self.features['location']['geo_position'][index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:49: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['loc_neighbourhood'] = torch.tensor(self.features['location']['neighbourhood'][index], dtype=torch.long)
    /tmp/ipython-input-1781483063.py:48: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['loc_geo_position'] = torch.tensor(self.features['location']['geo_position'][index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:52: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item[f'size_{k}'] = torch.tensor(v[index], dtype=dtype)
    /tmp/ipython-input-1781483063.py:49: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['loc_neighbourhood'] = torch.tensor(self.features['location']['neighbourhood'][index], dtype=torch.long)
    /tmp/ipython-input-1781483063.py:54: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item[f'qual_{k}'] = torch.tensor(v[index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:52: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item[f'size_{k}'] = torch.tensor(v[index], dtype=dtype)
    /tmp/ipython-input-1781483063.py:54: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item[f'qual_{k}'] = torch.tensor(v[index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:56: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['season_cyclical'] = torch.tensor(self.features['seasonality']['cyclical'][index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:57: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['target'] = torch.tensor(self.features['target_log_price'][index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:56: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['season_cyclical'] = torch.tensor(self.features['seasonality']['cyclical'][index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:58: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['sample_weight'] = torch.tensor(self.features['sample_weight'][index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:57: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['target'] = torch.tensor(self.features['target_log_price'][index], dtype=torch.float32)
    /tmp/ipython-input-1781483063.py:58: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
      item['sample_weight'] = torch.tensor(self.features['sample_weight'][index], dtype=torch.float32)



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /tmp/ipython-input-1293461057.py in <cell line: 0>()
         14 
         15 # Run the end-to-end training pipeline
    ---> 16 training_artifacts = main(config)
    

    /tmp/ipython-input-152939801.py in main(config)
         35 
         36     # 5. Run training
    ---> 37     trained_model, training_history = train_model(model, train_loader, val_loader, optimizer, config)
         38 
         39     # 6. Collate all artifacts into a single dictionary


    /tmp/ipython-input-1581783203.py in train_model(model, train_loader, val_loader, optimizer, config)
        100 
        101             # --- Training Step ---
    --> 102             predictions = model(batch)
        103             loss = (weights * (predictions - targets)**2).mean()
        104 


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /tmp/ipython-input-3806591564.py in forward(self, batch)
         57         # Amenities
         58         amenities_tokens = batch['amenities_tokens']
    ---> 59         amenities_embed = self.amenities_transformer(amenities_tokens)['sentence_embedding']
         60 
         61         # Get price contributions


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/sentence_transformers/SentenceTransformer.py in forward(self, input, **kwargs)
       1173                     if key in module_kwarg_keys or (hasattr(module, "forward_kwargs") and key in module.forward_kwargs)
       1174                 }
    -> 1175             input = module(input, **module_kwargs)
       1176         return input
       1177 


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/sentence_transformers/models/Transformer.py in forward(self, features, **kwargs)
        259         trans_features = {key: value for key, value in features.items() if key in self.model_forward_params}
        260 
    --> 261         outputs = self.auto_model(**trans_features, **kwargs, return_dict=True)
        262         token_embeddings = outputs[0]
        263         features["token_embeddings"] = token_embeddings


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/transformers/models/bert/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict, cache_position)
        933                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        934 
    --> 935         embedding_output = self.embeddings(
        936             input_ids=input_ids,
        937             position_ids=position_ids,


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/transformers/models/bert/modeling_bert.py in forward(self, input_ids, token_type_ids, position_ids, inputs_embeds, past_key_values_length)
        178 
        179         if inputs_embeds is None:
    --> 180             inputs_embeds = self.word_embeddings(input_ids)
        181         token_type_embeddings = self.token_type_embeddings(token_type_ids)
        182 


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/sparse.py in forward(self, input)
        190 
        191     def forward(self, input: Tensor) -> Tensor:
    --> 192         return F.embedding(
        193             input,
        194             self.weight,


    /usr/local/lib/python3.12/dist-packages/torch/nn/functional.py in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
       2544         # remove once script supports set_grad_enabled
       2545         _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    -> 2546     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
       2547 
       2548 


    RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__index_select)



```python
# (This cell replaces your existing Dataloader Creation cell)

# --- DEBUG: Function to print tensor devices in a nested dictionary ---
def print_tensor_devices(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"{prefix}{k}: {v.device}")
        elif isinstance(v, dict):
            print_tensor_devices(v, prefix=f"{k}.")

def preprocess_and_tensorize(processor, df, device):
    """Applies the feature processor and moves all resulting tensors to the specified device."""
    features_cpu = processor.transform(df)
    features_gpu = {}
    for key, value in features_cpu.items():
        if key == 'amenities':
            features_gpu[key] = value
        elif isinstance(value, dict):
            features_gpu[key] = {}
            for sub_key, sub_val in value.items():
                dtype = torch.long if sub_key != 'accommodates' and 'geo_position' not in sub_key else torch.float32
                features_gpu[key][sub_key] = torch.from_numpy(sub_val).to(device, dtype=dtype)
        else:
            features_gpu[key] = torch.from_numpy(value).to(device, dtype=torch.float32)

    print("\n--- DEBUG: After preprocess_and_tensorize ---")
    print("Verifying devices of pre-loaded tensors...")
    print_tensor_devices(features_gpu)
    return features_gpu

def create_dataloaders_DEBUG(train_features_gpu, val_features_gpu, config: Config):
    """Creates DataLoaders with a debug-enabled collate function."""

    tokenizer_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=config.DEVICE)

    def custom_collate_fn_DEBUG(batch: list) -> dict:
        amenities_texts = [item.pop('amenities_text') for item in batch]
        collated_batch = {key: torch.stack([d[key] for d in batch]) for key in batch[0].keys()}

        # Tokenizer creates new tensors. Let's see what device they are on.
        tokenized = tokenizer_model.tokenizer(
            amenities_texts, padding=True, truncation=True, return_tensors='pt', max_length=128
        )
        collated_batch['amenities_tokens'] = tokenized

        # --- DEBUG PRINT ---
        if not hasattr(custom_collate_fn_DEBUG, '_printed'):
            print("\n--- DEBUG: Inside custom_collate_fn ---")
            print("Verifying devices of tensors in the FIRST collated batch:")
            print_tensor_devices(collated_batch)
            custom_collate_fn_DEBUG._printed = True # Print only once

        return collated_batch

    train_dataset = AirbnbPriceDataset(train_features_gpu)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_DEBUG) # Shuffle=False for deterministic debug

    print(f"\nDataLoaders created. Debug prints are enabled.")
    return train_loader
```


```python
# (This cell replaces your existing Model Architecture cell)

class AdditiveAxisModel_DEBUG(nn.Module):
    # __init__ is identical to your original AdditiveAxisModel
    def __init__(self, processor: FeatureProcessor, device: str):
        super().__init__()
        self.vocabs, self.device = processor.vocabs, device
        self.embed_neighbourhood = nn.Embedding(len(self.vocabs['neighbourhood_cleansed']), 16)
        self.embed_property_type = nn.Embedding(len(self.vocabs['property_type']), 8)
        self.embed_room_type = nn.Embedding(len(self.vocabs['room_type']), 4)
        self.embed_bathrooms_type = nn.Embedding(len(self.vocabs['bathrooms_type']), 2)
        self.embed_bedrooms = nn.Embedding(len(self.vocabs['bedrooms']), 4)
        self.embed_beds = nn.Embedding(len(self.vocabs['beds']), 4)
        self.embed_bathrooms_numeric = nn.Embedding(len(self.vocabs['bathrooms_numeric']), 4)
        self.amenities_transformer = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        for param in self.amenities_transformer.parameters(): param.requires_grad = False
        self.loc_subnet = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))
        self.size_subnet = nn.Sequential(nn.Linear(27, 32), nn.ReLU(), nn.Linear(32, 1))
        self.qual_subnet = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 1))
        self.amenities_subnet = nn.Linear(384, 1)
        self.season_subnet = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.global_bias = nn.Parameter(torch.randn(1))
        self.to(self.device)

    def forward(self, batch: dict) -> torch.Tensor:
        # --- DEBUG PRINT ---
        print("\n--- DEBUG: Inside model.forward() ---")
        print(f"Model's own device: {self.device}")
        print("Devices of tensors received in the batch:")
        print_tensor_devices(batch)

        # --- The rest of the forward pass ---
        loc_geo = batch['loc_geo_position']
        # ... (rest of the forward pass is identical)
        loc_hood_embed = self.embed_neighbourhood(batch['loc_neighbourhood'])
        loc_input = torch.cat([loc_geo, loc_hood_embed], dim=1)
        size_embeds = [self.embed_property_type(batch['size_property_type']), self.embed_room_type(batch['size_room_type']), self.embed_bathrooms_type(batch['size_bathrooms_type']), self.embed_beds(batch['size_beds']), self.embed_bedrooms(batch['size_bedrooms']), self.embed_bathrooms_numeric(batch['size_bathrooms_numeric']), batch['size_accommodates'].unsqueeze(1)]
        size_input = torch.cat(size_embeds, dim=1)
        qual_inputs = [batch[f'qual_{col}'].unsqueeze(1) for col in ["review_scores_rating", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value", "host_response_rate", "host_acceptance_rate", "number_of_reviews_ltm", "host_is_superhost", "host_identity_verified", "instant_bookable"]]
        qual_input = torch.cat(qual_inputs, dim=1)
        amenities_embed = self.amenities_transformer(batch['amenities_tokens'])['sentence_embedding']
        p_loc = self.loc_subnet(loc_input)
        p_size = self.size_subnet(size_input)
        p_qual = self.qual_subnet(qual_input)
        p_amenities = self.amenities_subnet(amenities_embed)
        p_season = self.season_subnet(batch['season_cyclical'])
        return (self.global_bias + p_loc + p_size + p_qual + p_amenities + p_season).squeeze(-1)
```


```python
# (This replaces your existing main execution cell)

def main_DEBUG(config: Config):
    """Runs a single-batch debug pipeline to trace tensor devices."""
    # 1. Load data
    train_df, _ = load_and_split_data(config)

    # 2. Process features and move to GPU
    processor = FeatureProcessor()
    processor.fit(train_df)
    train_features_gpu = preprocess_and_tensorize(processor, train_df, config.DEVICE)

    # 3. Create DEBUG DataLoader
    train_loader_debug = create_dataloaders_DEBUG(train_features_gpu, {}, config)

    # 4. Initialize DEBUG model
    model_debug = AdditiveAxisModel_DEBUG(processor, device=config.DEVICE)
    model_debug.eval()

    # 5. Get a single batch
    print("\n--- Getting a single batch from the DataLoader ---")
    first_batch = next(iter(train_loader_debug))

    # 6. Attempt a single forward pass and catch the error
    print("\n--- Attempting a single forward pass ---")
    with torch.no_grad():
        output = model_debug(first_batch)
    print("\nSUCCESS: Forward pass completed without error.")
    print(f"Output tensor device: {output.device}")

# Run the debug pipeline
main_DEBUG(config)
```

    Loading dataset from: ./nyc_final_modeling_dataset.parquet
    Removed price outliers. New size: 81,643 records.
    Removed small strata. New size: 79,485 records.
    Split complete. Training: 63,588, Validation: 15,897
    
    --- Sample Record from Training Data ---




  <div id="df-29c39db1-d03a-4288-9e33-dc3cb2b3b47a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>listing_id</th>
      <td>779010937952266773</td>
    </tr>
    <tr>
      <th>year_month</th>
      <td>2024-11</td>
    </tr>
    <tr>
      <th>target_price</th>
      <td>90.0</td>
    </tr>
    <tr>
      <th>estimated_occupancy_rate</th>
      <td>0.066667</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>40.63478</td>
    </tr>
    <tr>
      <th>longitude</th>
      <td>-73.9501</td>
    </tr>
    <tr>
      <th>neighbourhood_cleansed</th>
      <td>Flatbush</td>
    </tr>
    <tr>
      <th>property_type</th>
      <td>Entire rental unit</td>
    </tr>
    <tr>
      <th>room_type</th>
      <td>Entire home/apt</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>2</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>beds</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bathrooms_numeric</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>bathrooms_type</th>
      <td>private</td>
    </tr>
    <tr>
      <th>amenities</th>
      <td>["Fire extinguisher", "Smoke alarm", "Gas stov...</td>
    </tr>
    <tr>
      <th>review_scores_rating</th>
      <td>4.8</td>
    </tr>
    <tr>
      <th>review_scores_cleanliness</th>
      <td>4.7</td>
    </tr>
    <tr>
      <th>review_scores_checkin</th>
      <td>4.8</td>
    </tr>
    <tr>
      <th>review_scores_communication</th>
      <td>4.9</td>
    </tr>
    <tr>
      <th>review_scores_location</th>
      <td>4.6</td>
    </tr>
    <tr>
      <th>review_scores_value</th>
      <td>4.7</td>
    </tr>
    <tr>
      <th>number_of_reviews_ltm</th>
      <td>3</td>
    </tr>
    <tr>
      <th>host_is_superhost</th>
      <td>True</td>
    </tr>
    <tr>
      <th>host_response_rate</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>host_acceptance_rate</th>
      <td>0.89</td>
    </tr>
    <tr>
      <th>host_identity_verified</th>
      <td>True</td>
    </tr>
    <tr>
      <th>instant_bookable</th>
      <td>False</td>
    </tr>
    <tr>
      <th>month</th>
      <td>11</td>
    </tr>
    <tr>
      <th>price_bin</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-29c39db1-d03a-4288-9e33-dc3cb2b3b47a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-29c39db1-d03a-4288-9e33-dc3cb2b3b47a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-29c39db1-d03a-4288-9e33-dc3cb2b3b47a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-d2963b50-1dd9-4a59-8f98-51e4f4951670">
      <button class="colab-df-quickchart" onclick="quickchart('df-d2963b50-1dd9-4a59-8f98-51e4f4951670')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-d2963b50-1dd9-4a59-8f98-51e4f4951670 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>



    
    --- DEBUG: After preprocess_and_tensorize ---
    Verifying devices of pre-loaded tensors...
    location.geo_position: cuda:0
    location.neighbourhood: cuda:0
    size_capacity.property_type: cuda:0
    size_capacity.room_type: cuda:0
    size_capacity.bathrooms_type: cuda:0
    size_capacity.bedrooms: cuda:0
    size_capacity.beds: cuda:0
    size_capacity.bathrooms_numeric: cuda:0
    size_capacity.accommodates: cuda:0
    quality.review_scores_rating: cuda:0
    quality.review_scores_cleanliness: cuda:0
    quality.review_scores_checkin: cuda:0
    quality.review_scores_communication: cuda:0
    quality.review_scores_location: cuda:0
    quality.review_scores_value: cuda:0
    quality.host_response_rate: cuda:0
    quality.host_acceptance_rate: cuda:0
    quality.number_of_reviews_ltm: cuda:0
    quality.host_is_superhost: cuda:0
    quality.host_identity_verified: cuda:0
    quality.instant_bookable: cuda:0
    seasonality.cyclical: cuda:0
    target_log_price: cuda:0
    sample_weight: cuda:0
    
    DataLoaders created. Debug prints are enabled.
    
    --- Getting a single batch from the DataLoader ---
    
    --- DEBUG: Inside custom_collate_fn ---
    Verifying devices of tensors in the FIRST collated batch:
    loc_geo_position: cuda:0
    loc_neighbourhood: cuda:0
    size_property_type: cuda:0
    size_room_type: cuda:0
    size_bathrooms_type: cuda:0
    size_bedrooms: cuda:0
    size_beds: cuda:0
    size_bathrooms_numeric: cuda:0
    size_accommodates: cuda:0
    qual_review_scores_rating: cuda:0
    qual_review_scores_cleanliness: cuda:0
    qual_review_scores_checkin: cuda:0
    qual_review_scores_communication: cuda:0
    qual_review_scores_location: cuda:0
    qual_review_scores_value: cuda:0
    qual_host_response_rate: cuda:0
    qual_host_acceptance_rate: cuda:0
    qual_number_of_reviews_ltm: cuda:0
    qual_host_is_superhost: cuda:0
    qual_host_identity_verified: cuda:0
    qual_instant_bookable: cuda:0
    season_cyclical: cuda:0
    target: cuda:0
    sample_weight: cuda:0
    
    --- Attempting a single forward pass ---
    
    --- DEBUG: Inside model.forward() ---
    Model's own device: cuda
    Devices of tensors received in the batch:
    loc_geo_position: cuda:0
    loc_neighbourhood: cuda:0
    size_property_type: cuda:0
    size_room_type: cuda:0
    size_bathrooms_type: cuda:0
    size_bedrooms: cuda:0
    size_beds: cuda:0
    size_bathrooms_numeric: cuda:0
    size_accommodates: cuda:0
    qual_review_scores_rating: cuda:0
    qual_review_scores_cleanliness: cuda:0
    qual_review_scores_checkin: cuda:0
    qual_review_scores_communication: cuda:0
    qual_review_scores_location: cuda:0
    qual_review_scores_value: cuda:0
    qual_host_response_rate: cuda:0
    qual_host_acceptance_rate: cuda:0
    qual_number_of_reviews_ltm: cuda:0
    qual_host_is_superhost: cuda:0
    qual_host_identity_verified: cuda:0
    qual_instant_bookable: cuda:0
    season_cyclical: cuda:0
    target: cuda:0
    sample_weight: cuda:0



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    /tmp/ipython-input-2126646931.py in <cell line: 0>()
         30 
         31 # Run the debug pipeline
    ---> 32 main_DEBUG(config)
    

    /tmp/ipython-input-2126646931.py in main_DEBUG(config)
         25     print("\n--- Attempting a single forward pass ---")
         26     with torch.no_grad():
    ---> 27         output = model_debug(first_batch)
         28     print("\nSUCCESS: Forward pass completed without error.")
         29     print(f"Output tensor device: {output.device}")


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /tmp/ipython-input-3947796058.py in forward(self, batch)
         39         qual_inputs = [batch[f'qual_{col}'].unsqueeze(1) for col in ["review_scores_rating", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value", "host_response_rate", "host_acceptance_rate", "number_of_reviews_ltm", "host_is_superhost", "host_identity_verified", "instant_bookable"]]
         40         qual_input = torch.cat(qual_inputs, dim=1)
    ---> 41         amenities_embed = self.amenities_transformer(batch['amenities_tokens'])['sentence_embedding']
         42         p_loc = self.loc_subnet(loc_input)
         43         p_size = self.size_subnet(size_input)


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/sentence_transformers/SentenceTransformer.py in forward(self, input, **kwargs)
       1173                     if key in module_kwarg_keys or (hasattr(module, "forward_kwargs") and key in module.forward_kwargs)
       1174                 }
    -> 1175             input = module(input, **module_kwargs)
       1176         return input
       1177 


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/sentence_transformers/models/Transformer.py in forward(self, features, **kwargs)
        259         trans_features = {key: value for key, value in features.items() if key in self.model_forward_params}
        260 
    --> 261         outputs = self.auto_model(**trans_features, **kwargs, return_dict=True)
        262         token_embeddings = outputs[0]
        263         features["token_embeddings"] = token_embeddings


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/transformers/models/bert/modeling_bert.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict, cache_position)
        933                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        934 
    --> 935         embedding_output = self.embeddings(
        936             input_ids=input_ids,
        937             position_ids=position_ids,


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/transformers/models/bert/modeling_bert.py in forward(self, input_ids, token_type_ids, position_ids, inputs_embeds, past_key_values_length)
        178 
        179         if inputs_embeds is None:
    --> 180             inputs_embeds = self.word_embeddings(input_ids)
        181         token_type_embeddings = self.token_type_embeddings(token_type_ids)
        182 


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1771             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1772         else:
    -> 1773             return self._call_impl(*args, **kwargs)
       1774 
       1775     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1782                 or _global_backward_pre_hooks or _global_backward_hooks
       1783                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1784             return forward_call(*args, **kwargs)
       1785 
       1786         result = None


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/sparse.py in forward(self, input)
        190 
        191     def forward(self, input: Tensor) -> Tensor:
    --> 192         return F.embedding(
        193             input,
        194             self.weight,


    /usr/local/lib/python3.12/dist-packages/torch/nn/functional.py in embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
       2544         # remove once script supports set_grad_enabled
       2545         _no_grad_embedding_renorm_(weight, input, max_norm, norm_type)
    -> 2546     return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
       2547 
       2548 


    RuntimeError: Expected all tensors to be on the same device, but got index is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA__index_select)



```python

```
