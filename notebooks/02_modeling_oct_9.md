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
    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
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
    EARLY_STOPPING_MIN_DELTA: float = 0.05 # <-- CHANGED to a sensible value for RW-MSE in dollars

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
            # "target_log_price": np.log1p(df["target_price"].values),
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

        # Amenities - The key fix is here:
        amenities_tokens = batch['amenities_tokens'].to(self.device)
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
def evaluate_model(model, data_loader, device):
    """Runs a full evaluation pass on the provided data_loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            # Move all parts of the batch to the correct device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        batch[key][sub_key] = sub_value.to(device, non_blocking=True)

            targets = batch['target']
            weights = batch['sample_weight']

            predictions = model(batch)
            loss = (weights * (predictions - targets)**2).mean().item()
            total_loss += loss

    return total_loss / len(data_loader)

def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Config
    ):
    """
    Trains the model with a Step 0 baseline and prints a concise, single-line log for each validation step.
    """
    print("\n--- Starting Model Training ---")
    start_time = time.time()

    history = []
    total_steps = len(train_loader) * config.N_EPOCHS
    global_step_count = 0
    patience_counter = 0
    early_stop_flag = False

    # --- Step 0: Initial Evaluation ---
    print("Performing Step 0 evaluation...")
    avg_val_loss_0 = evaluate_model(model, val_loader, config.DEVICE)
    first_train_batch = next(iter(train_loader))
    avg_train_loss_0 = evaluate_model(model, [first_train_batch], config.DEVICE)
    print("Step 0 evaluation complete.\n")

    best_val_loss = avg_val_loss_0 # Note: this is still the raw MSE
    best_model_state = model.state_dict()

    elapsed_seconds = time.time() - start_time
    elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))

    step_stats_0 = {
        'Steps': 0, 'Epoch': 0.00,
        'Train RW-MSE ($)': np.sqrt(avg_train_loss_0),
        'Val RW-MSE ($)': np.sqrt(avg_val_loss_0),
        'Patience': 0,
        'Elapsed Time': elapsed_time_str
    }
    history.append(step_stats_0)

    # --- Print Header and Step 0 Results ---
    header = (
        f"{'Steps':>5} | {'Epoch':>5} | {'Train RW-MSE ($)':>16} | "
        f"{'Val RW-MSE ($)':>14} | {'Patience':>8} | {'Elapsed Time'}"
    )
    print(header)
    print("-" * len(header))

    log_line = (
        f"{step_stats_0['Steps']:>5d} | {step_stats_0['Epoch']:>5.2f} | "
        f"{step_stats_0['Train RW-MSE ($)']:>16.2f} | {step_stats_0['Val RW-MSE ($)']:>14.2f} | "
        f"{step_stats_0['Patience']:>8d} | {step_stats_0['Elapsed Time']}"
    )
    print(log_line)
    # --- End of Step 0 ---

    for epoch in range(config.N_EPOCHS):
        if early_stop_flag: break
        model.train()

        for i, batch in enumerate(train_loader):
            if early_stop_flag: break

            for key, value in batch.items():
                if isinstance(value, torch.Tensor): batch[key] = value.to(config.DEVICE, non_blocking=True)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items(): batch[key][sub_key] = sub_value.to(config.DEVICE, non_blocking=True)

            targets, weights = batch['target'], batch['sample_weight']
            predictions = model(batch)
            loss = (weights * (predictions - targets)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step_count += 1

            if (global_step_count % config.LOG_EVERY_N_STEPS == 0) or (global_step_count == total_steps):
                avg_val_loss = evaluate_model(model, val_loader, config.DEVICE)

                # --- CORRECTED EARLY STOPPING LOGIC ---
                # Compare the square root of the loss (RW-MSE)
                current_rw_mse = np.sqrt(avg_val_loss)
                best_rw_mse = np.sqrt(best_val_loss)

                if current_rw_mse < best_rw_mse - config.EARLY_STOPPING_MIN_DELTA:
                    best_val_loss = avg_val_loss # Still save the raw MSE
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                # --- END OF CORRECTION ---

                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    early_stop_flag = True

                elapsed_seconds = time.time() - start_time
                elapsed_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))

                step_stats = {
                    'Steps': global_step_count, 'Epoch': epoch + (i + 1) / len(train_loader),
                    'Train RW-MSE ($)': np.sqrt(loss.item()),
                    'Val RW-MSE ($)': np.sqrt(avg_val_loss),
                    'Patience': patience_counter,
                    'Elapsed Time': elapsed_time_str
                }
                history.append(step_stats)

                log_line = (
                    f"{step_stats['Steps']:>5d} | {step_stats['Epoch']:>5.2f} | "
                    f"{step_stats['Train RW-MSE ($)']:>16.2f} | {step_stats['Val RW-MSE ($)']:>14.2f} | "
                    f"{step_stats['Patience']:>8d} | {step_stats['Elapsed Time']}"
                )
                print(log_line)

                if early_stop_flag:
                    print(f"\n--- Early Stopping Triggered at Step {global_step_count} ---")


    print("\n--- Training Complete ---")

    if best_model_state is not None:
        print(f"Loading best model state (Val RW-MSE ($): {np.sqrt(best_val_loss):.2f})")
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
    display(training_history)

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
    Number of Epochs: 30
    Logging Interval: 10
    ==================================================
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
    
    --- Starting Model Training ---
    Performing Step 0 evaluation...
    Step 0 evaluation complete.
    
    Steps | Epoch | Train RW-MSE ($) | Val RW-MSE ($) | Patience | Elapsed Time
    ---------------------------------------------------------------------------
        0 |  0.00 |           132.88 |         123.03 |        0 | 00:00:09
       10 |  0.16 |           114.93 |         122.54 |        0 | 00:00:20
       20 |  0.32 |           137.67 |         122.02 |        0 | 00:00:31
       30 |  0.48 |           119.90 |         121.43 |        0 | 00:00:42
       40 |  0.63 |           117.47 |         120.75 |        0 | 00:00:53
       50 |  0.79 |           116.41 |         119.97 |        0 | 00:01:04
       60 |  0.95 |           115.42 |         119.08 |        0 | 00:01:14
       70 |  1.11 |           115.83 |         118.09 |        0 | 00:01:26
       80 |  1.27 |           112.29 |         117.00 |        0 | 00:01:36
       90 |  1.43 |           117.05 |         115.81 |        0 | 00:01:47
      100 |  1.59 |           126.62 |         114.52 |        0 | 00:01:58
      110 |  1.75 |           115.06 |         113.13 |        0 | 00:02:09
      120 |  1.90 |           110.25 |         111.65 |        0 | 00:02:20
      130 |  2.06 |           104.43 |         110.06 |        0 | 00:02:31
      140 |  2.22 |           121.95 |         108.38 |        0 | 00:02:41
      150 |  2.38 |           106.62 |         106.61 |        0 | 00:02:52
      160 |  2.54 |           101.91 |         104.73 |        0 | 00:03:03
      170 |  2.70 |            99.96 |         102.76 |        0 | 00:03:13
      180 |  2.86 |           102.41 |         100.70 |        0 | 00:03:24
      190 |  3.02 |           101.82 |          98.55 |        0 | 00:03:36
      200 |  3.17 |            98.79 |          96.32 |        0 | 00:03:47
      210 |  3.33 |            99.71 |          94.02 |        0 | 00:03:57
      220 |  3.49 |            96.84 |          91.64 |        0 | 00:04:08
      230 |  3.65 |            88.92 |          89.21 |        0 | 00:04:19
      240 |  3.81 |            93.25 |          86.75 |        0 | 00:04:30
      250 |  3.97 |            84.39 |          84.26 |        0 | 00:04:40
      260 |  4.13 |            77.69 |          81.76 |        0 | 00:04:52
      270 |  4.29 |            86.43 |          79.29 |        0 | 00:05:02
      280 |  4.44 |            84.47 |          76.86 |        0 | 00:05:13
      290 |  4.60 |            71.48 |          74.52 |        0 | 00:05:24
      300 |  4.76 |            76.62 |          72.28 |        0 | 00:05:35
      310 |  4.92 |            68.07 |          70.18 |        0 | 00:05:46
      320 |  5.08 |            68.44 |          68.26 |        0 | 00:05:57
      330 |  5.24 |            66.54 |          66.54 |        0 | 00:06:08
      340 |  5.40 |            68.54 |          64.99 |        0 | 00:06:18
      350 |  5.56 |            67.10 |          63.62 |        0 | 00:06:29
      360 |  5.71 |            68.93 |          62.41 |        0 | 00:06:40
      370 |  5.87 |            55.18 |          61.27 |        0 | 00:06:51
      380 |  6.03 |            62.67 |          60.20 |        0 | 00:07:02
      390 |  6.19 |            58.01 |          59.21 |        0 | 00:07:13
      400 |  6.35 |            55.80 |          58.32 |        0 | 00:07:24
      410 |  6.51 |            56.28 |          57.55 |        0 | 00:07:35
      420 |  6.67 |            58.35 |          56.82 |        0 | 00:07:45
      430 |  6.83 |            57.45 |          56.21 |        0 | 00:07:56
      440 |  6.98 |            61.37 |          55.66 |        0 | 00:08:07
      450 |  7.14 |            54.73 |          55.12 |        0 | 00:08:18
      460 |  7.30 |            47.72 |          54.58 |        0 | 00:08:29
      470 |  7.46 |            48.47 |          54.11 |        0 | 00:08:40
      480 |  7.62 |            61.62 |          53.67 |        0 | 00:08:50
      490 |  7.78 |            58.32 |          53.25 |        0 | 00:09:01
      500 |  7.94 |            58.24 |          52.86 |        0 | 00:09:12
      510 |  8.10 |            53.94 |          52.46 |        0 | 00:09:23
      520 |  8.25 |            57.77 |          52.11 |        0 | 00:09:34
      530 |  8.41 |            53.89 |          51.75 |        0 | 00:09:45
      540 |  8.57 |            51.67 |          51.37 |        0 | 00:09:56
      550 |  8.73 |            54.88 |          51.04 |        0 | 00:10:07
      560 |  8.89 |            47.72 |          50.75 |        0 | 00:10:18
      570 |  9.05 |            42.65 |          50.44 |        0 | 00:10:29
      580 |  9.21 |            51.71 |          50.16 |        0 | 00:10:40
      590 |  9.37 |            46.31 |          49.87 |        0 | 00:10:51
      600 |  9.52 |            57.16 |          49.60 |        0 | 00:11:02
      610 |  9.68 |            50.84 |          49.35 |        0 | 00:11:13
      620 |  9.84 |            51.95 |          49.08 |        0 | 00:11:23
      630 | 10.00 |            74.24 |          48.83 |        0 | 00:11:34
      640 | 10.16 |            50.29 |          48.62 |        0 | 00:11:46
      650 | 10.32 |            52.00 |          48.40 |        0 | 00:11:56
      660 | 10.48 |            45.65 |          48.18 |        0 | 00:12:07
      670 | 10.63 |            53.89 |          47.98 |        0 | 00:12:18
      680 | 10.79 |            45.60 |          47.80 |        0 | 00:12:28
      690 | 10.95 |            52.45 |          47.62 |        0 | 00:12:39
      700 | 11.11 |            46.96 |          47.46 |        0 | 00:12:51
      710 | 11.27 |            52.60 |          47.31 |        0 | 00:13:01
      720 | 11.43 |            49.07 |          47.15 |        0 | 00:13:12
      730 | 11.59 |            48.70 |          46.98 |        0 | 00:13:23
      740 | 11.75 |            52.72 |          46.84 |        0 | 00:13:33
      750 | 11.90 |            49.66 |          46.70 |        0 | 00:13:44
      760 | 12.06 |            47.45 |          46.60 |        0 | 00:13:56
      770 | 12.22 |            41.36 |          46.49 |        0 | 00:14:06
      780 | 12.38 |            47.54 |          46.37 |        0 | 00:14:17
      790 | 12.54 |            43.79 |          46.26 |        0 | 00:14:28
      800 | 12.70 |            48.04 |          46.16 |        0 | 00:14:39
      810 | 12.86 |            44.99 |          46.05 |        0 | 00:14:49
      820 | 13.02 |            48.20 |          45.97 |        0 | 00:15:01
      830 | 13.17 |            45.97 |          45.89 |        0 | 00:15:11
      840 | 13.33 |            42.75 |          45.82 |        0 | 00:15:22
      850 | 13.49 |            45.62 |          45.76 |        0 | 00:15:33
      860 | 13.65 |            42.97 |          45.71 |        1 | 00:15:44
      870 | 13.81 |            46.50 |          45.64 |        0 | 00:15:54
      880 | 13.97 |            44.96 |          45.57 |        0 | 00:16:05
      890 | 14.13 |            46.84 |          45.50 |        0 | 00:16:16
      900 | 14.29 |            42.61 |          45.48 |        1 | 00:16:27
      910 | 14.44 |            46.75 |          45.40 |        0 | 00:16:38
      920 | 14.60 |            45.91 |          45.35 |        1 | 00:16:49
      930 | 14.76 |            46.56 |          45.29 |        0 | 00:17:00
      940 | 14.92 |            40.12 |          45.24 |        1 | 00:17:10
      950 | 15.08 |            47.04 |          45.20 |        0 | 00:17:21
      960 | 15.24 |            45.46 |          45.17 |        1 | 00:17:32
      970 | 15.40 |            48.55 |          45.11 |        0 | 00:17:43
      980 | 15.56 |            47.61 |          45.07 |        1 | 00:17:54
      990 | 15.71 |            44.02 |          45.02 |        0 | 00:18:05
     1000 | 15.87 |            42.63 |          44.98 |        1 | 00:18:16
     1010 | 16.03 |            49.80 |          44.96 |        0 | 00:18:28
     1020 | 16.19 |            48.30 |          44.93 |        1 | 00:18:38
     1030 | 16.35 |            45.61 |          44.86 |        0 | 00:18:49
     1040 | 16.51 |            48.75 |          44.82 |        1 | 00:19:00
     1050 | 16.67 |            45.26 |          44.79 |        0 | 00:19:10
     1060 | 16.83 |            46.12 |          44.76 |        1 | 00:19:21
     1070 | 16.98 |            44.99 |          44.72 |        0 | 00:19:31
     1080 | 17.14 |            47.10 |          44.68 |        1 | 00:19:43
     1090 | 17.30 |            47.19 |          44.65 |        0 | 00:19:54
     1100 | 17.46 |            44.18 |          44.61 |        1 | 00:20:05
     1110 | 17.62 |            47.02 |          44.58 |        0 | 00:20:15
     1120 | 17.78 |            43.29 |          44.54 |        1 | 00:20:26
     1130 | 17.94 |            43.51 |          44.51 |        0 | 00:20:37
     1140 | 18.10 |            46.70 |          44.49 |        1 | 00:20:48
     1150 | 18.25 |            45.97 |          44.48 |        2 | 00:20:59
     1160 | 18.41 |            44.81 |          44.42 |        0 | 00:21:10
     1170 | 18.57 |            45.59 |          44.38 |        1 | 00:21:21
     1180 | 18.73 |            46.75 |          44.36 |        0 | 00:21:32
     1190 | 18.89 |            52.50 |          44.33 |        1 | 00:21:42
     1200 | 19.05 |            44.04 |          44.31 |        0 | 00:21:54
     1210 | 19.21 |            45.14 |          44.28 |        1 | 00:22:05
     1220 | 19.37 |            44.06 |          44.25 |        0 | 00:22:16
     1230 | 19.52 |            41.75 |          44.23 |        1 | 00:22:27
     1240 | 19.68 |            49.97 |          44.20 |        0 | 00:22:38
     1250 | 19.84 |            47.94 |          44.18 |        1 | 00:22:49
     1260 | 20.00 |            36.56 |          44.15 |        2 | 00:22:59
     1270 | 20.16 |            43.43 |          44.14 |        0 | 00:23:11
     1280 | 20.32 |            46.63 |          44.12 |        1 | 00:23:22
     1290 | 20.48 |            44.71 |          44.12 |        2 | 00:23:33
     1300 | 20.63 |            44.10 |          44.07 |        0 | 00:23:44
     1310 | 20.79 |            47.80 |          44.04 |        1 | 00:23:54
     1320 | 20.95 |            45.78 |          44.01 |        0 | 00:24:05
     1330 | 21.11 |            45.20 |          43.98 |        1 | 00:24:17
     1340 | 21.27 |            49.17 |          43.98 |        2 | 00:24:28
     1350 | 21.43 |            40.93 |          43.97 |        3 | 00:24:39
     1360 | 21.59 |            40.99 |          43.92 |        0 | 00:24:50
     1370 | 21.75 |            43.11 |          43.91 |        1 | 00:25:01
     1380 | 21.90 |            42.39 |          43.89 |        2 | 00:25:12
     1390 | 22.06 |            46.78 |          43.86 |        0 | 00:25:23
     1400 | 22.22 |            45.65 |          43.84 |        1 | 00:25:34
     1410 | 22.38 |            42.03 |          43.83 |        2 | 00:25:45
     1420 | 22.54 |            45.06 |          43.80 |        0 | 00:25:56
     1430 | 22.70 |            41.87 |          43.78 |        1 | 00:26:06
     1440 | 22.86 |            44.76 |          43.76 |        2 | 00:26:17
     1450 | 23.02 |            44.08 |          43.77 |        3 | 00:26:29
     1460 | 23.17 |            40.52 |          43.73 |        0 | 00:26:40
     1470 | 23.33 |            47.20 |          43.71 |        1 | 00:26:51
     1480 | 23.49 |            40.91 |          43.69 |        2 | 00:27:02
     1490 | 23.65 |            43.63 |          43.69 |        3 | 00:27:13
     1500 | 23.81 |            45.09 |          43.67 |        0 | 00:27:24
     1510 | 23.97 |            43.37 |          43.65 |        1 | 00:27:35
     1520 | 24.13 |            47.15 |          43.62 |        2 | 00:27:46
     1530 | 24.29 |            47.05 |          43.63 |        3 | 00:27:57
     1540 | 24.44 |            45.62 |          43.60 |        0 | 00:28:08
     1550 | 24.60 |            44.39 |          43.57 |        1 | 00:28:18
     1560 | 24.76 |            43.01 |          43.56 |        2 | 00:28:29
     1570 | 24.92 |            44.41 |          43.55 |        3 | 00:28:40
     1580 | 25.08 |            43.57 |          43.52 |        0 | 00:28:51
     1590 | 25.24 |            43.40 |          43.50 |        1 | 00:29:02
     1600 | 25.40 |            47.89 |          43.49 |        2 | 00:29:13
     1610 | 25.56 |            48.57 |          43.48 |        3 | 00:29:24
     1620 | 25.71 |            47.50 |          43.47 |        4 | 00:29:35
     1630 | 25.87 |            45.60 |          43.45 |        0 | 00:29:46
     1640 | 26.03 |            39.61 |          43.43 |        1 | 00:29:56
     1650 | 26.19 |            40.98 |          43.41 |        2 | 00:30:07
     1660 | 26.35 |            42.15 |          43.40 |        0 | 00:30:18
     1670 | 26.51 |            46.79 |          43.38 |        1 | 00:30:29
     1680 | 26.67 |            41.82 |          43.40 |        2 | 00:30:40
     1690 | 26.83 |            39.41 |          43.36 |        3 | 00:30:51
     1700 | 26.98 |            47.52 |          43.35 |        4 | 00:31:01
     1710 | 27.14 |            44.49 |          43.35 |        5 | 00:31:13
    
    --- Early Stopping Triggered at Step 1710 ---
    
    --- Training Complete ---
    Loading best model state (Val RW-MSE ($): 43.40)
    
    Saving artifacts to ./nyc_artifacts_20251009_142859.pt and /content/drive/MyDrive/Colab_Notebooks/Airbnb_Project/nyc_artifacts_20251009_142859.pt...
    Artifacts saved successfully.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_performance(model, data_loader, df, device):
    """
    Runs inference on a dataset and returns a detailed performance analysis,
    segmented by price range.
    """
    model.eval()
    predictions = []

    # We only need predictions, not targets from the loader, as they are in the df
    with torch.no_grad():
        for batch in data_loader:
            # Move all parts of the batch to the correct device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        batch[key][sub_key] = sub_value.to(device, non_blocking=True)

            preds = model(batch)
            predictions.extend(preds.cpu().numpy())

    # Ensure the number of predictions matches the dataframe size
    # This handles the case where the last batch is smaller
    results_df = df.iloc[:len(predictions)].copy()
    results_df['predicted_price'] = predictions

    # Calculate errors
    results_df['absolute_error'] = (results_df['predicted_price'] - results_df['target_price']).abs()
    results_df['percentage_error'] = (results_df['absolute_error'] / results_df['target_price']) * 100

    # Create price bins for segmentation
    price_bins = [0, 75, 150, 250, 400, results_df['target_price'].max()]
    bin_labels = ['_'.join(map(str, map(int, i))) for i in zip(price_bins[:-1], price_bins[1:])]
    results_df['price_bin'] = pd.cut(results_df['target_price'], bins=price_bins, labels=bin_labels, right=False)


    # Aggregate performance by price bin
    performance_summary = results_df.groupby('price_bin', as_index=False).agg(
        num_listings=('listing_id', 'count'),
        mean_abs_error=('absolute_error', 'mean'),
        mean_abs_perc_error=('percentage_error', 'mean')
    )

    return performance_summary, results_df
```


```python
# --- Main Analysis Execution ---

# 1. Load the artifacts from your training run
# Assumes 'training_artifacts' variable exists from the previous cell
processor = training_artifacts['processor']
config = training_artifacts['config']
model_state_dict = training_artifacts['model_state_dict']

# 2. Re-create the validation set and data loader
# This is necessary because the loader itself isn't saved in the artifacts
print("Re-creating validation data loader...")
_, val_df = load_and_split_data(config)
val_features_cpu = preprocess_and_tensorize_CPU(processor, val_df)
# We only need the validation loader, so we can ignore the training one
_, val_loader = create_dataloaders(None, val_features_cpu, config)
print("DataLoader created.")

# 3. Instantiate a new model and load the trained weights
print("Loading trained model...")
model = AdditiveAxisModel(processor, config.DEVICE)
model.load_state_dict(model_state_dict)
print("Model loaded.")

# 4. Run the performance analysis
print("\n--- Analyzing performance on the validation set ---")
performance_summary, detailed_results = analyze_model_performance(
    model=model,
    data_loader=val_loader,
    df=val_df,
    device=config.DEVICE
)

print("\nPerformance by Price Bracket:")
print(performance_summary.to_string(index=False, float_format="%.2f"))

# --- 5. Visualize the results ---
print("\nGenerating plot...")
plt.figure(figsize=(12, 6))
sns.barplot(data=performance_summary, x='price_bin', y='mean_abs_perc_error', palette='viridis')
plt.title('Mean Absolute Percentage Error (MAPE) by Price Bracket', fontsize=16)
plt.xlabel('True Price Range ($)', fontsize=12)
plt.ylabel('MAPE (%)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
```

    Re-creating validation data loader...
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



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    /tmp/ipython-input-2706119345.py in <cell line: 0>()
         13 val_features_cpu = preprocess_and_tensorize_CPU(processor, val_df)
         14 # We only need the validation loader, so we can ignore the training one
    ---> 15 _, val_loader = create_dataloaders(None, val_features_cpu, config)
         16 print("DataLoader created.")
         17 


    /tmp/ipython-input-1787674024.py in create_dataloaders(train_features_cpu, val_features_cpu, config)
         49         return collated_batch
         50 
    ---> 51     train_dataset = AirbnbPriceDataset(train_features_cpu)
         52     val_dataset = AirbnbPriceDataset(val_features_cpu)
         53 


    /tmp/ipython-input-2392459920.py in __init__(self, features)
          2     def __init__(self, features: dict):
          3         self.features = features
    ----> 4         self.n_samples = len(features['sample_weight'])
          5 
          6     def __len__(self):


    TypeError: 'NoneType' object is not subscriptable



```python
training_artifacts.keys()
```




    dict_keys(['config', 'processor', 'model_state_dict', 'optimizer_state_dict', 'history'])




```python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_performance(model, data_loader, df, device):
    """
    Runs inference on a dataset and returns a detailed performance analysis,
    segmented by price range.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=True)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        batch[key][sub_key] = sub_value.to(device, non_blocking=True)

            preds = model(batch)
            predictions.extend(preds.cpu().numpy())

    results_df = df.iloc[:len(predictions)].copy()
    results_df['predicted_price'] = predictions

    results_df['absolute_error'] = (results_df['predicted_price'] - results_df['target_price']).abs()
    results_df['percentage_error'] = (results_df['absolute_error'] / results_df['target_price']) * 100

    price_bins = [0, 75, 150, 250, 400, results_df['target_price'].max() + 1]
    results_df['price_bin'] = pd.cut(results_df['target_price'], bins=price_bins, right=False)

    performance_summary = results_df.groupby('price_bin').agg(
        num_listings=('listing_id', 'count'),
        mean_abs_error=('absolute_error', 'mean'),
        mean_abs_perc_error=('percentage_error', 'mean')
    ).reset_index()

    return performance_summary, results_df

# --- Performance Analysis ---

# 1. Unpack the necessary components from your saved artifacts
print("Unpacking artifacts...")
processor = training_artifacts['processor']
config = training_artifacts['config']
model_state_dict = training_artifacts['model_state_dict']

# 2. Recreate the data splits and data loaders exactly as they were during training
print("Recreating data splits and data loaders...")
train_df, val_df = load_and_split_data(config)
train_features_cpu = preprocess_and_tensorize_CPU(processor, train_df)
val_features_cpu = preprocess_and_tensorize_CPU(processor, val_df)
train_loader, val_loader = create_dataloaders(train_features_cpu, val_features_cpu, config)
print("Data loaders recreated.")

# 3. Instantiate a new model and load the trained (best) weights
print("Loading trained model...")
model = AdditiveAxisModel(processor, config.DEVICE)
model.load_state_dict(model_state_dict)
print("Model loaded.")

# 4. Analyze performance on the TRAINING set
print("\n--- Analyzing performance on the TRAINING set ---")
train_summary, _ = analyze_model_performance(model, train_loader, train_df, config.DEVICE)
print("\nTraining Set Performance by Price Bracket:")
print(train_summary.to_string(index=False, float_format="%.2f"))

# 5. Analyze performance on the VALIDATION set
print("\n--- Analyzing performance on the VALIDATION set ---")
val_summary, _ = analyze_model_performance(model, val_loader, val_df, config.DEVICE)
print("\nValidation Set Performance by Price Bracket:")
print(val_summary.to_string(index=False, float_format="%.2f"))

# 6. Generate separate plots
print("\nGenerating plots...")
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
fig.suptitle('Model Performance Analysis: Additive Dollar Model', fontsize=20)

# Plot for Training Data
sns.barplot(ax=axes[0], data=train_summary, x='price_bin', y='mean_abs_perc_error', palette='viridis')
axes[0].set_title('Training Set: MAPE by Price Bracket', fontsize=16)
axes[0].set_xlabel('True Price Range ($)', fontsize=12)
axes[0].set_ylabel('Mean Absolute Percentage Error (MAPE %)', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Plot for Validation Data
sns.barplot(ax=axes[1], data=val_summary, x='price_bin', y='mean_abs_perc_error', palette='viridis')
axes[1].set_title('Validation Set: MAPE by Price Bracket', fontsize=16)
axes[1].set_xlabel('True Price Range ($)', fontsize=12)
axes[1].set_ylabel('') # Hide redundant y-axis label
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```
  
    --- Analyzing performance on the TRAINING set ---
    
    Training Set Performance by Price Bracket:
         price_bin  num_listings  mean_abs_error  mean_abs_perc_error
       [0.0, 75.0)          9876          129.14               242.23
     [75.0, 150.0)         23841           94.03                90.46
    [150.0, 250.0)         17685           84.08                44.97
    [250.0, 400.0)          8304          144.66                46.65
    [400.0, 964.0)          3882          366.30                65.24
    
    --- Analyzing performance on the VALIDATION set ---
    
    Validation Set Performance by Price Bracket:
         price_bin  num_listings  mean_abs_error  mean_abs_perc_error
       [0.0, 75.0)          2433           36.82                66.85
     [75.0, 150.0)          6080           49.01                45.41
    [150.0, 250.0)          4399           53.01                28.56
    [250.0, 400.0)          2060           68.87                22.42
    [400.0, 963.0)           925          189.24                33.25
    
    Generating plots...



    
![png](02_modeling_oct_9_files/02_modeling_oct_9_26_3.png)
    



```python

```
