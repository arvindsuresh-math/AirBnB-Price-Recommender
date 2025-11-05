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
!pip install transformers
!pip install matplotlib
!pip install seaborn
```

```python
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
```


```python
class FeatureProcessor:
    """
    Prepares raw DataFrame columns into numerical features for the model.
    The fit/transform pattern prevents data leakage from the validation set.
    """
    def __init__(self, embedding_dim_geo: int = 32):
        self.vocabs, self.scalers = {}, {}
        self.embedding_dim_geo = embedding_dim_geo
        self.categorical_cols = ["property_type", "room_type"]
        self.numerical_cols = ["accommodates", "review_scores_rating", "review_scores_cleanliness",
                               "review_scores_checkin", "review_scores_communication",
                               "review_scores_location", "review_scores_value",
                               "bedrooms", "beds", "bathrooms"]
        self.log_transform_cols = ["total_reviews"]

    def fit(self, df: pd.DataFrame):
        """Fits scalers and vocabularies based on the training data."""
        print("Fitting FeatureProcessor...")
        for col in self.categorical_cols:
            self.vocabs[col] = {val: i for i, val in enumerate(["<UNK>"] + sorted(df[col].unique()))}

        for col in self.numerical_cols + self.log_transform_cols:
            vals = df[col].astype(float)
            vals = np.log1p(vals) if col in self.log_transform_cols else vals
            self.scalers[col] = {'mean': vals.mean(), 'std': vals.std()}
        print("Fit complete.")

    def transform(self, df: pd.DataFrame, neighborhood_log_means: dict) -> dict:
        """Transforms a DataFrame into a dictionary of feature tensors."""
        df = df.copy()
        # --- Target Variable Transformation ---
        df['neighborhood_log_mean'] = df['neighbourhood_cleansed'].map(neighborhood_log_means)
        # Handle neighborhoods present in validation but not training
        global_mean = sum(neighborhood_log_means.values()) / len(neighborhood_log_means)
        df['neighborhood_log_mean'].fillna(global_mean, inplace=True)

        target_log_deviation = (np.log1p(df["price"]) - df['neighborhood_log_mean']).to_numpy(dtype=np.float32)

        # --- Feature Engineering ---
        # Geospatial positional encoding
        half_dim = self.embedding_dim_geo // 2
        lat = df["latitude"].to_numpy(dtype=np.float32)
        lon = df["longitude"].to_numpy(dtype=np.float32)
        def pe(arr, max_val, d):
            pos = (arr / max_val) * 10000.0
            idx = np.arange(0, d, 2, dtype=np.float32)
            div = np.exp(-(np.log(10000.0) / d) * idx)
            s, c = np.sin(pos[:, None] * div[None, :]), np.cos(pos[:, None] * div[None, :])
            out = np.empty((arr.shape[0], d), dtype=np.float32)
            out[:, 0::2], out[:, 1::2] = s, c
            return out
        geo_position = np.hstack([pe(lat, 90.0, half_dim), pe(lon, 180.0, half_dim)])

        # Size & Capacity features
        size_features = {
            "property_type": df["property_type"].map(self.vocabs["property_type"]).fillna(0).astype(np.int64),
            "room_type": df["room_type"].map(self.vocabs["room_type"]).fillna(0).astype(np.int64)
        }
        for col in ["accommodates", "bedrooms", "beds", "bathrooms"]:
            x = df[col].astype(float)
            size_features[col] = ((x - self.scalers[col]["mean"]) / self.scalers[col]["std"]).astype(np.float32)

        # Quality & Reputation features
        quality_features = {}
        quality_num_cols = set(self.numerical_cols) - set(size_features.keys()) - set(self.categorical_cols)
        for col in quality_num_cols:
            x = df[col].astype(float)
            quality_features[col] = ((x - self.scalers[col]["mean"]) / self.scalers[col]["std"]).astype(np.float32)

        tr_log = np.log1p(df["total_reviews"].astype(float))
        quality_features["total_reviews"] = (tr_log - self.scalers["total_reviews"]["mean"]) / self.scalers["total_reviews"]["std"]
        quality_features["host_is_superhost"] = df["host_is_superhost"].astype(np.float32)

        # Seasonality (cyclical) features
        month = df["month"].to_numpy(np.float32)
        season_cyc = np.stack([np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)], axis=1)

        return {
            "location": {"geo_position": geo_position},
            "size_capacity": {k: v.to_numpy() for k, v in size_features.items()},
            "quality": {k: v.to_numpy() for k, v in quality_features.items()},
            "amenities_text": df["amenities"].tolist(),
            "description_text": df["description"].tolist(),
            "seasonality": {"cyclical": season_cyc},
            "target_price": df["price"].to_numpy(dtype=np.float32),
            "target_log_deviation": target_log_deviation,
            "neighborhood_log_mean": df['neighborhood_log_mean'].to_numpy(dtype=np.float32),
        }
```


```python
class AirbnbPriceDataset(Dataset):
    """PyTorch Dataset to handle tokenization and feature collation."""
    def __init__(self, features: dict, tokenizer):
        self.features = features
        self.tokenizer = tokenizer
        self.n_samples = len(features['target_price'])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        item = {
            'loc_geo_position': torch.tensor(self.features['location']['geo_position'][index], dtype=torch.float32),
            'season_cyclical': torch.tensor(self.features['seasonality']['cyclical'][index], dtype=torch.float32),
            'target_price': torch.tensor(self.features['target_price'][index], dtype=torch.float32),
            'target_log_deviation': torch.tensor(self.features['target_log_deviation'][index], dtype=torch.float32),
            'neighborhood_log_mean': torch.tensor(self.features['neighborhood_log_mean'][index], dtype=torch.float32),
        }
        for k, v in self.features['size_capacity'].items():
            dtype = torch.long if k in ['property_type', 'room_type'] else torch.float32
            item[f'size_{k}'] = torch.tensor(v[index], dtype=dtype)
        for k, v in self.features['quality'].items():
            item[f'qual_{k}'] = torch.tensor(v[index], dtype=dtype)

        # On-the-fly tokenization
        item['amenities_tokens'] = self.tokenizer(
            self.features['amenities_text'][index],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        item['description_tokens'] = self.tokenizer(
            self.features['description_text'][index],
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        return item

def create_dataloaders(train_features: dict, val_features: dict, config: dict):
    """Initializes and returns the training and validation DataLoaders."""
    tokenizer = AutoTokenizer.from_pretrained(config['TEXT_MODEL_NAME'], use_fast=True)
    train_dataset = AirbnbPriceDataset(train_features, tokenizer)
    val_dataset = AirbnbPriceDataset(val_features, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['VALIDATION_BATCH_SIZE'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("DataLoaders created.")
    return train_loader, val_loader
```


```python
class AdditiveAxisModel(nn.Module):
    """
    A multi-axis neural network that predicts price deviation from a baseline.

    The model is composed of six sub-networks, each processing a different
    modality of the listing data. The final output is the sum of the outputs
    from each sub-network, representing the predicted log-price deviation.
    """
    def __init__(self, processor: FeatureProcessor, config: dict):
        super().__init__()
        self.device = config['DEVICE']

        # --- Embeddings for Categorical Features ---
        self.embed_property_type = nn.Embedding(len(processor.vocabs['property_type']), 8)
        self.embed_room_type = nn.Embedding(len(processor.vocabs['room_type']), 4)

        # --- Text Transformer (with last layer unfrozen for fine-tuning) ---
        self.text_transformer = SentenceTransformer(config['TEXT_MODEL_NAME'], device=self.device)
        for param in self.text_transformer.parameters():
            param.requires_grad = False
        # Unfreeze the final transformer layer
        for param in self.text_transformer[0].auto_model.encoder.layer[-1].parameters():
            param.requires_grad = True

        # --- Helper to Dynamically Create MLP Sub-networks ---
        def _create_mlp(in_features, layer_sizes):
            layers = []
            for size in layer_sizes:
                layers.append(nn.Linear(in_features, size))
                layers.append(nn.ReLU())
                in_features = size
            return nn.Sequential(*layers)

        # --- Dynamically create sub-network "bodies" and "heads" ---
        text_embed_dim = self.text_transformer.get_sentence_embedding_dimension()
        self.loc_subnet_body = _create_mlp(32, config['HIDDEN_LAYERS_LOCATION'])
        self.size_subnet_body = _create_mlp(16, config['HIDDEN_LAYERS_SIZE_CAPACITY'])
        self.qual_subnet_body = _create_mlp(8, config['HIDDEN_LAYERS_QUALITY'])
        self.amenities_subnet_body = _create_mlp(text_embed_dim, config['HIDDEN_LAYERS_AMENITIES'])
        self.desc_subnet_body = _create_mlp(text_embed_dim, config['HIDDEN_LAYERS_DESCRIPTION'])
        self.season_subnet_body = _create_mlp(2, config['HIDDEN_LAYERS_SEASONALITY'])

        self.loc_subnet_head = nn.Linear(config['HIDDEN_LAYERS_LOCATION'][-1], 1)
        self.size_subnet_head = nn.Linear(config['HIDDEN_LAYERS_SIZE_CAPACITY'][-1], 1)
        self.qual_subnet_head = nn.Linear(config['HIDDEN_LAYERS_QUALITY'][-1], 1)
        self.amenities_subnet_head = nn.Linear(config['HIDDEN_LAYERS_AMENITIES'][-1], 1)
        self.desc_subnet_head = nn.Linear(config['HIDDEN_LAYERS_DESCRIPTION'][-1], 1)
        self.season_subnet_head = nn.Linear(config['HIDDEN_LAYERS_SEASONALITY'][-1], 1)

        self.to(self.device)

    def forward_with_hidden_states(self, batch: dict) -> dict:
        """Performs a full forward pass, returning predictions, contributions, and hidden states."""
        # --- Prepare Inputs for each Axis ---
        loc_input = batch['loc_geo_position']
        size_input = torch.cat(
            [self.embed_property_type(batch['size_property_type']),
             self.embed_room_type(batch['size_room_type']),
             batch['size_accommodates'].unsqueeze(1),
             batch['size_bedrooms'].unsqueeze(1),
             batch['size_beds'].unsqueeze(1),
             batch['size_bathrooms'].unsqueeze(1)
             ], dim=1
        )
        qual_cols = ["review_scores_rating", "review_scores_cleanliness", "review_scores_checkin",
                     "review_scores_communication", "review_scores_location", "review_scores_value",
                     "total_reviews", "host_is_superhost"]
        qual_input = torch.cat([batch[f'qual_{c}'].unsqueeze(1) for c in qual_cols], dim=1)

        amenities_tokens = {k: v.squeeze(1) for k, v in batch['amenities_tokens'].items()}
        desc_tokens = {k: v.squeeze(1) for k, v in batch['description_tokens'].items()}
        amenities_embed = self.text_transformer(amenities_tokens)['sentence_embedding']
        desc_embed = self.text_transformer(desc_tokens)['sentence_embedding']

        # --- Process through Sub-network Bodies (to get hidden states) ---
        h_loc = self.loc_subnet_body(loc_input)
        h_size = self.size_subnet_body(size_input)
        h_qual = self.qual_subnet_body(qual_input)
        h_amenities = self.amenities_subnet_body(amenities_embed)
        h_desc = self.desc_subnet_body(desc_embed)
        h_season = self.season_subnet_body(batch['season_cyclical'])

        # --- Process through Sub-network Heads (to get price contributions) ---
        p_loc = self.loc_subnet_head(h_loc)
        p_size = self.size_subnet_head(h_size)
        p_qual = self.qual_subnet_head(h_qual)
        p_amenities = self.amenities_subnet_head(h_amenities)
        p_desc = self.desc_subnet_head(h_desc)
        p_season = self.season_subnet_head(h_season)

        predicted_log_deviation = (p_loc + p_size + p_qual + p_amenities + p_desc + p_season)

        return {
            'predicted_log_deviation': predicted_log_deviation.squeeze(-1),
            'p_location': p_loc.squeeze(-1),
            'p_size_capacity': p_size.squeeze(-1),
            'p_quality': p_qual.squeeze(-1),
            'p_amenities': p_amenities.squeeze(-1),
            'p_description': p_desc.squeeze(-1),
            'p_seasonality': p_season.squeeze(-1),
            'h_location': h_loc,
            'h_size_capacity': h_size,
            'h_quality': h_qual,
            'h_amenities': h_amenities,
            'h_description': h_desc,
            'h_seasonality': h_season,
        }

    def forward_with_price(self, batch: dict) -> dict:
        """Calls the base method and returns only the price decomposition components."""
        all_outputs = self.forward_with_hidden_states(batch)
        return {k: v for k, v in all_outputs.items() if not k.startswith('h_')}

    def forward(self, batch: dict) -> torch.Tensor:
        """The standard forward pass for training, returning only the final prediction tensor."""
        return self.forward_with_hidden_states(batch)['predicted_log_deviation']
```


```python
def run_inference_with_details(model, data_loader, device):
    """
    Runs inference and returns the full decomposition, including hidden states,
    for each listing.
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Inference with Details", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                else: batch[k] = {sk: sv.to(device) for sk, sv in v.items()}

            # FIX: Call forward_with_hidden_states to get ALL outputs
            batch_outputs = model.forward_with_hidden_states(batch)

            # Add neighborhood mean to reconstruct full log price later
            batch_outputs['neighborhood_log_mean'] = batch['neighborhood_log_mean']
            outputs.append({k: v.cpu() for k, v in batch_outputs.items()})

    # This part remains the same, but will now process 'h_' tensors as well
    final_outputs = {key: torch.cat([o[key] for o in outputs]).numpy() for key in outputs[0].keys()}

    # Reconstruct final predicted price
    predicted_log = final_outputs['predicted_log_deviation'] + final_outputs['neighborhood_log_mean']
    final_outputs['predicted_price'] = np.expm1(predicted_log)
    return final_outputs
```


```python
# --- MAIN SCRIPT TO BUILD THE APP DATABASE (Corrected) ---

CITIES = ["nyc", "toronto"]
ARTIFACTS_PATH = "/content/drive/MyDrive/Colab_Notebooks/Airbnb_Project/"
DATA_PATH = "./"
OUTPUT_PATH = "/content/drive/MyDrive/Colab_Notebooks/Airbnb_Project/app_data/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Map city names to their specific artifact filenames ---
artifact_filenames = {
    "nyc": "nyc_model_artifacts_20251021_151319.pt",
    "toronto": "toronto_model_artifacts_20251021_164438.pt"
}

for city in CITIES:
    print("\n" + "="*50)
    print(f"Processing city: {city.upper()}")
    print("="*50)

    # 1. Load the saved artifact dictionary
    filename = artifact_filenames.get(city)
    if not filename:
        print(f"ERROR: No artifact filename found for '{city}' in the dictionary. Skipping.")
        continue

    artifact_file = os.path.join(ARTIFACTS_PATH, filename)
    if not os.path.exists(artifact_file):
        print(f"ERROR: Artifact file not found at path: {artifact_file}. Skipping.")
        continue

    print(f"Loading artifact: {filename}")
    # --- FIX: Add weights_only=False to allow loading of the FeatureProcessor object ---
    artifacts = torch.load(artifact_file, map_location=torch.device('cpu'), weights_only=False)

    config = artifacts['config']
    processor = artifacts['feature_processor']

    # 2. Re-instantiate the model and load the trained weights
    model = AdditiveAxisModel(processor, config)
    model.load_state_dict(artifacts['model_state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # 3. Load the raw dataset
    dataset_filename = f"{DATA_PATH}{city}_dataset_oct_20.parquet"
    if not os.path.exists(dataset_filename):
        print(f"ERROR: Raw dataset not found at path: {dataset_filename}. Skipping.")
        continue

    raw_df = pd.read_parquet(dataset_filename)
    raw_df = raw_df[raw_df["price"] > 0].copy().reset_index(drop=True)

    neighborhood_log_means = np.log1p(raw_df.groupby('neighbourhood_cleansed')['price'].mean()).to_dict()

    # 4. Create a DataLoader for the entire dataset
    features = processor.transform(raw_df, neighborhood_log_means)
    tokenizer = AutoTokenizer.from_pretrained(config['TEXT_MODEL_NAME'])
    dataset = AirbnbPriceDataset(features, tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=config.get('VALIDATION_BATCH_SIZE', 512),
        shuffle=False,
        num_workers=2
    )

    # 5. Run inference to get all predictions and hidden states
    print(f"Running inference on {len(raw_df)} listings for {city}...")
    details_outputs = run_inference_with_details(model, data_loader, device)

    # 6. Combine raw data with the new detailed outputs
    app_database_df = raw_df.copy()
    for key, value in details_outputs.items():
        if len(value) == len(app_database_df):
            app_database_df[key] = list(value)
        else:
            print(f"Warning: Length mismatch for key '{key}'. Skipping this column.")

    # Rename the column for app clarity
    app_database_df.rename(columns={'neighborhood_log_mean': 'p_base'}, inplace=True)

    # 7. Save the final, self-contained app database file
    output_filename = os.path.join(OUTPUT_PATH, f"{city}_app_database.parquet")
    app_database_df.to_parquet(output_filename)
    print(f"Successfully created app database for {city.upper()}!")
    print(f"Saved to: {output_filename}")
```

    
    ==================================================
    Processing city: NYC
    ==================================================
    Loading artifact: nyc_model_artifacts_20251021_151319.pt


    /tmp/ipython-input-2606712342.py:35: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df['neighborhood_log_mean'].fillna(global_mean, inplace=True)


    Running inference on 127590 listings for nyc...



    Running Inference with Details:   0%|          | 0/250 [00:00<?, ?it/s]


    Successfully created app database for NYC!
    Saved to: /content/drive/MyDrive/Colab_Notebooks/Airbnb_Project/app_data/nyc_app_database.parquet
    
    ==================================================
    Processing city: TORONTO
    ==================================================
    Loading artifact: toronto_model_artifacts_20251021_164438.pt


    /tmp/ipython-input-2606712342.py:35: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
    The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
    
    For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
    
    
      df['neighborhood_log_mean'].fillna(global_mean, inplace=True)


    Running inference on 86392 listings for toronto...



    Running Inference with Details:   0%|          | 0/169 [00:00<?, ?it/s]


    Successfully created app database for TORONTO!
    Saved to: /content/drive/MyDrive/Colab_Notebooks/Airbnb_Project/app_data/toronto_app_database.parquet

