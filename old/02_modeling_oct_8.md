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
import json
```


```python
class Config:
    # --- Data and Environment ---
    CITY: str = "nyc"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data Pre-processing ---
    VAL_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # --- Model Training ---
    BATCH_SIZE: int = 1024
    LEARNING_RATE: float = 1e-3
    N_EPOCHS: int = 20

    # --- Logging ---
    PRINT_EVERY_N_STEPS: int = 25 # How often to print train loss
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
        random_state=config.RANDOM_STATE,
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
class AirbnbPriceDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features
        self.n_samples = len(features['sample_weight'])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        item = {}
        # Location
        item['loc_geo_position'] = torch.tensor(self.features['location']['geo_position'][index], dtype=torch.float32)
        item['loc_neighbourhood'] = torch.tensor(self.features['location']['neighbourhood'][index], dtype=torch.long)

        # Size & Capacity
        for k, v in self.features['size_capacity'].items():
            dtype = torch.float32 if k == 'accommodates' else torch.long
            item[f'size_{k}'] = torch.tensor(v[index], dtype=dtype)

        # Quality
        for k, v in self.features['quality'].items():
            item[f'qual_{k}'] = torch.tensor(v[index], dtype=torch.float32)

        # Amenities & Seasonality
        item['amenities_text'] = self.features['amenities']['text'][index]
        item['season_cyclical'] = torch.tensor(self.features['seasonality']['cyclical'][index], dtype=torch.float32)

        # Target & Weight
        item['target'] = torch.tensor(self.features['target_log_price'][index], dtype=torch.float32)
        item['sample_weight'] = torch.tensor(self.features['sample_weight'][index], dtype=torch.float32)

        return item
```

### **5. Dataloader Creation**

A function to create the `DataLoader` instances, including the custom collate function for batch tokenization.


```python
def create_dataloaders(train_features, val_features, config: Config):
    """Creates the PyTorch DataLoaders."""
    tokenizer_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=config.DEVICE)

    def custom_collate_fn(batch: list) -> dict:
        amenities_texts = [item.pop('amenities_text') for item in batch]
        collated_batch = {key: torch.stack([d[key] for d in batch]) for key in batch[0].keys()}
        tokenized = tokenizer_model.tokenizer(
            amenities_texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128
        )
        collated_batch['amenities_tokens'] = tokenized
        return collated_batch

    train_dataset = AirbnbPriceDataset(train_features)
    val_dataset = AirbnbPriceDataset(val_features)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    print(f"DataLoaders created. Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    return train_loader, val_loader
```

### **6. Model Architecture**

This is the `AdditiveAxisModel`, our core neural network. As detailed in `MODELING.md`, it's a multi-headed architecture where each "head" or sub-network is responsible for a distinct feature axis (Location, Size, etc.). The final price is the sum of contributions from each axis plus a global bias. This design makes the model's predictions inherently explainable.


```python
class AdditiveAxisModel(nn.Module):
    # --- This class is identical to the last working version ---
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
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): batch[k] = v.to(self.device)
        amenities_tokens = batch['amenities_tokens']
        for k, v in amenities_tokens.items(): amenities_tokens[k] = v.to(self.device)
        loc_input = torch.cat(
            [batch['loc_geo_position'],
            self.embed_neighbourhood(batch['loc_neighbourhood'])], dim=1
            )
        size_input = torch.cat(
            [
            self.embed_property_type(batch['size_property_type']),
            self.embed_room_type(batch['size_room_type']),
            self.embed_bathrooms_type(batch['size_bathrooms_type']),
            self.embed_beds(batch['size_beds']),
            self.embed_bedrooms(batch['size_bedrooms']),
            self.embed_bathrooms_numeric(batch['size_bathrooms_numeric']),
            batch['size_accommodates'].unsqueeze(1)
            ], dim=1
        )
        qual_input = torch.cat(
            [
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
            ], dim=1
        )
        amenities_embed = self.amenities_transformer(amenities_tokens)['sentence_embedding']
        p_loc = self.loc_subnet(loc_input)
        p_size = self.size_subnet(size_input)
        p_qual = self.qual_subnet(qual_input)
        p_amenities = self.amenities_subnet(amenities_embed)
        p_season = self.season_subnet(batch['season_cyclical'])

        return (
            self.global_bias
            + self.loc_subnet(loc_input) #p_loc
            + self.size_subnet(size_input) #p_size
            + self.qual_subnet(qual_input) #p_qual
            + self.amenities_subnet(amenities_embed) #p_amenities
            + self.season_subnet(batch['season_cyclical']) #p_season
        ).squeeze(-1)
```

### **7. Training Function**

This function orchestrates the training and validation loops for a given number of epochs.



```python
def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    config: Config
    ):
    """Main function to train and validate the model."""
    print("\n--- Starting Model Training ---")

    for epoch in range(config.N_EPOCHS):
        model.train()
        train_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.N_EPOCHS} [Train]")

        for i, batch in enumerate(train_iterator):
            predictions = model(batch)
            targets = batch['target'].to(config.DEVICE)
            weights = batch['sample_weight'].to(config.DEVICE)
            loss = (weights * (predictions - targets)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})

            if (i > 0) and (i % config.PRINT_EVERY_N_STEPS == 0):
                current_avg_loss = train_loss / (i + 1)
                print(f"  Epoch {epoch+1}, Step {i}/{len(train_loader)}, Avg Train Loss: {current_avg_loss:.4f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.N_EPOCHS} [Val]")
        with torch.no_grad():
            for batch in val_iterator:
                predictions = model(batch)
                targets = batch['target'].to(config.DEVICE)
                weights = batch['sample_weight'].to(config.DEVICE)
                loss = (weights * (predictions - targets)**2).mean()
                val_loss += loss.item()
                val_iterator.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config.N_EPOCHS} -> "
              f"Train W-MSLE: {avg_train_loss:.4f}, Val W-MSLE: {avg_val_loss:.4f} | "
              f"Train RW-MSLE: {np.sqrt(avg_train_loss):.4f}, Val RW-MSLE: {np.sqrt(avg_val_loss):.4f}")

    print("\n--- Training Complete ---")
    return model
```

### **8. Main Execution Function**

This single cell runs the entire pipeline from start to finish using the settings defined in the `Config` class.


```python
def main(config: Config):
    # 1. Load and split data
    train_df, val_df = load_and_split_data(config)

    # 2. Process features
    processor = FeatureProcessor()
    processor.fit(train_df)
    train_features = processor.transform(train_df)
    val_features = processor.transform(val_df)

    # 4. Create DataLoaders
    train_loader, val_loader = create_dataloaders(train_features, val_features, config)

    # 5. Initialize model and optimizer
    model = AdditiveAxisModel(processor, device=config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 6. Run training
    trained_model = train_model(model, train_loader, val_loader, optimizer, config)
```

#### **9. Final execution cell**

Requires two steps-- First, instantiate a Config object (`config`, say), changing any attributes from the default as needed. Next, simply run `main(config)`


```python
# Instantiate the configuration
config = Config()
print(f"Configuration loaded. Using device: {config.DEVICE}")

# Run the end-to-end training pipeline
main(config)
```

    Configuration loaded. Using device: cuda
    Loading dataset from: ./nyc_final_modeling_dataset.parquet
    Removed price outliers. New size: 81,643 records.
    Removed small strata. New size: 79,485 records.
    Split complete. Training: 63,588, Validation: 15,897
    
    --- Sample Record from Training Data ---




  <div id="df-aff11aac-678e-4f38-a9b3-00a72c714806" class="colab-df-container">
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
    <button class="colab-df-convert" onclick="convertToInteractive('df-aff11aac-678e-4f38-a9b3-00a72c714806')"
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
        document.querySelector('#df-aff11aac-678e-4f38-a9b3-00a72c714806 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-aff11aac-678e-4f38-a9b3-00a72c714806');
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


    <div id="df-532af253-bcf7-4616-ade6-efacf782fda0">
      <button class="colab-df-quickchart" onclick="quickchart('df-532af253-bcf7-4616-ade6-efacf782fda0')"
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
            document.querySelector('#df-532af253-bcf7-4616-ade6-efacf782fda0 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>




    modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]



    config_sentence_transformers.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



    README.md: 0.00B [00:00, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/52.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/743 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/133M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/366 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/125 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]


    DataLoaders created. Batches per epoch: Train=63, Val=16
    
    --- Starting Model Training ---


    Epoch 1/20 [Train]:  41%|████▏     | 26/63 [01:14<01:59,  3.23s/it, loss=0.6899]

      Epoch 1, Step 25/63, Avg Train Loss: 2.4650


    Epoch 1/20 [Train]:  81%|████████  | 51/63 [02:28<00:34,  2.84s/it, loss=0.0946]

      Epoch 1, Step 50/63, Avg Train Loss: 1.3688


    Epoch 1/20 [Train]: 100%|██████████| 63/63 [03:00<00:00,  2.87s/it, loss=0.0773]
    Epoch 1/20 [Val]: 100%|██████████| 16/16 [00:43<00:00,  2.74s/it, loss=0.0826]


    Epoch 1/20 -> Train W-MSLE: 1.1244, Val W-MSLE: 0.0737 | Train RW-MSLE: 1.0604, Val RW-MSLE: 0.2714


    Epoch 2/20 [Train]:  41%|████▏     | 26/63 [01:16<01:51,  3.00s/it, loss=0.0529]

      Epoch 2, Step 25/63, Avg Train Loss: 0.0631


    Epoch 2/20 [Train]:  81%|████████  | 51/63 [02:30<00:37,  3.09s/it, loss=0.0487]

      Epoch 2, Step 50/63, Avg Train Loss: 0.0568


    Epoch 2/20 [Train]: 100%|██████████| 63/63 [03:03<00:00,  2.92s/it, loss=0.0452]
    Epoch 2/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.75s/it, loss=0.0472]


    Epoch 2/20 -> Train W-MSLE: 0.0549, Val W-MSLE: 0.0441 | Train RW-MSLE: 0.2344, Val RW-MSLE: 0.2099


    Epoch 3/20 [Train]:  41%|████▏     | 26/63 [01:16<01:47,  2.90s/it, loss=0.0403]

      Epoch 3, Step 25/63, Avg Train Loss: 0.0418


    Epoch 3/20 [Train]:  81%|████████  | 51/63 [02:29<00:34,  2.90s/it, loss=0.0399]

      Epoch 3, Step 50/63, Avg Train Loss: 0.0406


    Epoch 3/20 [Train]: 100%|██████████| 63/63 [03:02<00:00,  2.90s/it, loss=0.0310]
    Epoch 3/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.76s/it, loss=0.0391]


    Epoch 3/20 -> Train W-MSLE: 0.0400, Val W-MSLE: 0.0375 | Train RW-MSLE: 0.1999, Val RW-MSLE: 0.1936


    Epoch 4/20 [Train]:  41%|████▏     | 26/63 [01:16<01:49,  2.97s/it, loss=0.0388]

      Epoch 4, Step 25/63, Avg Train Loss: 0.0363


    Epoch 4/20 [Train]:  81%|████████  | 51/63 [02:29<00:36,  3.00s/it, loss=0.0329]

      Epoch 4, Step 50/63, Avg Train Loss: 0.0357


    Epoch 4/20 [Train]: 100%|██████████| 63/63 [03:02<00:00,  2.89s/it, loss=0.0313]
    Epoch 4/20 [Val]: 100%|██████████| 16/16 [00:43<00:00,  2.74s/it, loss=0.0354]


    Epoch 4/20 -> Train W-MSLE: 0.0356, Val W-MSLE: 0.0345 | Train RW-MSLE: 0.1887, Val RW-MSLE: 0.1856


    Epoch 5/20 [Train]:  41%|████▏     | 26/63 [01:16<01:50,  2.97s/it, loss=0.0333]

      Epoch 5, Step 25/63, Avg Train Loss: 0.0337


    Epoch 5/20 [Train]:  81%|████████  | 51/63 [02:29<00:35,  2.95s/it, loss=0.0335]

      Epoch 5, Step 50/63, Avg Train Loss: 0.0334


    Epoch 5/20 [Train]: 100%|██████████| 63/63 [03:03<00:00,  2.91s/it, loss=0.0339]
    Epoch 5/20 [Val]: 100%|██████████| 16/16 [00:43<00:00,  2.72s/it, loss=0.0338]


    Epoch 5/20 -> Train W-MSLE: 0.0333, Val W-MSLE: 0.0329 | Train RW-MSLE: 0.1824, Val RW-MSLE: 0.1815


    Epoch 6/20 [Train]:  41%|████▏     | 26/63 [01:16<01:45,  2.86s/it, loss=0.0294]

      Epoch 6, Step 25/63, Avg Train Loss: 0.0319


    Epoch 6/20 [Train]:  81%|████████  | 51/63 [02:30<00:36,  3.08s/it, loss=0.0307]

      Epoch 6, Step 50/63, Avg Train Loss: 0.0317


    Epoch 6/20 [Train]: 100%|██████████| 63/63 [03:02<00:00,  2.90s/it, loss=0.0343]
    Epoch 6/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.76s/it, loss=0.0319]


    Epoch 6/20 -> Train W-MSLE: 0.0317, Val W-MSLE: 0.0311 | Train RW-MSLE: 0.1781, Val RW-MSLE: 0.1765


    Epoch 7/20 [Train]:  41%|████▏     | 26/63 [01:16<01:47,  2.91s/it, loss=0.0323]

      Epoch 7, Step 25/63, Avg Train Loss: 0.0302


    Epoch 7/20 [Train]:  81%|████████  | 51/63 [02:29<00:35,  2.95s/it, loss=0.0310]

      Epoch 7, Step 50/63, Avg Train Loss: 0.0304


    Epoch 7/20 [Train]: 100%|██████████| 63/63 [03:02<00:00,  2.89s/it, loss=0.0435]
    Epoch 7/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.76s/it, loss=0.0311]


    Epoch 7/20 -> Train W-MSLE: 0.0307, Val W-MSLE: 0.0310 | Train RW-MSLE: 0.1753, Val RW-MSLE: 0.1761


    Epoch 8/20 [Train]:  41%|████▏     | 26/63 [01:16<01:44,  2.82s/it, loss=0.0331]

      Epoch 8, Step 25/63, Avg Train Loss: 0.0301


    Epoch 8/20 [Train]:  81%|████████  | 51/63 [02:30<00:34,  2.91s/it, loss=0.0291]

      Epoch 8, Step 50/63, Avg Train Loss: 0.0301


    Epoch 8/20 [Train]: 100%|██████████| 63/63 [03:02<00:00,  2.90s/it, loss=0.0204]
    Epoch 8/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.76s/it, loss=0.0304]


    Epoch 8/20 -> Train W-MSLE: 0.0296, Val W-MSLE: 0.0298 | Train RW-MSLE: 0.1721, Val RW-MSLE: 0.1725


    Epoch 9/20 [Train]:  41%|████▏     | 26/63 [01:16<01:51,  3.01s/it, loss=0.0284]

      Epoch 9, Step 25/63, Avg Train Loss: 0.0293


    Epoch 9/20 [Train]:  81%|████████  | 51/63 [02:30<00:34,  2.86s/it, loss=0.0291]

      Epoch 9, Step 50/63, Avg Train Loss: 0.0292


    Epoch 9/20 [Train]: 100%|██████████| 63/63 [03:02<00:00,  2.90s/it, loss=0.0445]
    Epoch 9/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.75s/it, loss=0.0292]


    Epoch 9/20 -> Train W-MSLE: 0.0293, Val W-MSLE: 0.0289 | Train RW-MSLE: 0.1712, Val RW-MSLE: 0.1699


    Epoch 10/20 [Train]:  41%|████▏     | 26/63 [01:15<01:48,  2.94s/it, loss=0.0275]

      Epoch 10, Step 25/63, Avg Train Loss: 0.0287


    Epoch 10/20 [Train]:  81%|████████  | 51/63 [02:29<00:35,  2.97s/it, loss=0.0258]

      Epoch 10, Step 50/63, Avg Train Loss: 0.0286


    Epoch 10/20 [Train]: 100%|██████████| 63/63 [03:01<00:00,  2.88s/it, loss=0.0228]
    Epoch 10/20 [Val]: 100%|██████████| 16/16 [00:43<00:00,  2.75s/it, loss=0.0286]


    Epoch 10/20 -> Train W-MSLE: 0.0285, Val W-MSLE: 0.0285 | Train RW-MSLE: 0.1687, Val RW-MSLE: 0.1687


    Epoch 11/20 [Train]:  41%|████▏     | 26/63 [01:15<01:45,  2.84s/it, loss=0.0267]

      Epoch 11, Step 25/63, Avg Train Loss: 0.0284


    Epoch 11/20 [Train]:  81%|████████  | 51/63 [02:29<00:34,  2.92s/it, loss=0.0253]

      Epoch 11, Step 50/63, Avg Train Loss: 0.0283


    Epoch 11/20 [Train]: 100%|██████████| 63/63 [03:01<00:00,  2.89s/it, loss=0.0335]
    Epoch 11/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.76s/it, loss=0.0280]


    Epoch 11/20 -> Train W-MSLE: 0.0282, Val W-MSLE: 0.0280 | Train RW-MSLE: 0.1678, Val RW-MSLE: 0.1674


    Epoch 12/20 [Train]:  41%|████▏     | 26/63 [01:16<01:48,  2.94s/it, loss=0.0290]

      Epoch 12, Step 25/63, Avg Train Loss: 0.0276


    Epoch 12/20 [Train]:  81%|████████  | 51/63 [02:29<00:34,  2.86s/it, loss=0.0239]

      Epoch 12, Step 50/63, Avg Train Loss: 0.0275


    Epoch 12/20 [Train]: 100%|██████████| 63/63 [03:02<00:00,  2.89s/it, loss=0.0318]
    Epoch 12/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.75s/it, loss=0.0276]


    Epoch 12/20 -> Train W-MSLE: 0.0277, Val W-MSLE: 0.0275 | Train RW-MSLE: 0.1663, Val RW-MSLE: 0.1658


    Epoch 13/20 [Train]:  41%|████▏     | 26/63 [01:16<01:49,  2.96s/it, loss=0.0275]

      Epoch 13, Step 25/63, Avg Train Loss: 0.0272


    Epoch 13/20 [Train]:  81%|████████  | 51/63 [02:31<00:36,  3.06s/it, loss=0.0267]

      Epoch 13, Step 50/63, Avg Train Loss: 0.0274


    Epoch 13/20 [Train]: 100%|██████████| 63/63 [03:03<00:00,  2.91s/it, loss=0.0298]
    Epoch 13/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.79s/it, loss=0.0272]


    Epoch 13/20 -> Train W-MSLE: 0.0273, Val W-MSLE: 0.0270 | Train RW-MSLE: 0.1654, Val RW-MSLE: 0.1642


    Epoch 14/20 [Train]:  41%|████▏     | 26/63 [01:17<01:45,  2.86s/it, loss=0.0284]

      Epoch 14, Step 25/63, Avg Train Loss: 0.0267


    Epoch 14/20 [Train]:  81%|████████  | 51/63 [02:31<00:36,  3.08s/it, loss=0.0272]

      Epoch 14, Step 50/63, Avg Train Loss: 0.0271


    Epoch 14/20 [Train]: 100%|██████████| 63/63 [03:05<00:00,  2.95s/it, loss=0.0249]
    Epoch 14/20 [Val]: 100%|██████████| 16/16 [00:45<00:00,  2.87s/it, loss=0.0266]


    Epoch 14/20 -> Train W-MSLE: 0.0268, Val W-MSLE: 0.0269 | Train RW-MSLE: 0.1638, Val RW-MSLE: 0.1640


    Epoch 15/20 [Train]:  41%|████▏     | 26/63 [01:17<01:49,  2.97s/it, loss=0.0291]

      Epoch 15, Step 25/63, Avg Train Loss: 0.0267


    Epoch 15/20 [Train]:  81%|████████  | 51/63 [02:32<00:35,  2.92s/it, loss=0.0286]

      Epoch 15, Step 50/63, Avg Train Loss: 0.0268


    Epoch 15/20 [Train]: 100%|██████████| 63/63 [03:04<00:00,  2.94s/it, loss=0.0268]
    Epoch 15/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.77s/it, loss=0.0263]


    Epoch 15/20 -> Train W-MSLE: 0.0266, Val W-MSLE: 0.0264 | Train RW-MSLE: 0.1631, Val RW-MSLE: 0.1624


    Epoch 16/20 [Train]:  41%|████▏     | 26/63 [01:17<01:53,  3.06s/it, loss=0.0286]

      Epoch 16, Step 25/63, Avg Train Loss: 0.0264


    Epoch 16/20 [Train]:  81%|████████  | 51/63 [02:32<00:36,  3.05s/it, loss=0.0240]

      Epoch 16, Step 50/63, Avg Train Loss: 0.0264


    Epoch 16/20 [Train]: 100%|██████████| 63/63 [03:05<00:00,  2.94s/it, loss=0.0218]
    Epoch 16/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.78s/it, loss=0.0258]


    Epoch 16/20 -> Train W-MSLE: 0.0263, Val W-MSLE: 0.0264 | Train RW-MSLE: 0.1621, Val RW-MSLE: 0.1625


    Epoch 17/20 [Train]:  41%|████▏     | 26/63 [01:17<01:46,  2.89s/it, loss=0.0262]

      Epoch 17, Step 25/63, Avg Train Loss: 0.0262


    Epoch 17/20 [Train]:  81%|████████  | 51/63 [02:31<00:36,  3.00s/it, loss=0.0243]

      Epoch 17, Step 50/63, Avg Train Loss: 0.0263


    Epoch 17/20 [Train]: 100%|██████████| 63/63 [03:05<00:00,  2.95s/it, loss=0.0399]
    Epoch 17/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.76s/it, loss=0.0255]


    Epoch 17/20 -> Train W-MSLE: 0.0262, Val W-MSLE: 0.0261 | Train RW-MSLE: 0.1620, Val RW-MSLE: 0.1615


    Epoch 18/20 [Train]:  41%|████▏     | 26/63 [01:17<01:53,  3.05s/it, loss=0.0280]

      Epoch 18, Step 25/63, Avg Train Loss: 0.0257


    Epoch 18/20 [Train]:  81%|████████  | 51/63 [02:31<00:34,  2.88s/it, loss=0.0246]

      Epoch 18, Step 50/63, Avg Train Loss: 0.0260


    Epoch 18/20 [Train]: 100%|██████████| 63/63 [03:04<00:00,  2.93s/it, loss=0.0215]
    Epoch 18/20 [Val]: 100%|██████████| 16/16 [00:45<00:00,  2.83s/it, loss=0.0255]


    Epoch 18/20 -> Train W-MSLE: 0.0258, Val W-MSLE: 0.0261 | Train RW-MSLE: 0.1605, Val RW-MSLE: 0.1615


    Epoch 19/20 [Train]:  41%|████▏     | 26/63 [01:16<01:48,  2.92s/it, loss=0.0268]

      Epoch 19, Step 25/63, Avg Train Loss: 0.0253


    Epoch 19/20 [Train]:  81%|████████  | 51/63 [02:31<00:36,  3.04s/it, loss=0.0227]

      Epoch 19, Step 50/63, Avg Train Loss: 0.0256


    Epoch 19/20 [Train]: 100%|██████████| 63/63 [03:03<00:00,  2.91s/it, loss=0.0232]
    Epoch 19/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.79s/it, loss=0.0252]


    Epoch 19/20 -> Train W-MSLE: 0.0256, Val W-MSLE: 0.0255 | Train RW-MSLE: 0.1599, Val RW-MSLE: 0.1598


    Epoch 20/20 [Train]:  41%|████▏     | 26/63 [01:16<01:48,  2.93s/it, loss=0.0257]

      Epoch 20, Step 25/63, Avg Train Loss: 0.0251


    Epoch 20/20 [Train]:  81%|████████  | 51/63 [02:30<00:35,  2.93s/it, loss=0.0213]

      Epoch 20, Step 50/63, Avg Train Loss: 0.0253


    Epoch 20/20 [Train]: 100%|██████████| 63/63 [03:04<00:00,  2.92s/it, loss=0.0360]
    Epoch 20/20 [Val]: 100%|██████████| 16/16 [00:44<00:00,  2.78s/it, loss=0.0249]

    Epoch 20/20 -> Train W-MSLE: 0.0255, Val W-MSLE: 0.0253 | Train RW-MSLE: 0.1598, Val RW-MSLE: 0.1591
    
    --- Training Complete ---


    



```python

```


