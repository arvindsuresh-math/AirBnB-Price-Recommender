<a href="https://colab.research.google.com/github/arvindsuresh-math/Fall-2025-Team-Big-Data/blob/main/02_modeling_prototype.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### **Airbnb Price Prediction: Full Prototyping Pipeline for Google Colab**

This notebook contains the end-to-end code for data processing, model building, and training, optimized for a Google Colab GPU environment.


```python
from google.colab import userdata
from huggingface_hub import login
import os

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)
    print("Hugging Face login successful.")
except userdata.SecretNotFoundError:
    print("Secret 'HF_TOKEN' not found.")
    print("Please add your Hugging Face token to Colab Secrets.")
except Exception as e:
    print(f"An error occurred during Hugging Face login: {e}")
```

    Hugging Face login successful.



```python
!pip install pandas
!pip install pyarrow
!pip install sentence-transformers
!pip install scikit-learn
!pip install torch
!pip install tqdm
```

    Requirement already satisfied: pandas in /usr/local/lib/python3.12/dist-packages (2.2.2)
    Requirement already satisfied: numpy>=1.26.0 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.0.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas) (2025.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    Requirement already satisfied: pyarrow in /usr/local/lib/python3.12/dist-packages (18.1.0)
    Requirement already satisfied: sentence-transformers in /usr/local/lib/python3.12/dist-packages (5.1.1)
    Requirement already satisfied: transformers<5.0.0,>=4.41.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.56.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.67.1)
    Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (2.8.0+cu126)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.6.1)
    Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.16.2)
    Requirement already satisfied: huggingface-hub>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (0.35.3)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (11.3.0)
    Requirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.15.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.19.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2025.3.0)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (25.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0.3)
    Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2.32.4)
    Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.1.10)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (1.13.3)
    Requirement already satisfied: networkx in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.1.6)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.77)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.77)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.80)
    Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (9.10.2.21)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (11.3.0.4)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (10.3.7.77)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (11.7.1.2)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.5.4.2)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.27.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (2.27.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.77)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.85)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (1.11.1.6)
    Requirement already satisfied: triton==3.4.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.4.0)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2024.11.6)
    Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.22.1)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.6.2)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->sentence-transformers) (1.5.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn->sentence-transformers) (3.6.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.11.0->sentence-transformers) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.11.0->sentence-transformers) (3.0.3)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.4.3)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2025.8.3)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)
    Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (2.0.2)
    Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.16.2)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.5.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (2.8.0+cu126)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch) (3.19.1)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch) (4.15.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.13.3)
    Requirement already satisfied: networkx in /usr/local/lib/python3.12/dist-packages (from torch) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch) (3.1.6)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.12/dist-packages (from torch) (2025.3.0)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.80)
    Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch) (9.10.2.21)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch) (11.3.0.4)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch) (10.3.7.77)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch) (11.7.1.2)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch) (12.5.4.2)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.27.3 in /usr/local/lib/python3.12/dist-packages (from torch) (2.27.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.85)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch) (1.11.1.6)
    Requirement already satisfied: triton==3.4.0 in /usr/local/lib/python3.12/dist-packages (from torch) (3.4.0)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch) (3.0.3)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (4.67.1)


#### **2. Configuration and Data Upload**

Run this cell to set up the main configuration. **After running it, upload your `.parquet` file** using the "Files" tab on the left-hand side of the Colab interface.


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

# --- Main Configuration ---
CITY = "nyc"
DATASET_FILENAME = f"{CITY}_final_modeling_dataset.parquet"
DATASET_PATH = f"./{DATASET_FILENAME}" # Assumes file is uploaded to the root of the Colab runtime

VAL_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
N_EPOCHS = 5

# --- Device Selection for Colab GPU ---
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"GPU found: {torch.cuda.get_device_name(0)}. Using CUDA for acceleration.")
else:
    DEVICE = "cpu"
    print("GPU not available. Using CPU.")

# Configure pandas display
pd.options.display.max_columns = 100
```

    GPU found: Tesla T4. Using CUDA for acceleration.


#### **3. Data Loading and Stratified Split**

This step loads the dataset and performs the stratified split to create training and validation sets.


```python
# --- Load Data ---
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"'{DATASET_FILENAME}' not found. Please upload the file.")

print(f"Loading final dataset from: {DATASET_PATH}")
df = pd.read_parquet(DATASET_PATH)
print("Dataset loaded successfully.")

# --- Stratified Train/Validation Split ---
stratify_col = df['neighbourhood_cleansed'].astype(str) + '_' + df['month'].astype(str)
strata_counts = stratify_col.value_counts()
valid_strata = strata_counts[strata_counts >= 2].index
df_filtered = df[stratify_col.isin(valid_strata)].copy()

print(f"\nRemoved {len(df) - len(df_filtered):,} records belonging to strata with only 1 member.")

train_indices, val_indices = train_test_split(
    df_filtered.index,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_filtered['neighbourhood_cleansed'].astype(str) + '_' + df_filtered['month'].astype(str)
)

train_df = df_filtered.loc[train_indices].copy().reset_index(drop=True)
val_df = df_filtered.loc[val_indices].copy().reset_index(drop=True)

print(f"Training records: {len(train_df):,}, Validation records: {len(val_df):,}")
```

    Loading final dataset from: ./nyc_final_modeling_dataset.parquet
    Dataset loaded successfully.
    
    Removed 230 records belonging to strata with only 1 member.
    Training records: 66,390, Validation records: 16,598


#### **4. FeatureProcessor Class**

The class for handling all feature transformations.


```python
class FeatureProcessor:
    def __init__(self, embedding_dim_geo: int = 32):
        self.vocabs, self.scalers = {}, {}
        self.embedding_dim_geo = embedding_dim_geo
        self.categorical_cols = ["neighbourhood_cleansed", "property_type", "room_type", "bathrooms_type", "bedrooms", "beds", "bathrooms_numeric"]
        self.numerical_cols = ["accommodates", "review_scores_rating", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", "review_scores_location", "review_scores_value", "host_response_rate", "host_acceptance_rate"]
        self.log_transform_cols = ["number_of_reviews_ltm"]
        self.boolean_cols = ["host_is_superhost", "host_identity_verified", "instant_bookable"]

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

        location = {"geo_position": np.hstack([np.stack(lat_enc), np.stack(lon_enc)]), "neighbourhood": df["neighbourhood_cleansed"].apply(lambda x: self.vocabs["neighbourhood_cleansed"].get(x, 0)).values}
        size = {col: df[col].apply(lambda x: self.vocabs[col].get(x, 0) if pd.notna(x) else 0).values for col in ["property_type", "room_type", "bathrooms_type", "bedrooms", "beds", "bathrooms_numeric"]}
        size["accommodates"] = ((df["accommodates"] - self.scalers["accommodates"]["mean"]) / self.scalers["accommodates"]["std"]).values
        quality = {col: ((df[col] - self.scalers[col]["mean"]) / self.scalers[col]["std"]).values for col in self.numerical_cols if col != "accommodates"}
        quality["number_of_reviews_ltm"] = ((np.log1p(df["number_of_reviews_ltm"]) - self.scalers["number_of_reviews_ltm"]["mean"]) / self.scalers["number_of_reviews_ltm"]["std"]).values
        for col in self.boolean_cols: quality[col] = df[col].astype(float).values
        seasonality = {"cyclical": np.vstack([np.sin(2 * np.pi * df["month"] / 12), np.cos(2 * np.pi * df["month"] / 12)]).T}

        return {"location": location, "size_capacity": size, "quality": quality, "amenities": {"text": df["amenities"].tolist()}, "seasonality": seasonality, "target_price": df["target_price"].values, "sample_weight": df["estimated_occupancy_rate"].values}

# Instantiate, fit, and transform
processor = FeatureProcessor()
processor.fit(train_df)
train_features = processor.transform(train_df)
val_features = processor.transform(val_df)
print("\nFeature processing complete.")
```

    
    Feature processing complete.


#### **5. PyTorch Dataset and DataLoader**

This section defines the data loading and batching pipeline.


```python
class AirbnbPriceDataset(Dataset):
    def __init__(self, features: dict):
        self.features = features
        self.n_samples = len(features['target_price'])

    def __len__(self): return self.n_samples

    def __getitem__(self, index: int) -> dict:
        item = {}
        item['loc_geo_position'] = torch.tensor(self.features['location']['geo_position'][index], dtype=torch.float32)
        item['loc_neighbourhood'] = torch.tensor(self.features['location']['neighbourhood'][index], dtype=torch.long)
        for k, v in self.features['size_capacity'].items(): item[f'size_{k}'] = torch.tensor(v[index], dtype=torch.float32 if k == 'accommodates' else torch.long)
        for k, v in self.features['quality'].items(): item[f'qual_{k}'] = torch.tensor(v[index], dtype=torch.float32)
        item['amenities_text'] = self.features['amenities']['text'][index]
        item['season_cyclical'] = torch.tensor(self.features['seasonality']['cyclical'][index], dtype=torch.float32)
        item['target_price'] = torch.tensor(self.features['target_price'][index], dtype=torch.float32)
        item['sample_weight'] = torch.tensor(self.features['sample_weight'][index], dtype=torch.float32)
        return item

# The tokenizer is instantiated to use the GPU (DEVICE)
tokenizer_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=DEVICE)

def custom_collate_fn(batch: list) -> dict:
    amenities_texts = [item.pop('amenities_text') for item in batch]
    collated_batch = {key: torch.stack([d[key] for d in batch]) for key in batch[0].keys()}
    tokenized = tokenizer_model.tokenizer(amenities_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
    collated_batch['amenities_tokens'] = tokenized
    return collated_batch

train_dataset = AirbnbPriceDataset(train_features)
val_dataset = AirbnbPriceDataset(val_features)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

print(f"DataLoaders created. Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
```


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


    DataLoaders created. Batches per epoch: Train=260, Val=65


#### **6. Additive Model Architecture**

This is the `nn.Module` class for our 5-axis additive model. It is configured to run on the selected `DEVICE`.


```python
class AdditiveAxisModel(nn.Module):
    def __init__(self, processor: FeatureProcessor, device: str):
        super().__init__()
        self.vocabs, self.device = processor.vocabs, device

        # --- Embeddings ---
        self.embed_neighbourhood = nn.Embedding(len(self.vocabs['neighbourhood_cleansed']), 16)
        self.embed_property_type = nn.Embedding(len(self.vocabs['property_type']), 8)
        self.embed_room_type = nn.Embedding(len(self.vocabs['room_type']), 4)
        self.embed_bathrooms_type = nn.Embedding(len(self.vocabs['bathrooms_type']), 2)
        self.embed_bedrooms = nn.Embedding(len(self.vocabs['bedrooms']), 4)
        self.embed_beds = nn.Embedding(len(self.vocabs['beds']), 4)
        self.embed_bathrooms_numeric = nn.Embedding(len(self.vocabs['bathrooms_numeric']), 4)

        # --- Amenities Transformer (placed on the correct device) ---
        self.amenities_transformer = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        for param in self.amenities_transformer.parameters(): param.requires_grad = False

        # --- Sub-Networks ---
        self.loc_subnet = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 1))
        self.size_subnet = nn.Sequential(nn.Linear(27, 32), nn.ReLU(), nn.Linear(32, 1))
        self.qual_subnet = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 1))
        self.amenities_subnet = nn.Linear(384, 1)
        self.season_subnet = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self.global_bias = nn.Parameter(torch.randn(1))
        self.to(self.device)

    def forward(self, batch: dict) -> torch.Tensor:
        # --- FIX IS HERE ---
        # The original loop does not modify the batch in place correctly.
        # The most robust way is to handle the device placement explicitly for
        # the amenities tokens just before they are used.

        # 1. Move all standard tensors to the model's device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        # 2. Specifically move the amenities tokens dictionary to the device
        amenities_tokens = batch['amenities_tokens']
        for k, v in amenities_tokens.items():
            amenities_tokens[k] = v.to(self.device)

        # --- Process Axes ---
        loc_input = torch.cat([batch['loc_geo_position'], self.embed_neighbourhood(batch['loc_neighbourhood'])], dim=1)
        size_input = torch.cat([self.embed_property_type(batch['size_property_type']), self.embed_room_type(batch['size_room_type']), self.embed_bathrooms_type(batch['size_bathrooms_type']), self.embed_beds(batch['size_beds']), self.embed_bedrooms(batch['size_bedrooms']), self.embed_bathrooms_numeric(batch['size_bathrooms_numeric']), batch['size_accommodates'].unsqueeze(1)], dim=1)
        qual_input = torch.cat([batch['qual_review_scores_rating'].unsqueeze(1), batch['qual_review_scores_cleanliness'].unsqueeze(1), batch['qual_review_scores_checkin'].unsqueeze(1), batch['qual_review_scores_communication'].unsqueeze(1), batch['qual_review_scores_location'].unsqueeze(1), batch['qual_review_scores_value'].unsqueeze(1), batch['qual_host_response_rate'].unsqueeze(1), batch['qual_host_acceptance_rate'].unsqueeze(1), batch['qual_number_of_reviews_ltm'].unsqueeze(1), batch['qual_host_is_superhost'].unsqueeze(1), batch['qual_host_identity_verified'].unsqueeze(1), batch['qual_instant_bookable'].unsqueeze(1)], dim=1)

        # Now, amenities_tokens and the transformer model are both on the same GPU device
        amenities_embed = self.amenities_transformer(amenities_tokens)['sentence_embedding']

        # --- Get Contributions ---
        p_loc = self.loc_subnet(loc_input)
        p_size = self.size_subnet(size_input)
        p_qual = self.qual_subnet(qual_input)
        p_amenities = self.amenities_subnet(amenities_embed)
        p_season = self.season_subnet(batch['season_cyclical'])

        return (self.global_bias + p_loc + p_size + p_qual + p_amenities + p_season).squeeze(-1)
```

#### **7. Training Loop**

The final step is to instantiate the model and run the training process.


```python
model = AdditiveAxisModel(processor, device=DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Model Training ---")

for epoch in range(N_EPOCHS):
    model.train()
    train_loss = 0.0
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]")

    for batch in train_iterator:
        predictions = model(batch)
        targets = batch['target_price'].to(DEVICE)
        weights = batch['sample_weight'].to(DEVICE)
        loss = (weights * (predictions - targets)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iterator.set_postfix({'loss': f"{loss.item():.2f}"})

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Val]")
    with torch.no_grad():
        for batch in val_iterator:
            predictions = model(batch)
            targets = batch['target_price'].to(DEVICE)
            weights = batch['sample_weight'].to(DEVICE)
            loss = (weights * (predictions - targets)**2).mean()
            val_loss += loss.item()
            val_iterator.set_postfix({'loss': f"{loss.item():.2f}"})

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{N_EPOCHS} -> Train W-MSE: {avg_train_loss:.2f}, Val W-MSE: {avg_val_loss:.2f}")

print("\n--- Training Complete ---")
```

    
    --- Starting Model Training ---


    Epoch 1/5 [Train]: 100%|██████████| 260/260 [02:53<00:00,  1.50it/s, loss=8975.19]
    Epoch 1/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.53it/s, loss=10033.57]


    Epoch 1/5 -> Train W-MSE: 159919.99, Val W-MSE: 312283.91


    Epoch 2/5 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=2356.05]
    Epoch 2/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=6600.26]


    Epoch 2/5 -> Train W-MSE: 150499.25, Val W-MSE: 307818.81


    Epoch 3/5 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=2938.99]
    Epoch 3/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=5455.60]


    Epoch 3/5 -> Train W-MSE: 148390.47, Val W-MSE: 306210.13


    Epoch 4/5 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=6653.52]
    Epoch 4/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.53it/s, loss=4836.97]


    Epoch 4/5 -> Train W-MSE: 147376.51, Val W-MSE: 305171.51


    Epoch 5/5 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=9511.75]
    Epoch 5/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.53it/s, loss=4375.31]

    Epoch 5/5 -> Train W-MSE: 146668.50, Val W-MSE: 304344.00
    
    --- Training Complete ---


    



```python
LEARNING_RATE = 1e-2
model_2 = AdditiveAxisModel(processor, device=DEVICE)
optimizer = optim.AdamW(model_2.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Model Training ---")

for epoch in range(N_EPOCHS):
    model_2.train()
    train_loss = 0.0
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]")

    for batch in train_iterator:
        predictions = model_2(batch)
        targets = batch['target_price'].to(DEVICE)
        weights = batch['sample_weight'].to(DEVICE)
        loss = (weights * (predictions - targets)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iterator.set_postfix({'loss': f"{loss.item():.2f}"})

    avg_train_loss = train_loss / len(train_loader)

    model_2.eval()
    val_loss = 0.0
    val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Val]")
    with torch.no_grad():
        for batch in val_iterator:
            predictions = model_2(batch)
            targets = batch['target_price'].to(DEVICE)
            weights = batch['sample_weight'].to(DEVICE)
            loss = (weights * (predictions - targets)**2).mean()
            val_loss += loss.item()
            val_iterator.set_postfix({'loss': f"{loss.item():.2f}"})

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{N_EPOCHS} -> Train W-MSE: {avg_train_loss:.2f}, Val W-MSE: {avg_val_loss:.2f}")

print("\n--- Training Complete ---")
```

    
    --- Starting Model Training ---


    Epoch 1/5 [Train]: 100%|██████████| 260/260 [02:56<00:00,  1.47it/s, loss=3667.76]
    Epoch 1/5 [Val]: 100%|██████████| 65/65 [00:43<00:00,  1.51it/s, loss=3513.71]


    Epoch 1/5 -> Train W-MSE: 149073.49, Val W-MSE: 302355.02


    Epoch 2/5 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=4697.77]
    Epoch 2/5 [Val]: 100%|██████████| 65/65 [00:43<00:00,  1.51it/s, loss=5099.27]


    Epoch 2/5 -> Train W-MSE: 144413.44, Val W-MSE: 299192.04


    Epoch 3/5 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=2625.98]
    Epoch 3/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.51it/s, loss=4066.63]


    Epoch 3/5 -> Train W-MSE: 142926.05, Val W-MSE: 293573.97


    Epoch 4/5 [Train]: 100%|██████████| 260/260 [02:56<00:00,  1.48it/s, loss=8796.17]
    Epoch 4/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=5252.26]


    Epoch 4/5 -> Train W-MSE: 140520.60, Val W-MSE: 284370.72


    Epoch 5/5 [Train]: 100%|██████████| 260/260 [02:56<00:00,  1.47it/s, loss=4127.50]
    Epoch 5/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=7762.66]

    Epoch 5/5 -> Train W-MSE: 138064.06, Val W-MSE: 276477.98
    
    --- Training Complete ---


    



```python
LEARNING_RATE = 1e-1
model_3 = AdditiveAxisModel(processor, device=DEVICE)
optimizer = optim.AdamW(model_3.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Model Training ---")

for epoch in range(N_EPOCHS):
    model_3.train()
    train_loss = 0.0
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]")

    for batch in train_iterator:
        predictions = model_3(batch)
        targets = batch['target_price'].to(DEVICE)
        weights = batch['sample_weight'].to(DEVICE)
        loss = (weights * (predictions - targets)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iterator.set_postfix({'loss': f"{loss.item():.2f}"})

    avg_train_loss = train_loss / len(train_loader)

    model_3.eval()
    val_loss = 0.0
    val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Val]")
    with torch.no_grad():
        for batch in val_iterator:
            predictions = model_3(batch)
            targets = batch['target_price'].to(DEVICE)
            weights = batch['sample_weight'].to(DEVICE)
            loss = (weights * (predictions - targets)**2).mean()
            val_loss += loss.item()
            val_iterator.set_postfix({'loss': f"{loss.item():.2f}"})

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{N_EPOCHS} -> Train W-MSE: {avg_train_loss:.2f}, Val W-MSE: {avg_val_loss:.2f}")

print("\n--- Training Complete ---")
```

    
    --- Starting Model Training ---


    Epoch 1/5 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=3375.86]
    Epoch 1/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.51it/s, loss=4248.33]


    Epoch 1/5 -> Train W-MSE: 144509.35, Val W-MSE: 284081.39


    Epoch 2/5 [Train]: 100%|██████████| 260/260 [02:56<00:00,  1.48it/s, loss=16561.71]
    Epoch 2/5 [Val]: 100%|██████████| 65/65 [00:43<00:00,  1.51it/s, loss=3174.48]


    Epoch 2/5 -> Train W-MSE: 138679.32, Val W-MSE: 286510.93


    Epoch 3/5 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=6883.02]
    Epoch 3/5 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.51it/s, loss=5755.81]


    Epoch 3/5 -> Train W-MSE: 138890.16, Val W-MSE: 255885.86


    Epoch 4/5 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=14498.67]
    Epoch 4/5 [Val]: 100%|██████████| 65/65 [00:43<00:00,  1.51it/s, loss=3372.25]


    Epoch 4/5 -> Train W-MSE: 140166.90, Val W-MSE: 274841.36


    Epoch 5/5 [Train]: 100%|██████████| 260/260 [02:56<00:00,  1.48it/s, loss=7706.68]
    Epoch 5/5 [Val]: 100%|██████████| 65/65 [00:43<00:00,  1.49it/s, loss=12699.57]

    Epoch 5/5 -> Train W-MSE: 136655.15, Val W-MSE: 261030.59
    
    --- Training Complete ---


    



```python
LEARNING_RATE = 1e-2
N_EPOCHS = 20

model_2 = AdditiveAxisModel(processor, device=DEVICE)
optimizer = optim.AdamW(model_2.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Model Training ---")

for epoch in range(N_EPOCHS):
    model_2.train()
    train_loss = 0.0
    train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]")

    for batch in train_iterator:
        predictions = model_2(batch)
        targets = batch['target_price'].to(DEVICE)
        weights = batch['sample_weight'].to(DEVICE)
        loss = (weights * (predictions - targets)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iterator.set_postfix({'loss': f"{loss.item():.2f}"})

    avg_train_loss = train_loss / len(train_loader)

    model_2.eval()
    val_loss = 0.0
    val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Val]")
    with torch.no_grad():
        for batch in val_iterator:
            predictions = model_2(batch)
            targets = batch['target_price'].to(DEVICE)
            weights = batch['sample_weight'].to(DEVICE)
            loss = (weights * (predictions - targets)**2).mean()
            val_loss += loss.item()
            val_iterator.set_postfix({'loss': f"{loss.item():.2f}"})

    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{N_EPOCHS} -> Train W-MSE: {avg_train_loss:.2f}, Val W-MSE: {avg_val_loss:.2f}")

print("\n--- Training Complete ---")
```

    
    --- Starting Model Training ---


    Epoch 1/20 [Train]: 100%|██████████| 260/260 [02:56<00:00,  1.47it/s, loss=9511.12]
    Epoch 1/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=3285.27]


    Epoch 1/20 -> Train W-MSE: 149641.67, Val W-MSE: 303257.87


    Epoch 2/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=1304.93]
    Epoch 2/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=3431.67]


    Epoch 2/20 -> Train W-MSE: 144850.40, Val W-MSE: 300157.12


    Epoch 3/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.49it/s, loss=3485.45]
    Epoch 3/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=5862.91]


    Epoch 3/20 -> Train W-MSE: 142795.82, Val W-MSE: 292089.54


    Epoch 4/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=4185.72]
    Epoch 4/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=5434.29]


    Epoch 4/20 -> Train W-MSE: 140720.89, Val W-MSE: 284938.99


    Epoch 5/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=2364.32]
    Epoch 5/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=6805.74]


    Epoch 5/20 -> Train W-MSE: 138518.45, Val W-MSE: 278101.74


    Epoch 6/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=9258.39]
    Epoch 6/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.53it/s, loss=7250.23]


    Epoch 6/20 -> Train W-MSE: 136662.40, Val W-MSE: 274498.96


    Epoch 7/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=9548.23]
    Epoch 7/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=5946.40]


    Epoch 7/20 -> Train W-MSE: 135418.54, Val W-MSE: 270884.79


    Epoch 8/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=3333.93]
    Epoch 8/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=5041.03]


    Epoch 8/20 -> Train W-MSE: 134036.44, Val W-MSE: 266060.51


    Epoch 9/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=10855.36]
    Epoch 9/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.53it/s, loss=4359.11]


    Epoch 9/20 -> Train W-MSE: 133522.56, Val W-MSE: 264795.00


    Epoch 10/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=6413.00]
    Epoch 10/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=5382.54]


    Epoch 10/20 -> Train W-MSE: 133345.11, Val W-MSE: 263341.13


    Epoch 11/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=16941.41]
    Epoch 11/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=5502.39]


    Epoch 11/20 -> Train W-MSE: 132845.65, Val W-MSE: 260813.73


    Epoch 12/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=10921.37]
    Epoch 12/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=5184.70]


    Epoch 12/20 -> Train W-MSE: 132152.43, Val W-MSE: 257973.29


    Epoch 13/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.49it/s, loss=2035.43]
    Epoch 13/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=4391.58]


    Epoch 13/20 -> Train W-MSE: 131901.95, Val W-MSE: 258244.27


    Epoch 14/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.49it/s, loss=3334.38]
    Epoch 14/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=4204.29]


    Epoch 14/20 -> Train W-MSE: 131755.72, Val W-MSE: 257943.03


    Epoch 15/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=3593.28]
    Epoch 15/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.53it/s, loss=4227.32]


    Epoch 15/20 -> Train W-MSE: 131561.77, Val W-MSE: 258579.46


    Epoch 16/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=31538.52]
    Epoch 16/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=4169.39]


    Epoch 16/20 -> Train W-MSE: 131262.74, Val W-MSE: 258462.19


    Epoch 17/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=1973.08]
    Epoch 17/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=4070.57]


    Epoch 17/20 -> Train W-MSE: 131426.46, Val W-MSE: 258261.09


    Epoch 18/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=18206.05]
    Epoch 18/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.52it/s, loss=4426.19]


    Epoch 18/20 -> Train W-MSE: 131228.84, Val W-MSE: 257474.39


    Epoch 19/20 [Train]: 100%|██████████| 260/260 [02:54<00:00,  1.49it/s, loss=2040.05]
    Epoch 19/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.54it/s, loss=4013.96]


    Epoch 19/20 -> Train W-MSE: 130706.96, Val W-MSE: 258057.55


    Epoch 20/20 [Train]: 100%|██████████| 260/260 [02:55<00:00,  1.48it/s, loss=3603.62]
    Epoch 20/20 [Val]: 100%|██████████| 65/65 [00:42<00:00,  1.53it/s, loss=4172.12]

    Epoch 20/20 -> Train W-MSE: 130788.70, Val W-MSE: 257104.22
    
    --- Training Complete ---


    



```python

```
