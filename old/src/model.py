# model.py

"""
Defines the PyTorch model architecture and dataset class for the project.

This module contains:
- AdditiveAxisModel: A multi-headed neural network that predicts price deviation
  by summing the outputs of six specialized sub-networks.
- AirbnbPriceDataset: A PyTorch Dataset class that handles on-the-fly
  tokenization and collation of features for training and inference.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from collections import defaultdict

# Forward-referencing the type hint for FeatureProcessor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.src.data_processing import FeatureProcessor


class AirbnbPriceDataset(Dataset):
    """
    PyTorch Dataset to handle feature collation and on-the-fly tokenization.
    """
    def __init__(self, features: dict, tokenizer: AutoTokenizer):
        """
        Initializes the dataset.

        Args:
            features (dict): A dictionary of processed features, where each key
                maps to a list or numpy array of feature values.
            tokenizer (AutoTokenizer): The Hugging Face tokenizer instance used
                for processing text features.
        """
        self.features = features
        self.tokenizer = tokenizer
        self.n_samples = len(features['target_price'])

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves and formats a single sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing all features for one listing as
                  torch.Tensors, ready to be collated into a batch.
        """
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
            # All quality features are float, even booleans converted to 0/1
            item[f'qual_{k}'] = torch.tensor(v[index], dtype=torch.float32)

        # On-the-fly tokenization for text features
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


class AdditiveAxisModel(nn.Module):
    """
    A multi-axis neural network that predicts price deviation from a baseline.

    The model is composed of six sub-networks, each processing a different
    modality of the listing data (location, size, quality, etc.). The final
    output is the sum of the outputs from each sub-network, representing the
    predicted log-price deviation, which enables high interpretability.
    """
    def __init__(self, processor: 'FeatureProcessor', config: dict):
        """
        Initializes the model architecture, including all sub-networks.

        Args:
            processor (FeatureProcessor): The fitted FeatureProcessor instance,
                used to get vocabulary sizes for embedding layers.
            config (dict): The global configuration dictionary, used to define
                model architecture hyperparameters.
        """
        super().__init__()
        self.config = config
        self.device = self.config['DEVICE']

        # --- Embeddings for Categorical Features ---
        self.embed_property_type = nn.Embedding(len(processor.vocabs['property_type']), 8)
        self.embed_room_type = nn.Embedding(len(processor.vocabs['room_type']), 4)

        # --- Text Transformer (with last layer unfrozen for fine-tuning) ---
        self.text_transformer = SentenceTransformer(self.config['TEXT_MODEL_NAME'])
        for param in self.text_transformer.parameters():
            param.requires_grad = False
        # Unfreeze the final transformer layer for fine-tuning
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

        # --- Dynamically create sub-network "bodies" (feature extractors) ---
        text_embed_dim = self.text_transformer.get_sentence_embedding_dimension()
        self.loc_subnet_body = _create_mlp(self.config['GEO_EMBEDDING_DIM'], self.config['HIDDEN_LAYERS_LOCATION'])
        self.size_subnet_body = _create_mlp(16, self.config['HIDDEN_LAYERS_SIZE_CAPACITY'])
        self.qual_subnet_body = _create_mlp(8, self.config['HIDDEN_LAYERS_QUALITY'])
        self.amenities_subnet_body = _create_mlp(text_embed_dim, self.config['HIDDEN_LAYERS_AMENITIES'])
        self.desc_subnet_body = _create_mlp(text_embed_dim, self.config['HIDDEN_LAYERS_DESCRIPTION'])
        self.season_subnet_body = _create_mlp(2, self.config['HIDDEN_LAYERS_SEASONALITY'])

        # --- Dynamically create sub-network "heads" (output layers) ---
        self.loc_subnet_head = nn.Linear(self.config['HIDDEN_LAYERS_LOCATION'][-1], 1)
        self.size_subnet_head = nn.Linear(self.config['HIDDEN_LAYERS_SIZE_CAPACITY'][-1], 1)
        self.qual_subnet_head = nn.Linear(self.config['HIDDEN_LAYERS_QUALITY'][-1], 1)
        self.amenities_subnet_head = nn.Linear(self.config['HIDDEN_LAYERS_AMENITIES'][-1], 1)
        self.desc_subnet_head = nn.Linear(self.config['HIDDEN_LAYERS_DESCRIPTION'][-1], 1)
        self.season_subnet_head = nn.Linear(self.config['HIDDEN_LAYERS_SEASONALITY'][-1], 1)

        self.to(self.device)

    def forward_with_hidden_states(self, batch: dict) -> dict:
        """
        Performs a full forward pass, returning predictions, additive price
        contributions, and the final hidden state vectors for each sub-network.

        Args:
            batch (dict): A batch of data from the AirbnbPriceDataset.

        Returns:
            dict: A dictionary containing the final predicted log deviation,
                  the contribution of each axis ('p_*'), and the hidden state
                  vector of each axis ('h_*').
        """
        # --- Prepare Inputs for each Axis ---
        loc_input = batch['loc_geo_position']
        size_input = torch.cat([
            self.embed_property_type(batch['size_property_type']),
            self.embed_room_type(batch['size_room_type']),
            batch['size_accommodates'].unsqueeze(1),
            batch['size_bedrooms'].unsqueeze(1),
            batch['size_beds'].unsqueeze(1),
            batch['size_bathrooms'].unsqueeze(1)
        ], dim=1)
        qual_cols = [
            "review_scores_rating", "review_scores_cleanliness", "review_scores_checkin",
            "review_scores_communication", "review_scores_location", "review_scores_value",
            "total_reviews", "host_is_superhost"
        ]
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

    def forward(self, batch: dict) -> torch.Tensor:
        """
        The standard forward pass for training, returning only the final
        prediction tensor required for loss calculation.

        Args:
            batch (dict): A batch of data from the AirbnbPriceDataset.

        Returns:
            torch.Tensor: The predicted log deviation tensor.
        """
        return self.forward_with_hidden_states(batch)['predicted_log_deviation']
    
    def count_parameters(self):
        """
        Counts and prints the number of trainable and frozen parameters in the model.

        This utility method provides a detailed breakdown of parameters for each major
        component of the model, such as the text transformer, embeddings, and the
        various sub-networks.
        """
        param_counts = {
            'trainable': defaultdict(int),
            'frozen': defaultdict(int)
        }
        
        # Helper to categorize parameters
        def get_component(name):
            if name.startswith('text_transformer'):
                return 'Text Transformer'
            elif name.startswith('embed_'):
                return 'Categorical Embeddings'
            elif name.startswith(('loc_', 'size_', 'qual_', 'amenities_', 'desc_', 'season_')):
                return 'MLP Sub-Networks'
            else:
                return 'Other'

        # Iterate over all named parameters
        for name, param in self.named_parameters():
            component = get_component(name)
            if param.requires_grad:
                param_counts['trainable'][component] += param.numel()
            else:
                param_counts['frozen'][component] += param.numel()

        print("-" * 60)
        print(f"{'Model Parameter Analysis':^60}")
        print("-" * 60)
        
        total_trainable = 0
        print("\n--- Trainable Parameters ---")
        for component, count in param_counts['trainable'].items():
            print(f"  - {component:<25}: {count:12,d}")
            total_trainable += count

        total_frozen = 0
        print("\n--- Frozen Parameters ---")
        for component, count in param_counts['frozen'].items():
            print(f"  - {component:<25}: {count:12,d}")
            total_frozen += count

        print("\n" + "=" * 60)
        print(f"{'Total Trainable Parameters:':<30} {total_trainable:15,d}")
        print(f"{'Total Frozen Parameters:':<30} {total_frozen:15,d}")
        print(f"{'Total Parameters:':<30} {(total_trainable + total_frozen):15,d}")
        print("=" * 60)