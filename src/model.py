"""
Defines the PyTorch model architectures and the Dataset class.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

class AirbnbPriceDataset(Dataset):
    """PyTorch Dataset to handle feature collation and on-the-fly tokenization."""
    def __init__(self, features: dict, tokenizer: AutoTokenizer):
        self.features = features
        self.tokenizer = tokenizer
        self.n_samples = len(features['target_price'])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> dict:
        item = {
            'loc_geo_position': torch.tensor(self.features['location']['geo_position'][index]),
            'season_cyclical': torch.tensor(self.features['seasonality']['cyclical'][index]),
            'target_price': torch.tensor(self.features['target_price'][index]),
            'target_log_deviation': torch.tensor(self.features['target_log_deviation'][index]),
            'neighborhood_log_mean': torch.tensor(self.features['neighborhood_log_mean'][index]),
        }
        for k, v in self.features['size_capacity'].items():
            dtype = torch.long if k in ['property_type', 'room_type'] else torch.float32
            item[f'size_{k}'] = torch.tensor(v[index], dtype=dtype)
        for k, v in self.features['quality'].items():
            item[f'qual_{k}'] = torch.tensor(v[index], dtype=torch.float32)

        # This part is correct, the tokenizer creates the extra dimension here
        item['amenities_tokens'] = self.tokenizer(
            self.features['amenities_text'][index], padding='max_length', truncation=True,
            max_length=128, return_tensors="pt"
        )
        item['description_tokens'] = self.tokenizer(
            self.features['description_text'][index], padding='max_length', truncation=True,
            max_length=256, return_tensors="pt"
        )
        return item

class BaselineModel(nn.Module):
    """A regularized, fully-connected network that serves as a performance baseline."""
    def __init__(self, processor, config: dict):
        super().__init__()
        self.config = config
        self.embed_property_type = nn.Embedding(len(processor.vocabs['property_type']), 8)
        self.embed_room_type = nn.Embedding(len(processor.vocabs['room_type']), 4)
        self.text_transformer = SentenceTransformer(config['TEXT_MODEL_NAME'])
        for param in self.text_transformer.parameters():
            param.requires_grad = False

        text_embed_dim = self.text_transformer.get_sentence_embedding_dimension()
        total_input_dim = (
            config['GEO_EMBEDDING_DIM'] + 8 + 4 + 4 + 8 + 2 + text_embed_dim * 2
        )

        self.main_mlp = nn.Sequential(
            nn.Linear(total_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(p=config['DROPOUT_RATE']),
            nn.Linear(128, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(p=config['DROPOUT_RATE']),
            nn.Linear(32, 1)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        # --- CORRECTED CODE: Squeeze the extra dimension from the tokenizers ---
        amenities_tokens = {key: val.squeeze(1) for key, val in batch['amenities_tokens'].items()}
        desc_tokens = {key: val.squeeze(1) for key, val in batch['description_tokens'].items()}
        # --- END OF CORRECTION ---

        with torch.no_grad():
            amenities_embed = self.text_transformer(amenities_tokens)['sentence_embedding']
            desc_embed = self.text_transformer(desc_tokens)['sentence_embedding']

        size_input = torch.cat([
            self.embed_property_type(batch['size_property_type']), self.embed_room_type(batch['size_room_type']),
            batch['size_accommodates'].unsqueeze(1), batch['size_bedrooms'].unsqueeze(1),
            batch['size_beds'].unsqueeze(1), batch['size_bathrooms'].unsqueeze(1)
        ], dim=1)
        qual_cols = ["review_scores_rating", "review_scores_cleanliness", "review_scores_checkin",
                     "review_scores_communication", "review_scores_location", "review_scores_value",
                     "total_reviews", "host_is_superhost"]
        qual_input = torch.cat([batch[f'qual_{c}'].unsqueeze(1) for c in qual_cols], dim=1)

        full_input = torch.cat([
            batch['loc_geo_position'], size_input, qual_input, batch['season_cyclical'],
            amenities_embed, desc_embed
        ], dim=1)
        return self.main_mlp(full_input).squeeze(-1)

class AdditiveModel(nn.Module):
    """Final interpretable model with six specialized sub-networks."""
    def __init__(self, processor, config: dict):
        super().__init__()
        self.config = config
        self.embed_property_type = nn.Embedding(len(processor.vocabs['property_type']), 8)
        self.embed_room_type = nn.Embedding(len(processor.vocabs['room_type']), 4)
        self.text_transformer = SentenceTransformer(config['TEXT_MODEL_NAME'])
        for param in self.text_transformer.parameters():
            param.requires_grad = False
        for param in self.text_transformer[0].auto_model.encoder.layer[-1].parameters():
            param.requires_grad = True

        def _create_mlp(in_features, layer_sizes):
            layers = []
            for size in layer_sizes:
                layers.append(nn.Linear(in_features, size))
                layers.append(nn.BatchNorm1d(size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=config['DROPOUT_RATE']))
                in_features = size
            return nn.Sequential(*layers[:-1])

        text_embed_dim = self.text_transformer.get_sentence_embedding_dimension()
        self.loc_subnet_body = _create_mlp(config['GEO_EMBEDDING_DIM'], config['HIDDEN_LAYERS_LOCATION'])
        self.size_subnet_body = _create_mlp(16, config['HIDDEN_LAYERS_SIZE_CAPACITY'])
        self.qual_subnet_body = _create_mlp(8, config['HIDDEN_LAYERS_QUALITY'])
        self.amenities_subnet_body = _create_mlp(text_embed_dim, config['HIDDEN_LAYERS_AMENITIES'])
        self.description_subnet_body = _create_mlp(text_embed_dim, config['HIDDEN_LAYERS_DESCRIPTION'])
        self.season_subnet_body = _create_mlp(2, config['HIDDEN_LAYERS_SEASONALITY'])

        self.loc_subnet_head = nn.Linear(config['HIDDEN_LAYERS_LOCATION'][-1], 1)
        self.size_subnet_head = nn.Linear(config['HIDDEN_LAYERS_SIZE_CAPACITY'][-1], 1)
        self.qual_subnet_head = nn.Linear(config['HIDDEN_LAYERS_QUALITY'][-1], 1)
        self.amenities_subnet_head = nn.Linear(config['HIDDEN_LAYERS_AMENITIES'][-1], 1)
        self.description_subnet_head = nn.Linear(config['HIDDEN_LAYERS_DESCRIPTION'][-1], 1)
        self.season_subnet_head = nn.Linear(config['HIDDEN_LAYERS_SEASONALITY'][-1], 1)

    def forward(self, batch: dict, return_details: bool = False):
        # --- CORRECTED CODE: Squeeze the extra dimension from the tokenizers ---
        amenities_tokens = {key: val.squeeze(1) for key, val in batch['amenities_tokens'].items()}
        desc_tokens = {key: val.squeeze(1) for key, val in batch['description_tokens'].items()}
        # --- END OF CORRECTION ---
        
        amenities_embed = self.text_transformer(amenities_tokens)['sentence_embedding']
        desc_embed = self.text_transformer(desc_tokens)['sentence_embedding']
        
        size_input = torch.cat([
            self.embed_property_type(batch['size_property_type']), self.embed_room_type(batch['size_room_type']),
            batch['size_accommodates'].unsqueeze(1), batch['size_bedrooms'].unsqueeze(1),
            batch['size_beds'].unsqueeze(1), batch['size_bathrooms'].unsqueeze(1)
        ], dim=1)
        qual_cols = ["review_scores_rating", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication", 
                     "review_scores_location", "review_scores_value", "total_reviews", "host_is_superhost"]
        qual_input = torch.cat([batch[f'qual_{c}'].unsqueeze(1) for c in qual_cols], dim=1)

        h_loc = self.loc_subnet_body(batch['loc_geo_position'])
        h_size = self.size_subnet_body(size_input)
        h_qual = self.qual_subnet_body(qual_input)
        h_amenities = self.amenities_subnet_body(amenities_embed)
        h_desc = self.description_subnet_body(desc_embed)
        h_season = self.season_subnet_body(batch['season_cyclical'])

        p_loc = self.loc_subnet_head(h_loc)
        p_size = self.size_subnet_head(h_size)
        p_qual = self.qual_subnet_head(h_qual)
        p_amenities = self.amenities_subnet_head(h_amenities)
        p_desc = self.description_subnet_head(h_desc)
        p_season = self.season_subnet_head(h_season)

        predicted_log_deviation = (p_loc + p_size + p_qual + p_amenities + p_desc + p_season)

        if not return_details:
            return predicted_log_deviation.squeeze(-1)
        
        return {
            'predicted_log_deviation': predicted_log_deviation.squeeze(-1),
            'p_location': p_loc.squeeze(-1), 'p_size_capacity': p_size.squeeze(-1),
            'p_quality': p_qual.squeeze(-1), 'p_amenities': p_amenities.squeeze(-1),
            'p_description': p_desc.squeeze(-1), 'p_seasonality': p_season.squeeze(-1),
            'h_location': h_loc, 'h_size_capacity': h_size, 'h_quality': h_qual,
            'h_amenities': h_amenities, 'h_description': h_desc, 'h_seasonality': h_season
        }

class AblationAdditiveModel(AdditiveModel):
    """Subclass of AdditiveModel for running ablation studies."""
    def __init__(self, processor, config: dict, exclude_axes: list = None):
        super().__init__(processor, config)
        self.exclude_axes = set(exclude_axes) if exclude_axes else set()

    def forward(self, batch: dict, return_details: bool = False):
        # The base forward pass already calculates everything we need
        full_output = super().forward(batch, return_details=True)

        contributions = {
            'location': full_output['p_location'], 'size_capacity': full_output['p_size_capacity'],
            'quality': full_output['p_quality'], 'amenities': full_output['p_amenities'],
            'description': full_output['p_description'], 'seasonality': full_output['p_seasonality']
        }
        
        # Sum only the non-excluded axes
        total_deviation = torch.zeros_like(full_output['p_location'])
        for axis_name, contribution in contributions.items():
            if axis_name not in self.exclude_axes:
                total_deviation += contribution
        
        if return_details:
            full_output['predicted_log_deviation'] = total_deviation
            return full_output
        return total_deviation