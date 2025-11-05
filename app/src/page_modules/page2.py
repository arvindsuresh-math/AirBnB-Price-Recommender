# Basic Python libraries
import os  # For working with file paths and directories
import altair as alt  # For creating charts and visualizations
import streamlit as st  # The main web app framework
import pandas as pd  # For working with data tables (DataFrames)
import folium  # For creating interactive maps
import json  # For reading JSON files (like map data)

# Machine Learning and AI libraries
import torch  # PyTorch - for deep learning models
import torch.nn as nn  # For building neural network layers
import numpy as np  # For numerical computations and arrays
from torch.utils.data import Dataset, DataLoader  # For handling training data
from transformers import AutoTokenizer  # For processing text
from sentence_transformers import SentenceTransformer  # For text embeddings
import matplotlib.pyplot as plt  # For creating plots
import random  # For generating random numbers
from streamlit_folium import st_folium  # For embedding maps in Streamlit
import torch.serialization  # For loading saved models
import __main__  # For accessing the main module
from scipy.spatial.distance import cdist  # For calculating distances between vectors
import pickle  # For custom unpickling

# ========================================================================================
# FEATURE PROCESSOR CLASS
# ========================================================================================
# Prepares raw DataFrame columns into numerical features for the model.
# The fit/transform pattern prevents data leakage from the validation set.

class FeatureProcessor:
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

# ========================================================================================
# MODEL DEFINITION: ADDITIVE AXIS-BASED NEURAL NETWORK
# ========================================================================================
# A multi-axis neural network that predicts price deviation from a baseline.
# The model is composed of six sub-networks, each processing a different
# modality of the listing data. The final output is the sum of the outputs
# from each sub-network, representing the predicted log-price deviation.

class AdditiveAxisModel(nn.Module):
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

# ========================================================================================
# HELPER FUNCTIONS: DISTANCES AND AXIS-IMPORTANCE
# ========================================================================================
# This defines the core utility functions for the search algorithm. 
# It includes a function for calculating Haversine distance (for geospatial filtering), 
# functions for computing vector distances (Euclidean and Cosine), 
# and a function to calculate the personalized axis-importance weights for a given listing, 
# which now supports excluding certain axes from the calculation.

def haversine_distance(latlon1_rad, latlon2_rad):
    """Calculates the Haversine distance between one or more points in radians."""
    dlon = latlon2_rad[:, 1] - latlon1_rad[1]
    dlat = latlon2_rad[:, 0] - latlon1_rad[0]
    a = np.sin(dlat / 2.0)**2 + np.cos(latlon1_rad[0]) * np.cos(latlon2_rad[:, 0]) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3959 * c # Earth radius in miles

def calculate_axis_importances(p_contributions_single_listing: dict, exclude_axes: list = None) -> dict:
    """
    Calculates the normalized weights for each axis based on the absolute
    magnitude of its price contribution, excluding specified axes.
    """
    exclude_axes = exclude_axes or []
    filtered_contributions = {k: v for k, v in p_contributions_single_listing.items() if k not in exclude_axes}

    abs_contributions = {k: abs(v) for k, v in filtered_contributions.items()}
    total_abs_contribution = sum(abs_contributions.values())

    if total_abs_contribution == 0:
        return {k: 1.0 / len(abs_contributions) for k in abs_contributions}

    return {k: v / total_abs_contribution for k, v in abs_contributions.items()}

def euclidean_distance(vector, matrix):
    """Calculates the Euclidean distance from a vector to all rows in a matrix."""
    return cdist(vector.reshape(1, -1), matrix, 'euclidean').flatten()

def cosine_distance(vector, matrix):
    """Calculates the Cosine distance from a vector to all rows in a matrix."""
    return cdist(vector.reshape(1, -1), matrix, 'cosine').flatten()

# ========================================================================================
# MAIN SEARCH FUNCTION
# ========================================================================================
# This is the main search function. 
# It first identifies a candidate pool of listings within a 2-mile radius of the query listing.
# Then, it calculates a weighted similarity score across all other feature axes 
# (e.g., quality, size, description) to find the most relevantly similar listings within that geographic area.

def find_nearest_neighbors_temp(query_idx: int, temp_hidden_states: dict, temp_price_contributions: dict, temp_listing_ids: np.ndarray, temp_lat_lon_rad: np.ndarray, temp_search_df: pd.DataFrame, top_k: int = 5, radius_miles: float = 2.0):
    """
    Finds the top K nearest neighbors for a listing within a geographic radius,
    explicitly excluding other instances of the same listing (e.g., from
    different months), using temporary data structures.
    """
    # 1. Geospatial Filtering: Create the initial candidate pool
    query_lat_lon_rad = temp_lat_lon_rad[query_idx]
    distances_miles = haversine_distance(query_lat_lon_rad, temp_lat_lon_rad)
    candidate_indices = np.where((distances_miles > 0) & (distances_miles <= radius_miles))[0]

    # --- Filter out listings with the same ID as the query ---
    query_id = temp_listing_ids[query_idx]
    candidate_ids = temp_listing_ids[candidate_indices]
    mask = (candidate_ids != query_id)
    candidate_indices = candidate_indices[mask]

    if len(candidate_indices) < top_k:
        print(f"Warning: Found only {len(candidate_indices)} unique candidates within {radius_miles} miles.")
        top_k = len(candidate_indices)
        if top_k == 0: return [], {}

    # 2. Calculate Axis-Importance Weights (excluding location)
    query_contributions = {name: p_vec[query_idx] for name, p_vec in temp_price_contributions.items()}
    weights = calculate_axis_importances(query_contributions, exclude_axes=['location'])

    # 3. Calculate and combine weighted distances for the filtered candidates
    final_scores = np.zeros(len(candidate_indices))
    search_axes = [axis for axis in temp_hidden_states.keys() if axis != 'location']

    for axis in search_axes:
        h_matrix = temp_hidden_states[axis]
        query_h_vector = h_matrix[query_idx]
        candidate_h_matrix = h_matrix[candidate_indices]

        dist_func = cosine_distance if axis in ["amenities", "description"] else euclidean_distance
        raw_dists = dist_func(query_h_vector, candidate_h_matrix)

        min_dist, max_dist = raw_dists.min(), raw_dists.max()
        normalized_dists = (raw_dists - min_dist) / (max_dist - min_dist) if max_dist > min_dist else np.zeros_like(raw_dists)
        final_scores += weights.get(axis, 0) * normalized_dists

    # 4. Find and return the top_k results
    nearest_candidate_indices = np.argsort(final_scores)
    nearest_original_indices = candidate_indices[nearest_candidate_indices]
    return nearest_original_indices[:top_k], weights
    
# ========================================================================================
# MAIN STREAMLIT APPLICATION
# ========================================================================================
# This section contains the user interface and interactive components of the web app.
# Streamlit creates a web interface automatically from Python code.

# --- App Configuration ---
# Configure the Streamlit page layout and title
# NOTE: st.set_page_config is now in the main streamlit_app.py file
st.title("🏙️ Airbnb Price Navigator")  # Main heading shown at the top of the page

# Register classes in __main__ for unpickling
__main__.FeatureProcessor = FeatureProcessor
__main__.AdditiveAxisModel = AdditiveAxisModel

# --- Initialize city selector in session state ---
if "selected_city_page2" not in st.session_state:
    st.session_state["selected_city_page2"] = "NYC"

@st.cache_data
def load_data(city: str):
    """
    Load and prepare the Airbnb listing data for the application.
    
    Args:
        city: City name ('NYC' or 'Toronto')
    
    This function:
    1. Loads the main dataset (a parquet file with all Airbnb listings)
    2. Selects only the columns we need for display and analysis
    3. Cleans the data by removing listings with missing coordinates or prices
    4. Loads the neighborhood boundaries for map visualization
    
    Returns:
        listings_df: DataFrame with cleaned Airbnb listing data
        geojson_data: Geographic data for drawing neighborhood boundaries on the map
        trained_model: Loaded AI model for price prediction
        processor: Feature processor for the AI model
        config: Configuration settings for the AI model
        default_neighborhood: Default neighborhood for the city
        default_coords: Default coordinates for the city
    """
    # Get the directory where this Python file is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine file names and defaults based on city
    if city == 'NYC':
        parquet_file = "nyc_map_dataset.parquet"
        model_file = "nyc_model_artifacts.pt"
        geojson_file = "nyc-neighbourhoods.geojson"
        default_neighborhood = 'Midtown Manhattan'
        default_coords = {'lat': 40.7580, 'lon': -73.9855}
    else:  # Toronto
        parquet_file = "toronto_map_dataset.parquet"
        model_file = "toronto_model_artifacts.pt"
        geojson_file = "toronto-neighbourhoods.geojson"
        default_neighborhood = 'Waterfront Communities-The Island'
        default_coords = {'lat': 43.6426, 'lon': -79.3871}
    
    # Load the main dataset (Parquet is a fast, compressed file format for data)
    listings_df = pd.read_parquet(os.path.join(script_dir, parquet_file))

    # Load the artifacts needed for the AI model (load to CPU first)
    artifacts = torch.load(os.path.join(script_dir, model_file), map_location='cpu', weights_only=False)
    model = artifacts['model_state_dict']
    processor = artifacts['feature_processor']
    config = artifacts['config']
    config['DEVICE'] = 'cpu'  # Ensure model runs on CPU in Streamlit
    trained_model = AdditiveAxisModel(processor, config)

    # Load the neighborhood boundary data for map visualization
    geojson_path = os.path.join(script_dir, geojson_file)
    try:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.warning(f"⚠️ GeoJSON file '{geojson_file}' not found. Neighborhood boundaries will not be displayed.")
        geojson_data = None

    return listings_df, geojson_data, model, processor, config, trained_model, default_neighborhood, default_coords

listings_df, geojson_data, model, processor, config, trained_model, default_neighborhood, default_coords = load_data(st.session_state["selected_city_page2"])
trained_model.load_state_dict(model)

# Prepare data for similarity search at module level
hidden_states = {col.replace('h_', ''): np.stack(listings_df[col].values)
                for col in listings_df.columns if col.startswith('h_')}

price_contributions = {col.replace('p_', ''): listings_df[col].to_numpy()
                      for col in listings_df.columns if col.startswith('p_')}

listing_ids = listings_df['id'].to_numpy()
lat_lon_rad = np.deg2rad(listings_df[['latitude', 'longitude']].to_numpy())

# Create neighborhood_dict - handle missing neighborhood_log_mean column
if 'neighborhood_log_mean' in listings_df.columns:
    neighborhood_dict = listings_df.groupby('neighbourhood_cleansed')['neighborhood_log_mean'].mean().to_dict()
else:
    # If neighborhood_log_mean doesn't exist, compute it from the price column
    neighborhood_price_means = listings_df.groupby('neighbourhood_cleansed')['price'].apply(lambda x: np.log1p(x.mean())).to_dict()
    neighborhood_dict = neighborhood_price_means
    # Add the computed values to the dataframe for consistency
    listings_df['neighborhood_log_mean'] = listings_df['neighbourhood_cleansed'].map(neighborhood_dict)

# --- Default filter values ---
min_price = int(listings_df['price'].quantile(0.25))
max_price = int(listings_df['price'].quantile(0.50))

tokenizer = AutoTokenizer.from_pretrained(config['TEXT_MODEL_NAME'], use_fast=True)

# ========================================================================================
# STREAMLIT WEB APPLICATION SETUP
# ========================================================================================
# 
# This section creates the user interface and handles user interactions.
# Streamlit is a Python framework that makes it easy to create web apps for data science.
#

# --- DATA PREPARATION FOR UI COMPONENTS ---
# This ensures our filters have sensible starting values

# Get default room types (start with the most common one)
room_type_defaults = [listings_df['room_type'].unique()[0]]

# Get the most common neighborhood as a sensible default
neighborhood_mode = listings_df['neighbourhood_cleansed'].mode()
neighborhood_defaults = [neighborhood_mode[0] if len(neighborhood_mode) > 0 else "Unknown"]


# ========================================================================================
# NEW LISTING INPUT SECTION
# ========================================================================================
st.header("Create a New Listing Estimate")

# --- City Selector (outside the form) ---
city_selection = st.selectbox(
    "Select City",
    options=["NYC", "Toronto"],
    index=0 if st.session_state["selected_city_page2"] == "NYC" else 1,
    key="city_selector_page2"
)

# If city changed, update session state and reload data
if city_selection != st.session_state["selected_city_page2"]:
    st.session_state["selected_city_page2"] = city_selection
    st.rerun()

with st.form("new_listing_form"):
    st.subheader("Listing Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_name = st.text_input("Listing Name", value="Brand New Awesome Apartment with a View")
        input_description = st.text_area("Description", value="A cozy apartment with modern amenities and a great view of the city skyline. Perfect for a short or long stay.", height=100)
        
        # Use city-specific default neighborhood
        neighborhood_options = sorted(listings_df['neighbourhood_cleansed'].unique())
        default_idx = neighborhood_options.index(default_neighborhood) if default_neighborhood in neighborhood_options else 0
        input_neighbourhood = st.selectbox("Neighborhood", options=neighborhood_options, index=default_idx)
        
        input_room_type = st.selectbox("Room Type", options=listings_df['room_type'].unique(), index=list(listings_df['room_type'].unique()).index('Entire home/apt') if 'Entire home/apt' in listings_df['room_type'].unique() else 0)
        input_property_type = st.selectbox("Property Type", options=listings_df['property_type'].unique(), index=list(listings_df['property_type'].unique()).index('Apartment') if 'Apartment' in listings_df['property_type'].unique() else 0)
    
    with col2:
        # Use city-specific default coordinates
        input_latitude = st.number_input("Latitude", value=default_coords['lat'], format="%.4f")
        input_longitude = st.number_input("Longitude", value=default_coords['lon'], format="%.4f")
        input_accommodates = st.number_input("Accommodates", min_value=1, max_value=16, value=2)
        input_bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=1)
        input_beds = st.number_input("Beds", min_value=0, max_value=20, value=1)
        input_bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    
    st.subheader("Amenities & Features")
    input_amenities = st.text_area("Amenities (comma-separated)", value="Wifi, Kitchen, Air conditioning, Heating, Washer, Dryer, TV, Essentials, Hair dryer, Iron, Laptop friendly workspace, Coffee maker, Refrigerator, Microwave, Dishes and silverware, Cooking basics, Oven, Stove, Free parking on premises, Gym, Elevator, Balcony", height=100)
    
    col3, col4, col5 = st.columns(3)
    with col3:
        input_review_rating = st.slider("Review Score Rating", 0.0, 5.0, 4.8, 0.1)
        input_review_cleanliness = st.slider("Review Score Cleanliness", 0.0, 5.0, 4.9, 0.1)
        input_review_checkin = st.slider("Review Score Checkin", 0.0, 5.0, 5.0, 0.1)
    with col4:
        input_review_communication = st.slider("Review Score Communication", 0.0, 5.0, 5.0, 0.1)
        input_review_location = st.slider("Review Score Location", 0.0, 5.0, 4.7, 0.1)
        input_review_value = st.slider("Review Score Value", 0.0, 5.0, 4.8, 0.1)
    with col5:
        input_month = st.selectbox("Month", options=list(range(1, 13)), index=9)  # Default to October (10th month, index 9)
        input_host_superhost = st.checkbox("Host is Superhost", value=True)
        input_instant_bookable = st.checkbox("Instant Bookable", value=True)
    
    submit_button = st.form_submit_button("🔮 Predict Price & Find Similar Listings")

if submit_button:
    # Create the estimate listing DataFrame
    estimate_listing_data = pd.DataFrame({
        'id': [999999],
        'name': [input_name],
        'description': [input_description],
        'host_id': [12345],
        'host_name': ['AI Host'],
        'neighbourhood_cleansed': [input_neighbourhood],
        'latitude': [input_latitude],
        'longitude': [input_longitude],
        'room_type': [input_room_type],
        'price': [0.0],
        'minimum_nights': [3],
        'number_of_reviews': [0],
        'last_review': ['2023-01-01'],
        'reviews_per_month': [0.0],
        'calculated_host_listings_count': [1],
        'availability_365': [300],
        'property_type': [input_property_type],
        'accommodates': [input_accommodates],
        'bathrooms': [input_bathrooms],
        'bedrooms': [input_bedrooms],
        'beds': [input_beds],
        'amenities': [input_amenities],
        'review_scores_rating': [input_review_rating],
        'review_scores_accuracy': [5.0],
        'review_scores_cleanliness': [input_review_cleanliness],
        'review_scores_checkin': [input_review_checkin],
        'review_scores_communication': [input_review_communication],
        'review_scores_location': [input_review_location],
        'review_scores_value': [input_review_value],
        'instant_bookable': [input_instant_bookable],
        'host_is_superhost': [input_host_superhost],
        'total_reviews': [0],
        'month': [input_month]
    })
    
    try:
        with st.spinner("Processing listing and making predictions..."):
            # 1. Transform the new listing data using the fitted processor
            processed_features = processor.transform(estimate_listing_data, neighborhood_dict)
            
            # 2. Create the batch dictionary
            batch = {
                'loc_geo_position': torch.tensor(processed_features['location']['geo_position'], dtype=torch.float32),
                'season_cyclical': torch.tensor(processed_features['seasonality']['cyclical'], dtype=torch.float32),
                'target_price': torch.tensor(processed_features['target_price'], dtype=torch.float32),
                'target_log_deviation': torch.tensor(processed_features['target_log_deviation'], dtype=torch.float32),
                'neighborhood_log_mean': torch.tensor(processed_features['neighborhood_log_mean'], dtype=torch.float32),
            }
            
            # Add size_capacity features
            for k, v in processed_features['size_capacity'].items():
                dtype = torch.long if k in ['property_type', 'room_type'] else torch.float32
                batch[f'size_{k}'] = torch.tensor(v, dtype=dtype)
            
            # Add quality features
            for k, v in processed_features['quality'].items():
                batch[f'qual_{k}'] = torch.tensor(v, dtype=torch.float32)
            
            # Add tokenized text features
            batch['amenities_tokens'] = tokenizer(
                processed_features['amenities_text'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            batch['description_tokens'] = tokenizer(
                processed_features['description_text'],
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # 3. Move batch tensors to the appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
                else:
                    batch[k] = {sk: sv.to(device) for sk, sv in v.items()}
            
            # 4. Run inference
            trained_model.eval()
            trained_model.to(device)
            
            with torch.no_grad():
                outputs = trained_model.forward_with_hidden_states(batch)
            
            # 5. Extract predictions and contributions
            neighborhood_log_mean_val = batch['neighborhood_log_mean'].cpu().numpy()[0]
            predicted_log_deviation = outputs['predicted_log_deviation'].cpu().numpy()[0]
            predicted_log_price = neighborhood_log_mean_val + predicted_log_deviation
            predicted_price = np.exp(predicted_log_price)
            
            # Extract price contributions
            p_contributions = {
                'location': outputs['p_location'].cpu().numpy()[0],
                'size_capacity': outputs['p_size_capacity'].cpu().numpy()[0],
                'quality': outputs['p_quality'].cpu().numpy()[0],
                'amenities': outputs['p_amenities'].cpu().numpy()[0],
                'description': outputs['p_description'].cpu().numpy()[0],
                'seasonality': outputs['p_seasonality'].cpu().numpy()[0],
            }
            
            # Extract hidden states
            h_states = {
                'location': outputs['h_location'].cpu().numpy(),
                'size_capacity': outputs['h_size_capacity'].cpu().numpy(),
                'quality': outputs['h_quality'].cpu().numpy(),
                'amenities': outputs['h_amenities'].cpu().numpy(),
                'description': outputs['h_description'].cpu().numpy(),
                'seasonality': outputs['h_seasonality'].cpu().numpy(),
            }
            
            # 6. Create temporary data structures for similarity search
            # Combine the new listing with existing listings
            temp_search_df = pd.concat([estimate_listing_data, listings_df], ignore_index=True)
            temp_listing_ids = np.concatenate([[999999], listing_ids])
            temp_lat_lon = np.concatenate([[[input_latitude, input_longitude]], listings_df[['latitude', 'longitude']].to_numpy()])
            temp_lat_lon_rad = np.deg2rad(temp_lat_lon)
            
            # Combine hidden states
            temp_hidden_states = {}
            for key in h_states.keys():
                # Only combine hidden states that exist in both datasets
                # 'location' is excluded from the search, so skip it
                if key in hidden_states:
                    temp_hidden_states[key] = np.vstack([h_states[key], hidden_states[key]])
            
            # Combine price contributions
            temp_price_contributions = {}
            for key in p_contributions.keys():
                temp_price_contributions[key] = np.concatenate([[p_contributions[key]], price_contributions[key]])
            
            # 7. Find similar listings (query_idx = 0 for the new listing)
            neighbor_indices, weights = find_nearest_neighbors_temp(
                query_idx=0,
                temp_hidden_states=temp_hidden_states,
                temp_price_contributions=temp_price_contributions,
                temp_listing_ids=temp_listing_ids,
                temp_lat_lon_rad=temp_lat_lon_rad,
                temp_search_df=temp_search_df,
                top_k=5,
                radius_miles=2.0
            )
            
            # Adjust indices to match the original listings_df (subtract 1 because new listing is at index 0)
            neighbor_indices_original = neighbor_indices - 1
            
            # 8. Display results
            st.success("✅ Prediction Complete!")
            
            st.subheader("💰 Predicted Price")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Price", f"${predicted_price:.2f}")
            with col2:
                st.metric("Neighborhood Average", f"${np.exp(neighborhood_log_mean_val):.2f}")
            with col3:
                difference = predicted_price - np.exp(neighborhood_log_mean_val)
                st.metric("Difference from Average", f"${difference:.2f}", delta=f"{(difference / np.exp(neighborhood_log_mean_val) * 100):.1f}%")
            
            st.subheader("📊 Price Contribution Breakdown")
            contrib_data = []
            for axis, value in p_contributions.items():
                percentage_adjustment = (np.exp(value)-1)*100
                contrib_data.append({
                    'Factor': axis.replace('_', ' ').title(),
                    'Log Contribution': f"{value:.4f}",
                    'Price Adjustment': f"{percentage_adjustment:.4f}%"
                })
            contrib_df = pd.DataFrame(contrib_data)
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
            
            # 9. Display similar listings
            if len(neighbor_indices_original) > 0:
                st.subheader("🔍 Similar Properties Nearby")
                
                st.write("**Similarity factors (how we matched):**")
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                for axis, weight in sorted_weights:
                    st.write(f"- {axis.replace('_', ' ').title()}: {weight:.1%}")
                
                st.write(f"\n**Top {len(neighbor_indices_original)} similar listings:**")
                similar_listings = listings_df.loc[neighbor_indices_original]
                
                comparison_data = {
                    'Similarity Rank': ['Your Estimate'] + [f'#{i+1}' for i in range(len(neighbor_indices_original))],
                    'Name': [input_name] + similar_listings['name'].tolist(),
                    'Neighborhood': [input_neighbourhood] + similar_listings['neighbourhood_cleansed'].tolist(),
                    'Room Type': [input_room_type] + similar_listings['room_type'].tolist(),
                    'Accommodates': [input_accommodates] + similar_listings['accommodates'].tolist(),
                    'Bedrooms': [input_bedrooms] + similar_listings['bedrooms'].tolist(),
                    'Price': [f"${predicted_price:.2f}"] + [f"${p:.2f}" for p in similar_listings['price']],
                    'Predicted Price': [f"${predicted_price:.2f}"] + [f"${p:.2f}" for p in similar_listings['predicted_price']],
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No similar listings found within 2 miles.")
                
    except Exception as e:
        st.error(f"Error processing listing: {e}")
        import traceback
        st.code(traceback.format_exc())

# ========================================================================================
# END OF APPLICATION
# ========================================================================================