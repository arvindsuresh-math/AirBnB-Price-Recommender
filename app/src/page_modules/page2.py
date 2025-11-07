# Basic Python libraries
import os  # For working with file paths and directories
import altair as alt  # For creating charts and visualizations
import streamlit as st  # The main web app framework
import pandas as pd  # For working with data tables (DataFrames)
import folium  # For creating interactive maps
import json  # For reading JSON files (like map data)
import plotly.graph_objects as go  # For creating interactive plotly charts

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

# --- Custom Project Scripts ---
from config import config
from data_processing import FeatureProcessor
# Import BOTH model classes and the dataset
from model import AdditiveModel, AirbnbPriceDataset
# Import BOTH inference functions
from inference import run_inference, run_inference_with_details
# Helper functions for finding nearest neighbors
from similarity import haversine_distance, calculate_axis_importances, euclidean_distance, cosine_distance, find_nearest_neighbors
    
# ========================================================================================
# MAIN STREAMLIT APPLICATION
# ========================================================================================
# This section contains the user interface and interactive components of the web app.
# Streamlit creates a web interface automatically from Python code.

# --- App Configuration ---
# Configure the Streamlit page layout and title
# NOTE: st.set_page_config is now in the main streamlit_app.py file
st.title("🏙️ Airbnb Price Recommender")  # Main heading shown at the top of the page

# Register classes in __main__ for unpickling
__main__.FeatureProcessor = FeatureProcessor
__main__.AdditiveModel = AdditiveModel

# --- Initialize city selector in session state ---
if "selected_city_page2" not in st.session_state:
    st.session_state["selected_city_page2"] = "NYC"

@st.cache_resource  # Changed from cache_data to cache_resource for PyTorch models
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
    
    model = artifacts['model']['model_state_dict']
    processor = artifacts['feature_processor']

    config['DEVICE'] = 'cpu'  # Ensure model runs on CPU in Streamlit
    trained_model = AdditiveModel(processor, config)

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

# Ensure processor has required attributes (for backward compatibility with older saved models)
if not hasattr(processor, 'embedding_dim_geo'):
    processor.embedding_dim_geo = config.get('GEO_EMBEDDING_DIM', 32)

# Load model state dict
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

    submit_button = st.form_submit_button("Recommend Price & Find Similar Listings")

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
        with st.spinner("Processing listing and making recommendations..."):
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
            
            # 4. Run inference AND extract intermediate outputs
            trained_model.eval()
            trained_model.to(device)
            
            with torch.no_grad():
                # Call the model with return_details=True to get all intermediate outputs
                outputs = trained_model(batch, return_details=True)
            
            # 5. Extract predictions and contributions
            neighborhood_log_mean_val = batch['neighborhood_log_mean'].cpu().numpy()
            if neighborhood_log_mean_val.ndim > 0:
                neighborhood_log_mean_val = neighborhood_log_mean_val[0]
            
            # Extract prediction and detailed outputs
            if isinstance(outputs, dict):
                # Extract prediction
                if 'prediction' in outputs:
                    predicted_log_deviation = outputs['prediction'].cpu().numpy()
                elif 'log_deviation' in outputs:
                    predicted_log_deviation = outputs['log_deviation'].cpu().numpy()
                else:
                    # Try to find any key with 'pred' in it
                    pred_keys = [k for k in outputs.keys() if 'pred' in k.lower()]
                    if pred_keys:
                        predicted_log_deviation = outputs[pred_keys[0]].cpu().numpy()
                    else:
                        st.error(f"Could not find prediction in outputs. Keys: {list(outputs.keys())}")
                        st.stop()
                
                if predicted_log_deviation.ndim > 0:
                    predicted_log_deviation = predicted_log_deviation[0]
                
                # Extract price contributions (p_*
                p_contributions = {}
                for key in outputs.keys():
                    if key.startswith('p_'):
                        val = outputs[key].cpu().numpy()
                        p_contributions[key.replace('p_', '')] = val.item() if val.size == 1 else val[0]
                
                # Extract hidden states (h_*
                h_states = {}
                for key in outputs.keys():
                    if key.startswith('h_'):
                        h_states[key.replace('h_', '')] = outputs[key].cpu().numpy()
                
                if not p_contributions or not h_states:
                    st.warning("⚠️ Could not extract detailed outputs. Available keys:")
                    st.write(list(outputs.keys()))
                    p_contributions = None
                    h_states = None
            else:
                st.error(f"Expected dict output with return_details=True, got {type(outputs)}")
                st.stop()
                
            predicted_log_price = neighborhood_log_mean_val + predicted_log_deviation
            predicted_price = np.exp(predicted_log_price)
            
            # Only perform similarity search if we have hidden states
            if h_states is not None and p_contributions is not None:
                # 6. Create temporary data structures for similarity search
                temp_search_df = pd.concat([estimate_listing_data, listings_df], ignore_index=True)
                temp_listing_ids = np.concatenate([[999999], listing_ids])
                temp_lat_lon = np.concatenate([[[input_latitude, input_longitude]], listings_df[['latitude', 'longitude']].to_numpy()])
                temp_lat_lon_rad = np.deg2rad(temp_lat_lon)
                
                # Combine hidden states
                temp_hidden_states = {}
                for key in h_states.keys():
                    if key in hidden_states:
                        temp_hidden_states[key] = np.vstack([h_states[key], hidden_states[key]])
                
                # Combine price contributions
                temp_price_contributions = {}
                for key in p_contributions.keys():
                    temp_price_contributions[key] = np.concatenate([[p_contributions[key]], price_contributions[key]])
                
                # 7. Find similar listings (query_idx = 0 for the new listing)
                neighbor_indices, weights = find_nearest_neighbors(
                    query_idx=0,
                    hidden_states=temp_hidden_states,
                    price_contributions=temp_price_contributions,
                    all_listing_ids=temp_listing_ids,
                    lat_lon_rad=temp_lat_lon_rad,
                    top_k=5,
                    radius_miles=2.0
                )
                
                # Adjust indices to match the original listings_df (subtract 1 because new listing is at index 0)
                neighbor_indices_original = neighbor_indices - 1
            
            # 8. Display results
            st.success("✅ Recommendation Complete!")
            
            st.subheader("💰 Recommended Price")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recommended Price", f"${predicted_price:.2f}")
            with col2:
                st.metric("Neighborhood Average", f"${np.exp(neighborhood_log_mean_val):.2f}")
            with col3:
                difference = predicted_price - np.exp(neighborhood_log_mean_val)
                st.metric("Difference from Average", f"${difference:.2f}", delta=f"{(difference / np.exp(neighborhood_log_mean_val) * 100):.1f}%")
            
            # Only show detailed breakdown if we have contributions
            if p_contributions is not None:
                st.subheader("📊 Price Contribution Breakdown")
                contrib_data = []
                for axis, value in p_contributions.items():
                    percentage_adjustment = (np.exp(value)-1)*100
                    contrib_data.append({
                        'Factor': axis.replace('_', ' ').title(),
                        'Price Adjustment': f"{percentage_adjustment:.2f}%"
                    })
                contrib_df = pd.DataFrame(contrib_data)
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                
                # Waterfall Plot for Price Adjustments
                st.subheader("💧 Price Adjustment Waterfall")
                
                base_price = np.exp(neighborhood_log_mean_val)
                
                # Build waterfall by tracking cumulative price
                labels = ['Base Price (Neighborhood Avg)']
                adjustments = [base_price]
                measures = ["absolute"]
                cumulative_price = base_price
                
                # Each contribution is a multiplicative factor, so convert to additive changes
                for axis, log_contribution in p_contributions.items():
                    # The contribution represents: new_price = old_price * exp(log_contribution)
                    # So the dollar change is: cumulative_price * (exp(log_contribution) - 1)
                    dollar_change = cumulative_price * (np.exp(log_contribution) - 1)
                    adjustments.append(dollar_change)
                    labels.append(axis.replace('_', ' ').title())
                    measures.append("relative")
                    # Update cumulative for next iteration
                    cumulative_price = cumulative_price * np.exp(log_contribution)
                
                # Add final price as total
                labels.append('Final Price')
                adjustments.append(predicted_price)
                measures.append("total")
                
                fig = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=measures,
                    x=labels,
                    y=adjustments,
                    text=[f"${val:.2f}" for val in adjustments],
                    textposition="outside"
                ))
                
                fig.update_layout(title="Price Breakdown", showlegend=False, height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display similar listings if we performed the search
            if h_states is not None and len(neighbor_indices_original) > 0:
                st.subheader("🔍 Similar Properties Nearby")
                
                st.write("**Similarity factors (how we matched):**")
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                for axis, weight in sorted_weights:
                    st.write(f"- {axis.replace('_', ' ').title()}: {weight:.1%}")

                st.markdown("We have determined the most similar listings based on your input listing's features. Here are some comparable properties within a 2-mile radius:")

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
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            elif h_states is not None:
                st.warning("No similar listings found within 2 miles.")
                
    except Exception as e:
        st.error(f"Error processing listing: {e}")
        import traceback
        st.code(traceback.format_exc())

# ========================================================================================
# END OF APPLICATION
# ========================================================================================