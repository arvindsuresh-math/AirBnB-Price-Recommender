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

def find_nearest_neighbors(query_idx: int, top_k: int = 5, radius_miles: float = 2.0):
    """
    Finds the top K nearest neighbors for a listing within a geographic radius,
    explicitly excluding other instances of the same listing (e.g., from
    different months).
    """
    # 1. Geospatial Filtering: Create the initial candidate pool
    query_lat_lon_rad = lat_lon_rad[query_idx]
    distances_miles = haversine_distance(query_lat_lon_rad, lat_lon_rad)
    candidate_indices = np.where((distances_miles > 0) & (distances_miles <= radius_miles))[0]

    # ---: Filter out listings with the same ID as the query ---
    query_id = listing_ids[query_idx]
    candidate_ids = listing_ids[candidate_indices]
    mask = (candidate_ids != query_id)
    candidate_indices = candidate_indices[mask]
    

    if len(candidate_indices) < top_k:
        print(f"Warning: Found only {len(candidate_indices)} unique candidates within {radius_miles} miles.")
        top_k = len(candidate_indices)
        if top_k == 0: return [], {}

    # 2. Calculate Axis-Importance Weights (excluding location)
    query_contributions = {name: p_vec[query_idx] for name, p_vec in price_contributions.items()}
    weights = calculate_axis_importances(query_contributions, exclude_axes=['location'])

    # 3. Calculate and combine weighted distances for the filtered candidates
    final_scores = np.zeros(len(candidate_indices))
    search_axes = [axis for axis in hidden_states.keys() if axis != 'location']

    for axis in search_axes:
        h_matrix = hidden_states[axis]
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

# Retrieves the neighborhood_log_mean for a given neighborhood name.
def get_neighborhood_log_mean(neighborhood_name: str, neighborhood_df: pd.DataFrame) -> float:
    log_mean = neighborhood_df[neighborhood_df['neighbourhood_cleansed'] == neighborhood_name]['neighborhood_log_mean']
    if not log_mean.empty:
        return log_mean.iloc[0]
    else:
        return None
    
# ========================================================================================
# MAIN STREAMLIT APPLICATION
# ========================================================================================
# This section contains the user interface and interactive components of the web app.
# Streamlit creates a web interface automatically from Python code.

# --- App Configuration ---
# NOTE: st.set_page_config is now in the main streamlit_app.py file
st.title("🏙️ Airbnb Map Explorer")  # Main heading shown at the top of the page

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
        city_coords: Coordinates for centering the map
        geojson_neighborhood_field: The field name for neighborhoods in the GeoJSON
    """
    # Get the directory where this Python file is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine file names based on city
    if city == 'NYC':
        parquet_file = "nyc_map_dataset.parquet"
        geojson_file = "nyc-neighbourhoods.geojson"
        city_coords = [40.7128, -74.0060]  # NYC coordinates
        geojson_neighborhood_field = 'neighbourhood'
    else:  # Toronto
        parquet_file = "toronto_map_dataset.parquet"
        geojson_file = "toronto-neighbourhoods.geojson"
        city_coords = [43.6532, -79.3832]  # Toronto coordinates
        geojson_neighborhood_field = 'AREA_NAME'
    
    # Load the main dataset (Parquet is a fast, compressed file format for data)
    listings_df = pd.read_parquet(os.path.join(script_dir, parquet_file))
    
    # Load the neighborhood boundary data for map visualization
    geojson_path = os.path.join(script_dir, geojson_file)
    try:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.warning(f"⚠️ GeoJSON file '{geojson_file}' not found. Neighborhood boundaries will not be displayed.")
        geojson_data = None

    return listings_df, geojson_data, city_coords, geojson_neighborhood_field

# --- Initialize city selector in session state ---
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = "NYC"

# Load data based on selected city
listings_df, geojson_data, city_coords, geojson_neighborhood_field = load_data(st.session_state["selected_city"])

# Prepare data for similarity search at module level
hidden_states = {col.replace('h_', ''): np.stack(listings_df[col].values)
                for col in listings_df.columns if col.startswith('h_')}

price_contributions = {col.replace('p_', ''): listings_df[col].to_numpy()
                      for col in listings_df.columns if col.startswith('p_')}

listing_ids = listings_df['id'].to_numpy()
lat_lon_rad = np.deg2rad(listings_df[['latitude', 'longitude']].to_numpy())

# Create neighborhood_df - handle missing neighborhood_log_mean column
if 'neighborhood_log_mean' in listings_df.columns:
    neighborhood_df = listings_df[['neighbourhood_cleansed', 'neighborhood_log_mean']].drop_duplicates()
else:
    # If neighborhood_log_mean doesn't exist, compute it from the price column
    neighborhood_price_means = listings_df.groupby('neighbourhood_cleansed')['price'].apply(lambda x: np.log1p(x.mean())).to_dict()
    listings_df['neighborhood_log_mean'] = listings_df['neighbourhood_cleansed'].map(neighborhood_price_means)
    neighborhood_df = listings_df[['neighbourhood_cleansed', 'neighborhood_log_mean']].drop_duplicates()

# --- Default filter values ---
min_price = int(listings_df['price'].quantile(0.25))
max_price = int(listings_df['price'].quantile(0.50))


# ========================================================================================
# STREAMLIT WEB APPLICATION SETUP
# ========================================================================================
# 
# This section creates the user interface and handles user interactions.
# Streamlit is a Python framework that makes it easy to create web apps for data science.
#

# --- DATA PREPARATION FOR UI COMPONENTS ---
# Calculate price ranges and defaults for the filter controls
# This ensures our filters have sensible starting values

# Get the price range from the actual data (rounded to integers for user-friendly display)
price_min = int(listings_df['price'].min())
price_max = int(listings_df['price'].max())

# Get default room types (start with the most common one)
room_type_defaults = [listings_df['room_type'].unique()[0]]

# Get the most common neighborhood as a sensible default
neighborhood_mode = listings_df['neighbourhood_cleansed'].mode()
neighborhood_defaults = [neighborhood_mode[0] if len(neighborhood_mode) > 0 else "Unknown"]

# Get accommodates and bedrooms ranges
accommodates_min = int(listings_df['accommodates'].min())
accommodates_max = int(listings_df['accommodates'].max())
bedrooms_min = int(listings_df['bedrooms'].min())
bedrooms_max = int(listings_df['bedrooms'].max())

# --- STREAMLIT SESSION STATE MANAGEMENT ---
# Streamlit "session state" preserves values between user interactions
# Think of it as the app's memory - it remembers what the user selected

# Initialize filter settings if they don't exist yet
# This happens when the app first loads
if "price_range" not in st.session_state:
    st.session_state["price_range"] = (min_price, max_price)  # Default to showing all prices
if "room_types" not in st.session_state:
    st.session_state["room_types"] = room_type_defaults  # Default to most common room type
if "neighbourhoods" not in st.session_state:
    st.session_state["neighbourhoods"] = neighborhood_defaults  # Default to most common neighborhood
if "accommodates_range" not in st.session_state:
    st.session_state["accommodates_range"] = (accommodates_min, accommodates_max)
if "bedrooms_range" not in st.session_state:
    st.session_state["bedrooms_range"] = (bedrooms_min, bedrooms_max)
if "show_listings" not in st.session_state:
    st.session_state["show_listings"] = False  # Start with map hidden until user clicks "Load"

# Ensure session state values are within valid ranges for current dataset
# This handles city changes where min/max values may differ
price_range_min_clamped = max(price_min, min(st.session_state["price_range"][0], price_max))
price_range_max_clamped = max(price_min, min(st.session_state["price_range"][1], price_max))
st.session_state["price_range"] = (price_range_min_clamped, price_range_max_clamped)

accommodates_min_clamped = max(accommodates_min, min(st.session_state["accommodates_range"][0], accommodates_max))
accommodates_max_clamped = max(accommodates_min, min(st.session_state["accommodates_range"][1], accommodates_max))
st.session_state["accommodates_range"] = (accommodates_min_clamped, accommodates_max_clamped)

bedrooms_min_clamped = max(bedrooms_min, min(st.session_state["bedrooms_range"][0], bedrooms_max))
bedrooms_max_clamped = max(bedrooms_min, min(st.session_state["bedrooms_range"][1], bedrooms_max))
st.session_state["bedrooms_range"] = (bedrooms_min_clamped, bedrooms_max_clamped)

# --- CALLBACK FUNCTIONS ---
# These functions are called when users click buttons
# They update the session state to trigger different app behaviors

def show_listings_callback():
    """Called when user clicks 'Load Listings' button."""
    st.session_state["show_listings"] = True

def reset_filters_callback():
    """Called when user clicks 'Reset Filters' button."""
    st.session_state["price_range"] = (min_price, max_price)
    st.session_state["room_types"] = room_type_defaults
    st.session_state["neighbourhoods"] = neighborhood_defaults
    st.session_state["accommodates_range"] = (accommodates_min, accommodates_max)
    st.session_state["bedrooms_range"] = (bedrooms_min, bedrooms_max)
    st.session_state["show_listings"] = False

# --- SIDEBAR USER CONTROLS ---
# The sidebar provides filtering controls for users to customize their experience
# Each control is connected to session state so values persist across interactions

st.sidebar.header("🔍 Filter Listings")

# --- City Selector ---
city_selection = st.sidebar.selectbox(
    "Select City",
    options=["NYC", "Toronto"],
    index=0 if st.session_state["selected_city"] == "NYC" else 1,
    key="city_selector"
)

# If city changed, update session state and reload data
if city_selection != st.session_state["selected_city"]:
    st.session_state["selected_city"] = city_selection
    st.session_state["show_listings"] = False  # Reset map display
    st.rerun()

# Action buttons for loading data and resetting filters
st.sidebar.button("Load Listings", on_click=show_listings_callback)
st.sidebar.button("Reset Filters", on_click=reset_filters_callback)

# Room type selector - users can choose multiple room types
st.sidebar.multiselect(
    "Room Type",
    options=listings_df['room_type'].unique(),  # All available room types in our data
    key="room_types"                            # Connected to st.session_state["room_types"]
)

# Neighborhood selector - users can choose multiple neighborhoods
st.sidebar.multiselect(
    "Neighborhood",
    options=listings_df['neighbourhood_cleansed'].unique(),  # All NYC neighborhoods in our data
    #default=neighborhood_defaults,                  # Default to most common neighborhood
    key="neighbourhoods"                           # Connected to st.session_state["neighbourhoods"]
)

# Price range inputs - allows users to type in min and max prices
st.sidebar.subheader("Price Range ($)")
col1, col2 = st.sidebar.columns(2)
with col1:
    price_range_min = st.number_input(
        "Min Price",
        min_value=price_min,
        max_value=price_max,
        value=st.session_state["price_range"][0],
        step=10,
        key="temp_price_min"
    )
with col2:
    price_range_max = st.number_input(
        "Max Price",
        min_value=price_min,
        max_value=price_max,
        value=st.session_state["price_range"][1],
        step=10,
        key="temp_price_max"
    )

# Update session state with the typed values
st.session_state["price_range"] = (price_range_min, price_range_max)

# --- Accommodates filter ---
st.sidebar.subheader("Accommodates")
col3, col4 = st.sidebar.columns(2)
with col3:
    accommodates_range_min = st.number_input(
        "Min",
        min_value=accommodates_min,
        max_value=accommodates_max,
        value=st.session_state["accommodates_range"][0],
        step=1,
        key="temp_accommodates_min"
    )
with col4:
    accommodates_range_max = st.number_input(
        "Max",
        min_value=accommodates_min,
        max_value=accommodates_max,
        value=st.session_state["accommodates_range"][1],
        step=1,
        key="temp_accommodates_max"
    )

st.session_state["accommodates_range"] = (accommodates_range_min, accommodates_range_max)

# --- Bedrooms filter ---
st.sidebar.subheader("Bedrooms")
col5, col6 = st.sidebar.columns(2)
with col5:
    bedrooms_range_min = st.number_input(
        "Min",
        min_value=bedrooms_min,
        max_value=bedrooms_max,
        value=st.session_state["bedrooms_range"][0],
        step=1,
        key="temp_bedrooms_min"
    )
with col6:
    bedrooms_range_max = st.number_input(
        "Max",
        min_value=bedrooms_min,
        max_value=bedrooms_max,
        value=st.session_state["bedrooms_range"][1],
        step=1,
        key="temp_bedrooms_max"
    )

st.session_state["bedrooms_range"] = (bedrooms_range_min, bedrooms_range_max)

# --- MAIN MAP DISPLAY AND DATA FILTERING ---
# This section only runs when the user has clicked "Load Listings" and selected filters
# It creates an interactive map with filtered Airbnb listings

if st.session_state["show_listings"] and st.session_state["neighbourhoods"] and st.session_state["room_types"]:
    
    # --- DATA FILTERING ---
    # Apply all user-selected filters to create a subset of listings to display
    filtered_df = listings_df[
        (listings_df['price'] >= st.session_state["price_range"][0]) &    # Price minimum
        (listings_df['price'] <= st.session_state["price_range"][1]) &    # Price maximum
        (listings_df['room_type'].isin(st.session_state["room_types"])) & # Selected room types
        (listings_df['neighbourhood_cleansed'].isin(st.session_state["neighbourhoods"])) &  # Selected neighborhoods
        (listings_df['accommodates'] >= st.session_state["accommodates_range"][0]) &  # Accommodates minimum
        (listings_df['accommodates'] <= st.session_state["accommodates_range"][1]) &  # Accommodates maximum
        (listings_df['bedrooms'] >= st.session_state["bedrooms_range"][0]) &  # Bedrooms minimum
        (listings_df['bedrooms'] <= st.session_state["bedrooms_range"][1])    # Bedrooms maximum
    ]
    
    # Display how many listings match the current filters
    st.header(f"🗺️ Showing {len(filtered_df)} Listings")
    
    # --- MAP CREATION ---
    # Create the base map centered on the selected city with a clean, minimal style
    m = folium.Map(
        location=city_coords,          # City coordinates (from load_data)
        zoom_start=11,                 # Zoom level (higher = more zoomed in)
        tiles="cartodb positron"       # Clean, light-colored map style
    )
    # --- NEIGHBORHOOD BOUNDARIES ---
    # Add neighborhood boundary lines to help users understand the geographic context
    if geojson_data:
        # Enrich GeoJSON with neighborhood_log_mean data
        for feature in geojson_data['features']:
            neighborhood_name = feature['properties'].get(geojson_neighborhood_field)
            if neighborhood_name:
                log_mean = get_neighborhood_log_mean(neighborhood_name, neighborhood_df)
                if log_mean is not None:
                    avg_price = np.exp(log_mean)
                    feature['properties']['avg_price'] = f"${avg_price:.2f}"
                else:
                    feature['properties']['avg_price'] = "N/A"
        
        folium.GeoJson(
            geojson_data,                    # Geographic boundary data loaded earlier
            name="Neighborhoods",           # Layer name (for map controls)
            style_function=lambda feature: {
                "color": "red",             # Boundary line color
                "weight": 1.5,              # Line thickness
                "fillOpacity": 0            # No fill (transparent inside)
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[geojson_neighborhood_field, "avg_price"],   # What data to show on hover
                aliases=["Neighborhood:", "Average Price:"],  # User-friendly labels
                localize=True              # Format for local conventions
            )
        ).add_to(m)
    
    # --- FALLBACK NEIGHBORHOOD MARKERS ---
    # If we don't have boundary data, show neighborhood center points instead
    if not geojson_data:
        # Calculate the center point of each neighborhood by averaging coordinates
        neighborhood_centers = filtered_df.groupby('neighbourhood_cleansed')[['latitude', 'longitude']].mean()
        
        if not neighborhood_centers.empty:
            # Add a marker at each neighborhood center
            for neighborhood, coords in neighborhood_centers.iterrows():
                folium.Marker(
                    location=[coords['latitude'], coords['longitude']],
                    popup=neighborhood,                                     # Text shown when clicked
                    icon=folium.Icon(color='red', icon='info-sign')        # Red info icon
                ).add_to(m)
    
    # --- LISTING MARKERS ---
    # Add a clickable marker for each Airbnb listing that passes the filters
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            # Create HTML content for the popup that appears when marker is clicked
            popup_html = f"""
            <b>Neighborhood:</b> {row['neighbourhood_cleansed']}<br>
            <b>Room Type:</b> {row['room_type']}<br>
            <b>Price:</b> ${row['price']}<br>
            <b>Accommodates:</b> {row['accommodates']}<br>
            <b>Bedrooms:</b> {row['bedrooms']}<br>
            <b>Listing ID:</b> {row['id']}
            """
            
            # Create a small circle marker for each listing
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],  # GPS coordinates
                radius=6,                                      # Circle size in pixels
                color="darkred",                              # Border color
                fill=True,                                    # Fill the circle
                fill_color="red",                            # Interior color
                fill_opacity=0.8,                            # Transparency (0.8 = 80% opaque)
                popup=popup_html,                            # Info shown when clicked
            ).add_to(m)
    
    # --- DISPLAY THE INTERACTIVE MAP ---
    # Embed the folium map into the Streamlit app and capture user interactions
    st_data = st_folium(
        m,                    # The map object we created
        width='100%',         # Take full width of the container
        height=700           # Fixed height in pixels
    )
    # --- AI ANALYSIS SECTION ---
    # This section handles when users click on map markers to get detailed AI analysis
    st.header("📊 AI Analysis")
    
    # Check if the user clicked on something in the map
    if st_data and st_data.get("last_object_clicked"):
        st.write("You clicked a listing marker!")
        clicked_data = st_data["last_object_clicked"]
        #st.json(clicked_data)  # Show raw click data for debugging
        
        # --- FIND THE CLICKED LISTING ---
        # Match the clicked coordinates to an actual listing in our data
        if 'lat' in clicked_data and 'lng' in clicked_data:
            clicked_lat = clicked_data['lat']
            clicked_lng = clicked_data['lng']
            
            # Find the closest listing within a small tolerance
            # GPS coordinates can have tiny rounding differences, so we need some tolerance
            tolerance = 0.001  # About 100 meters accuracy
            matching_listing = filtered_df[
                (abs(filtered_df['latitude'] - clicked_lat) < tolerance) &
                (abs(filtered_df['longitude'] - clicked_lng) < tolerance)
            ]
            
            if not matching_listing.empty:
                # Use the first matching listing
                listing_row = matching_listing.iloc[0]
                clicked_listing = listing_row.to_dict()
                clicked_listing['lat'] = clicked_lat
                clicked_listing['lng'] = clicked_lng
                st.write(f"**Found exact match:** {clicked_listing.get('name', 'Unknown')} in {clicked_listing.get('neighbourhood_cleansed', 'Unknown')}")
                
                # Get the index in the full dataset for similarity search
                clicked_idx = listings_df[listings_df['id'] == clicked_listing['id']].index[0]
                
                # --- FIND SIMILAR LISTINGS ---
                st.subheader("🔍 Similar Properties Nearby")
                
                try:
                    neighbor_indices, weights = find_nearest_neighbors(clicked_idx, top_k=5, radius_miles=2.0)
                    
                    if len(neighbor_indices) > 0:
                        # Show why these are similar
                        st.write("**Similarity factors (how we matched):**")
                        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                        for axis, weight in sorted_weights:
                            st.write(f"- {axis.replace('_', ' ').title()}: {weight:.1%}")
                        
                        # Display similar listings
                        st.write(f"\n**Top {len(neighbor_indices)} similar listings:**")
                        similar_listings = listings_df.loc[neighbor_indices]
                        
                        # Create comparison table with additional columns
                        comparison_data = {
                            'Similarity Rank': ['Your Selection'] + [f'#{i+1}' for i in range(len(neighbor_indices))],
                            'Name': [clicked_listing.get('name', 'Unknown')] + similar_listings['name'].tolist(),
                            'Neighborhood': [clicked_listing.get('neighbourhood_cleansed', 'Unknown')] + similar_listings['neighbourhood_cleansed'].tolist(),
                            'Room Type': [clicked_listing.get('room_type', 'Unknown')] + similar_listings['room_type'].tolist(),
                            'Accommodates': [clicked_listing.get('accommodates', 0)] + similar_listings['accommodates'].tolist(),
                            'Bedrooms': [clicked_listing.get('bedrooms', 0)] + similar_listings['bedrooms'].tolist(),
                            'Price': [f"${clicked_listing.get('price', 0):.2f}"] + [f"${p:.2f}" for p in similar_listings['price']],
                            'Predicted Price': [f"${clicked_listing.get('predicted_price', 0):.2f}"] + [f"${p:.2f}" for p in similar_listings['predicted_price']],
                            'Neighborhood Average': [f"${np.exp(clicked_listing.get('neighborhood_log_mean', 0)):.2f}"] + [f"${np.exp(p):.2f}" for p in similar_listings['neighborhood_log_mean']],
                            'Location % Adjustment': [f"{(np.exp(clicked_listing.get('p_location', 0))-1)*100:.4f}%"] + [f"{(np.exp(p)-1)*100:.4f}%" for p in similar_listings['p_location']],
                            'Size Capacity % Adjustment': [f"{(np.exp(clicked_listing.get('p_size_capacity', 0))-1)*100:.4f}%"] + [f"{(np.exp(p)-1)*100:.4f}%" for p in similar_listings['p_size_capacity']],
                            'Quality % Adjustment': [f"{(np.exp(clicked_listing.get('p_quality', 0))-1)*100:.4f}%"] + [f"{(np.exp(p)-1)*100:.4f}%" for p in similar_listings['p_quality']],
                            'Amenities % Adjustment': [f"{(np.exp(clicked_listing.get('p_amenities', 0))-1)*100:.4f}%"] + [f"{(np.exp(p)-1)*100:.4f}%" for p in similar_listings['p_amenities']],
                            'Description % Adjustment': [f"{(np.exp(clicked_listing.get('p_description', 0))-1)*100:.4f}%"] + [f"{(np.exp(p)-1)*100:.4f}%" for p in similar_listings['p_description']],
                            'Seasonality % Adjustment': [f"{(np.exp(clicked_listing.get('p_seasonality', 0))-1)*100:.4f}%"] + [f"{(np.exp(p)-1)*100:.4f}%" for p in similar_listings['p_seasonality']],
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                    else:
                        st.warning("No similar listings found within 2 miles.")
                        
                except Exception as e:
                    st.error(f"Error finding similar listings: {e}")
                
            else:
                # If no exact match, find the closest listing
                distances = ((filtered_df['latitude'] - clicked_lat)**2 + (filtered_df['longitude'] - clicked_lng)**2)**0.5
                closest_idx = distances.idxmin()
                listing_row = filtered_df.loc[closest_idx]
                clicked_listing = listing_row.to_dict()
                clicked_listing['lat'] = clicked_lat
                clicked_listing['lng'] = clicked_lng
                st.write(f"**Using closest match:** {clicked_listing.get('name', 'Unknown')} in {clicked_listing.get('neighbourhood_cleansed', 'Unknown')}")
                st.write(f"**Distance:** {distances[closest_idx]:.4f} degrees")
        else:
            # Fallback if coordinates not available
            clicked_listing = clicked_data
            st.write("**Using fallback data - no coordinates found**")
    else:
        st.write("Click a marker on the map to view listing details and find similar properties.")

# --- FEATURE TRENDS OVER TIME SECTION ---
# This section shows how different features of a listing change across months
if st.session_state["show_listings"] and st_data and st_data.get("last_object_clicked"):
    if 'lat' in clicked_data and 'lng' in clicked_data:
        st.header("📈 Feature Trends Over Time")
        
        # Get all instances of this listing across different months
        listing_id = clicked_listing['id']
        listing_history = listings_df[listings_df['id'] == listing_id].copy()
        
        # Only proceed if we have month data
        if 'month' in listing_history.columns and len(listing_history) > 0:
            # Sort by month for proper time series display
            listing_history = listing_history.sort_values('month')
            
            # Transform p_* columns to percentage adjustments
            p_columns = ['p_location', 'p_size_capacity', 'p_quality', 'p_amenities', 'p_description', 'p_seasonality']
            for col in p_columns:
                if col in listing_history.columns:
                    listing_history[f'{col}_pct'] = (np.exp(listing_history[col]) - 1) * 100
            
            # Define available features to plot
            available_features = {
                'price': 'Price ($)',
                #'predicted_price': 'Predicted Price ($)',
                'accommodates': 'Accommodates',
                'bedrooms': 'Bedrooms',
                'p_location_pct': 'Location Adjustment (%)',
                'p_size_capacity_pct': 'Size/Capacity Adjustment (%)',
                'p_quality_pct': 'Quality Adjustment (%)',
                'p_amenities_pct': 'Amenities Adjustment (%)',
                'p_description_pct': 'Description Adjustment (%)',
                'p_seasonality_pct': 'Seasonality Adjustment (%)',
            }
            
            # Filter to only features that exist in the dataset
            available_features = {k: v for k, v in available_features.items() if k in listing_history.columns}
            
            # Feature selector
            selected_feature = st.selectbox(
                "Select feature to visualize",
                options=list(available_features.keys()),
                format_func=lambda x: available_features[x],
                index=0  # Default to 'price'
            )
            
            # Create the line chart
            if selected_feature in listing_history.columns:
                # Prepare data for plotting
                chart_data = listing_history[['month', selected_feature]].dropna()
                
                if not chart_data.empty:
                    # Create Altair chart
                    chart = alt.Chart(chart_data).mark_line(point=True).encode(
                        x=alt.X('month:Q', 
                                title='Month',
                                axis=alt.Axis(format='d')),
                        y=alt.Y(f'{selected_feature}:Q', 
                                title=available_features[selected_feature]),
                        tooltip=[
                            alt.Tooltip('month:Q', title='Month'),
                            alt.Tooltip(f'{selected_feature}:Q', 
                                       title=available_features[selected_feature],
                                       format='.2f')
                        ]
                    ).properties(
                        width='container',
                        height=400,
                        title=f'{available_features[selected_feature]} Over Time for: {clicked_listing.get("name", "Selected Listing")}'
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Show summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    feature_values = chart_data[selected_feature]
                    with col1:
                        st.metric("Mean", f"{feature_values.mean():.2f}")
                    with col2:
                        st.metric("Min", f"{feature_values.min():.2f}")
                    with col3:
                        st.metric("Max", f"{feature_values.max():.2f}")
                    with col4:
                        st.metric("Std Dev", f"{feature_values.std():.2f}")
                    
                else:
                    st.warning(f"No data available for {available_features[selected_feature]} across months.")
            else:
                st.error(f"Selected feature '{selected_feature}' not found in the dataset.")
        else:
            st.info("Month data not available for this listing. Feature trends over time cannot be displayed.")

# ========================================================================================
# END OF APPLICATION
# ========================================================================================