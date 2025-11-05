import streamlit as st
import pandas as pd
import folium
import json
from streamlit_folium import st_folium

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("NYC Airbnb Price Navigator")

# --- Load Data (using dummy data for this example) ---
# In your real app, you would load your 'full_dataset_with_details.parquet'
@st.cache_data
def load_data():
    # Create some dummy listing data
    data = {
        'id': [101, 102, 103, 104, 105],
        'latitude': [40.615, 40.871, 40.7128, 40.7831, 40.81],
        'longitude': [-73.768, -73.85, -74.0060, -73.9712, -73.94],
        'price': [80, 120, 250, 400, 150],
        'neighbourhood_cleansed': ['Bayswater', 'Allerton', 'Financial District', 'Upper West Side', 'Harlem'],
        'room_type': ['Private room', 'Entire home/apt', 'Entire home/apt', 'Entire home/apt', 'Private room']
    }
    listings_df = pd.DataFrame(data)

    # Load your actual GeoJSON file
    try:
        with open('nyc_neighbourhoods.geojson', 'r') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error("Make sure 'nyc_neighbourhoods.geojson' is in the same directory.")
        geojson_data = None
        
    return listings_df, geojson_data

listings_df, geojson_data = load_data()


# --- Sidebar Filters ---
st.sidebar.header("Filter Listings")
price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=int(listings_df['price'].min()),
    max_value=int(listings_df['price'].max()),
    value=(int(listings_df['price'].min()), int(listings_df['price'].max()))
)

room_types = st.sidebar.multiselect(
    "Room Type",
    options=listings_df['room_type'].unique(),
    default=listings_df['room_type'].unique()
)

# Filter the dataframe based on sidebar selections
filtered_df = listings_df[
    (listings_df['price'] >= price_range[0]) &
    (listings_df['price'] <= price_range[1]) &
    (listings_df['room_type'].isin(room_types))
]

# --- Main Map Display ---
st.header(f"Showing {len(filtered_df)} Listings")

# Create a Folium map centered on NYC
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Add the neighborhood boundaries from the GeoJSON file
if geojson_data:
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            'color': 'blue',
            'weight': 1,
            'fillOpacity': 0.1
        }
    ).add_to(m)

# Add a marker for each filtered listing
for _, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color='red',
        fill=True,
        fill_color='darkred',
        popup=f"<b>{row['neighbourhood_cleansed']}</b><br>Price: ${row['price']}<br>Type: {row['room_type']}"
    ).add_to(m)

# Render the map in Streamlit
st_folium(m, width='100%')

# --- Placeholder for Analysis Section ---
st.header("Analysis")
st.write("Click a listing on the map to see its comps analysis here.")
# In your real app, clicking a marker would trigger the nearest neighbor search
# and the results would be displayed in this section.