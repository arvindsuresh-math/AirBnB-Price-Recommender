import streamlit as st
import pandas as pd
import folium
import json
from streamlit_folium import st_folium

# --- App Configuration ---
st.set_page_config(layout="wide")
st.title("🏙️ NYC Airbnb Price Navigator")

# --- Load Data ---
@st.cache_data
def load_data():
    # Example listing data — replace this with your parquet load
    data = {
        'id': [101, 102, 103, 104, 105],
        'latitude': [40.615, 40.871, 40.7128, 40.7831, 40.81],
        'longitude': [-73.768, -73.85, -74.0060, -73.9712, -73.94],
        'price': [80, 120, 250, 400, 150],
        'neighbourhood': ['Bayswater', 'Allerton', 'Financial District', 'Upper West Side', 'Harlem'],
        'room_type': ['Private room', 'Entire home/apt', 'Entire home/apt', 'Entire home/apt', 'Private room']
    }
    listings_df = pd.DataFrame(data)

    # Load GeoJSON safely
    geojson_path = "nyc_neighbourhoods.geojson"
    try:
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        st.error(f"⚠️ GeoJSON file '{geojson_path}' not found. Please place it in the same directory as this app.")
        geojson_data = None

    return listings_df, geojson_data


listings_df, geojson_data = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Listings")

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

# Apply filters
filtered_df = listings_df[
    (listings_df['price'] >= price_range[0]) &
    (listings_df['price'] <= price_range[1]) &
    (listings_df['room_type'].isin(room_types))
]

# --- Map Display ---
st.header(f"🗺️ Showing {len(filtered_df)} Listings")

# Create Folium map centered on NYC
m = folium.Map(location=[40.7128, -74.0060], zoom_start=11, tiles="cartodb positron")

# --- Add neighborhood boundaries (red outlines) ---
if geojson_data:
    folium.GeoJson(
        geojson_data,
        name="Neighborhoods",
        style_function=lambda feature: {
            "color": "red",
            "weight": 1.5,
            "fillOpacity": 0
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["neighbourhood"],
            aliases=["Neighborhood:"],
            localize=True
        )
    ).add_to(m)

# --- Add Airbnb listings as clickable points ---
for _, row in filtered_df.iterrows():
    popup_html = f"""
    <b>Neighborhood:</b> {row['neighbourhood']}<br>
    <b>Room Type:</b> {row['room_type']}<br>
    <b>Price:</b> ${row['price']}<br>
    <b>Listing ID:</b> {row['id']}
    """
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        color="darkred",
        fill=True,
        fill_color="red",
        fill_opacity=0.8,
        popup=popup_html,
    ).add_to(m)

# Display map
st_data = st_folium(m, width='100%', height=700)

# --- Analysis Placeholder ---
st.header("📊 Analysis")
if st_data and st_data.get("last_object_clicked"):
    st.write("You clicked a listing marker!")
    st.json(st_data["last_object_clicked"])
else:
    st.write("Click a marker on the map to view listing details here.")