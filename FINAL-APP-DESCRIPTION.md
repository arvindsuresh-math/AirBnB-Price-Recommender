# Project Milestone: App Database & Prototyping Plan

**Objective:** This document outlines the structure and contents of our pre-computed application databases and details the plan for building an interactive front-end prototype.

### 1. The App Database: A Comprehensive Data Dictionary

We have successfully run our trained models on the complete datasets for each city. The output is a single, self-contained Parquet file for each city (e.g., `nyc_app_database.parquet`). This file is the **only data source our application will need.**

Each row in the file represents a unique Airbnb listing for a specific month and contains the following columns:

| Column Group | Column Name(s) | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **Core Listing Features** | `id`, `name`, `description`, `neighbourhood_cleansed`, `property_type`, `room_type`, `accommodates`, `bedrooms`, `price`, `month`, `latitude`, `longitude`, `amenities`, etc. | `int`, `str`, `float` | These are the original, raw features from the dataset. They will be used for filtering, display, and map placement. |
| **Model-Generated Predictions** | `predicted_price` | `float` | The final, calibrated price prediction for the listing, in dollars. |
| **Model-Generated Price Components (Log-Space)** | `p_base` <br> `p_location` <br> `p_size_capacity` <br> `p_quality` <br> `p_amenities` <br> `p_description` <br> `p_seasonality` | `float` | The core output of our model. These are the **additive contributions in log-space**. `p_base` is the neighborhood's average log-price. The other `p_` values are the learned deviations from that base. The sum of all `p_` values gives the total predicted log-price. |
| **Model-Generated Embeddings (Hidden States)** | `h_location` <br> `h_size_capacity` <br> `h_quality` <br> `h_amenities` <br> `h_description` <br> `h_seasonality` | `numpy.ndarray` | High-dimensional vector representations (embeddings) for each feature axis. These vectors capture the model's nuanced understanding of a listing's characteristics. **These are the "What" that will be used for the nearest-neighbor similarity search.** |

---

### 2. Sample Code for Local Analysis & Exploration

To explore the contents of these files, you can use the following code in a Jupyter Notebook. This will load the data and create a few sample visualizations to verify the model's performance.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- User Input: Provide the path to your Parquet file ---
# For example: 'C:/Users/YourUser/Documents/airbnb-project/data/nyc_app_database.parquet'
FILEPATH = "path/to/your/city_app_database.parquet"

# --- Load the Database ---
try:
    df = pd.read_parquet(FILEPATH)
    print(f"Successfully loaded {len(df):,} listings.")
    print("\nData Columns:")
    print(df.columns)
except FileNotFoundError:
    print(f"Error: File not found at '{FILEPATH}'. Please update the path.")
    
# --- Sample Visualizations ---
if 'df' in locals():
    # 1. Scatter Plot: True Price vs. Predicted Price
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x='price', 
        y='predicted_price', 
        data=df.sample(min(len(df), 2000)), # Sample to avoid overplotting
        alpha=0.5
    )
    plt.plot([0, df['price'].max()], [0, df['price'].max()], 'r--', label='Ideal Fit')
    plt.title("True Price vs. Predicted Price")
    plt.xlabel("True Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 2. Histogram: Distribution of the 'Quality' Price Contribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['p_quality'], bins=50, kde=True)
    plt.title("Distribution of the Quality & Reviews Price Component (p_quality)")
    plt.xlabel("Contribution to Log-Price (Negative = Cheaper, Positive = Pricier)")
    plt.ylabel("Number of Listings")
    plt.grid(True)
    plt.show()
```

---

### 3. How the App Database Will Be Used

The central design principle of our application is to separate heavy computation from the user-facing app. Our `_app_database.parquet` file is the result of a **one-time, offline inference process**.

**This means the app itself will not use the model for inference.** Instead, it will:

1. **Load** the entire pre-computed Parquet file into memory at startup.
2. **Filter** this DataFrame based on user selections (e.g., price, room type).
3. When a user clicks a listing, the app **looks up** its pre-computed hidden states (`h_...` vectors) and price contributions (`p_...` values) from the DataFrame.
4. It then performs a **fast, vectorized similarity search** using NumPy/SciPy on the pre-computed hidden states of all other listings.

This architecture ensures our application is:

* **Blazing Fast:** All results are based on lookups and optimized numerical calculations, not slow model inference.
* **Cost-Effective:** The app does not require a GPU to run, allowing it to be deployed on cheaper, standard CPU hardware (including the free tier on Hugging Face Spaces).

---

### 4. Application Blueprint

This section details the proposed application, keeping in mind that our team's strength is in mathematics and analysis, not web development.

#### Tech Stack

We will use a pure Python stack that is powerful yet easy to learn:

* **Framework:** **Streamlit**. It allows us to build a web application using simple Python scripts, without needing to know any HTML, CSS, or JavaScript. It's designed for data-heavy applications.
* **Mapping:** **Folium**. A Python library that makes it trivial to create interactive Leaflet.js maps and overlay our listing data and neighborhood boundaries on them.

#### User Interface and Workflow

The app will have a clean, two-column layout.

**Left Column (Sidebar - The "Controls"):**

* **City Selector:** A dropdown to choose between "NYC" and "Toronto". This will trigger a reload of the correct Parquet and GeoJSON files.
* **Filter Widgets:** A series of simple controls (sliders for price, multi-select boxes for room type, etc.) that users can adjust. These filters will directly query our main DataFrame.
  * *Data Used:* `price`, `room_type`, `accommodates`, `neighbourhood_group`.

**Right Column (Main Area - The "Display"):**

1. **Interactive Map:**
    * An interactive map will be the centerpiece.
    * The neighborhood boundaries will be drawn on the map using the GeoJSON file.
    * Markers will be placed on the map for every listing that matches the current filters.
    * *Data Used:* `latitude`, `longitude`.
2. **Analysis Section (below the map):**
    * This section is initially empty, prompting the user to "Click a listing on the map to analyze."
    * When the user clicks a marker, the app identifies the listing and triggers our `find_nearest_neighbors` function.
    * The section then populates with the **Side-by-Side Comparison Table**, showing the query listing and its top 5 competitors.
    * *Data Used:* `id`, `hidden_states (h_...)`, `price_contributions (p_...)`.
    * Next to the table, we will display the **Multiplicative Price Breakdown** for the selected listing, converting the `p_` values into intuitive multipliers.
        * *Data Used:* All `p_` columns.

---

### 5. Enhancing the App with LLMs

Once the core functionality is built, we can integrate a small Large Language Model (e.g., `google/flan-t5-base`) to provide natural language insights. The primary goal of the LLM is to synthesize the complex numerical data from the comparison table into a clear, comparative narrative.

#### Enhancement: The "Comparative Intelligence" Report

* **Goal:** To automatically generate a textual summary that explains *why* the selected listing is priced differently from its closest competitors, focusing on the most impactful feature axes.

* **How it Works:** When a user selects a listing, the app will:
    1. Find the top nearest neighbors (e.g., 3 comps).
    2. Identify the 2-3 feature axes with the **largest absolute price contributions** for the query listing (these are the most important pricing factors).
    3. For each of these key axes, it will gather the `p_` value (the log-space contribution) for the query and for each of its competitors.
    4. It will also pull a small, relevant snippet of the raw data (e.g., the first 50 characters of the `description`, or the number of `bedrooms`).
    5. This structured data is then formatted into a detailed prompt for the LLM.

* **Example Data Sent to LLM:**

    Let's say the most important factors for the query listing are **Size & Capacity** and **Description**.

  * **Query Listing:**
    * `p_size_capacity`: **0.14** (+15% price impact)
    * Raw data: 2 bedrooms, 2 bathrooms, accommodates 4
    * `p_description`: **-0.09** (-9% price impact)
    * Raw data: "Simple room in a shared apartment."
  * **Competitor 1:**
    * `p_size_capacity`: **0.22** (+25% price impact)
    * Raw data: 3 bedrooms, 2 bathrooms, accommodates 6
    * `p_description`: **0.05** (+5% price impact)
    * Raw data: "Luxury penthouse with stunning skyline views."
  * **Competitor 2:**
    * `p_size_capacity`: **0.05** (+5% price impact)
    * Raw data: 1 bedroom (studio), 1 bathroom, accommodates 2
    * `p_description`: **-0.12** (-11% price impact)
    * Raw data: "Cozy basement room, budget-friendly."

* **Example Prompt:**

    ```
    You are an expert real estate analyst. Based on the following data, write a short, comparative analysis explaining why the Query Listing is priced the way it is relative to its competitors. Focus on the key factors provided.

    **Analysis Data:**

    **Key Factor: Size & Capacity**
    - Query Listing: +15% price impact (2 bedrooms, 4 guests)
    - Competitor 1: +25% price impact (3 bedrooms, 6 guests)
    - Competitor 2: +5% price impact (1 bedroom, 2 guests)

    **Key Factor: Description**
    - Query Listing: -9% price impact (Description: "Simple room in a shared apartment.")
    - Competitor 1: +5% price impact (Description: "Luxury penthouse with stunning skyline views.")
    - Competitor 2: -11% price impact (Description: "Cozy basement room, budget-friendly.")
    ```

* **Sample LLM Output:**

    "This listing's price is heavily influenced by its size and description when compared to its closest competitors.

    In terms of **Size & Capacity**, its two bedrooms give it a significant **+15% price advantage** over smaller studio apartments in the area. However, it is priced lower than larger three-bedroom units, which command an even higher premium.

    The listing's **Description** currently results in a **-9% price disadvantage**. The language "Simple room in a shared apartment" is perceived as less valuable than descriptions that highlight luxury features like "stunning skyline views," but more appealing than budget-focused terms like "basement room." Improving the descriptive language could help close this price gap."
