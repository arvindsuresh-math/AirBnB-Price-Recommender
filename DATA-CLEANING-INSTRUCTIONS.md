# DATA-CLEANING-INSTRUCTIONS.md

## 1. Objective

This specification details the creation of a Python script responsible for the complete data cleaning and pre-processing pipeline. The script will produce the final modeling dataset for a **single month** of NYC Airbnb data. It will ingest raw data sources, engineer features and the target variable according to the project's design documents (`TARGET-PRICE.md`, `FEATURE-AXES.md`), and output a single, versioned Parquet file along with necessary metadata artifacts.

## 2. File and Path Specifications

### 2.1. Script Location and Name

* **Location:** `/Fall-2025-Team-Big-Data/scripts/`
* **Name:** `01_create_modeling_dataset.py`

### 2.2. Input Files

* **Listings Data (Single Month):** `/Fall-2025-Team-Big-Data/data/nyc/nyc-listings-detailed-insideairbnb.csv`
* **Reviews Data (Historical):** `<REVIEWS_CSV_PATH>` (This is an external, user-provided path to the full historical `reviews.csv` file. The script should accept this path as a command-line argument.)

### 2.3. Output Artifacts

The script MUST produce two output files in a new directory.

* **Output Directory:** `/Fall-2025-Team-Big-Data/data/processed/nyc/`
  * *The script must create this directory if it does not exist.*

* **Output 1: Cleaned Modeling Data (Parquet File)**
  * **Name:** `nyc-modeling-data-YYYY-MM.parquet`
  * **Logic:** The `YYYY-MM` in the filename MUST be dynamically determined from the `last_scraped` date in the input listings file. For example, if the scrape date is `2025-08-03`, the filename will be `nyc-modeling-data-2025-08.parquet`.

* **Output 2: Feature Pre-processing Artifacts (JSON File)**
  * **Name:** `nyc-feature-artifacts-YYYY-MM.json`
  * **Logic:** The `YYYY-MM` must match the Parquet file.
  * **Purpose:** This file stores all the necessary information for pre-processing new data at inference time. It MUST contain:
    * **Imputation Values:** A dictionary mapping each column that was imputed to the value used for imputation (e.g., `{"bedrooms": 1.0, "review_scores_rating": 4.65}`).
    * **Categorical Vocabularies:** For every feature that will be treated as categorical by the model (`neighbourhood_cleansed`, `property_type`, `bedrooms`, `beds`, etc.), a dictionary mapping each unique string/value to a unique integer index (starting from 0).

## 3. Step-by-Step Execution Plan

The script should be implemented using the Pandas library and should follow this exact sequence.

### Step 1: Argument Parsing and Setup

1. Use Python's `argparse` library to accept one required command-line argument: `--reviews-path`.
2. Define all input and output paths as specified in Section 2.
3. Read the `last_scraped` date from the first row of the listings CSV to determine the `YYYY-MM` string for output filenames.

### Step 2: Pre-process Historical Reviews Data

(Same as before, resulting in `reviews_monthly_df`)

### Step 3: Load and Prepare Main Listings Data

(Same as before, resulting in a main `listings_df`)

### Step 4: Engineer Occupancy and Filter Data

(Same as before, resulting in a filtered `listings_df` with the `estimated_occupancy_rate` column)

### Step 5: Process All Columns for Final Schema

Process the columns of `listings_df` to match the final schema defined in `DATASET-SCHEMA.md`. **During this step, you must collect the imputation values and create the vocabulary mappings for the artifacts file.**

* **For Imputation:** When you calculate a mean or median to fill NaNs (e.g., for `bedrooms`), store this value in a dictionary.
* **For Vocabularies:** For each categorical column, get the list of unique values, sort them, and create a dictionary mapping each value to its index (`{value: i for i, value in enumerate(sorted_unique_values)}`).

### Step 6: Finalize and Save Artifacts

1. **Select Final Columns:** Create the final DataFrame by selecting **only** the columns specified in `DATASET-SCHEMA.md` and ensuring they are in the correct order.
2. **Save Modeling Data:** Save this final DataFrame to the Parquet file path defined in Section 2.3. Use `index=False`.
3. **Save Artifacts File:**
    * Create a single dictionary containing two top-level keys: `"imputation_values"` and `"categorical_vocabularies"`.
    * Populate this dictionary with the values collected in Step 5.
    * Save this dictionary to the JSON file path defined in Section 2.3 using `json.dump` with an indent for readability.

## 4. Column-by-Column Processing Specification

This table details the creation logic for every column in the final output file.

| Column Name | Final Data Type | Processing Logic |
| :--- | :--- | :--- |
| **--- Keys ---** | | |
| `listing_id` | `int64` | Source: `id` from listings.csv. Cast to integer. |
| `year_month` | `string` | **Computed:** Format `last_scraped` date as 'YYYY-MM'. |
| **--- Target & Weighting ---** | | |
| `target_price` | `float64` | **Computed:** Parse `price` column from listings.csv by removing '$' and ',' characters and casting to float. |
| `estimated_occupancy_rate` | `float64` | **Computed:** As per the formula in Step 3.2. |
| **--- Axis 1: Location ---** | | |
| `latitude` | `float64` | Source: `latitude` from listings.csv. |
| `longitude` | `float64` | Source: `longitude` from listings.csv. |
| `neighbourhood_cleansed` | `string` | Source: `neighbourhood_cleansed` from listings.csv. |
| **--- Axis 2: Size & Capacity ---** | | |
| `property_type` | `string` | Source: `property_type` from listings.csv. |
| `room_type` | `string` | Source: `room_type` from listings.csv. |
| `accommodates` | `int64` | Source: `accommodates` from listings.csv. Cast to integer. |
| `bedrooms` | `float64` | Source: `bedrooms` from listings.csv. Impute NaNs with the column median. |
| `beds` | `float64` | Source: `beds` from listings.csv. Impute NaNs with the column median. |
| `bathrooms_numeric` | `float64` | **Computed:** Parse `bathrooms_text` string using regex to extract the first numerical value (e.g., '1.5 shared baths' -> 1.5). Impute NaNs with the column median. |
| `bathrooms_type` | `string` | **Computed:** Parse `bathrooms_text`. If it contains 'private', set to 'private'. If it contains 'shared', set to 'shared'. Otherwise, set to 'unknown'. |
| **--- Axis 3: Amenities ---** | | |
| `amenities` | `string` | Source: `amenities` from listings.csv. Keep as the raw string. |
| **--- Axis 4: Quality & Reputation ---** | | |
| `review_scores_rating`, `..._cleanliness`, `..._checkin`, `..._communication`, `..._location`, `..._value` | `float64` | Source from listings.csv. Impute NaNs with the column mean for each respective column. (6 columns total). |
| `number_of_reviews` | `int64` | Source: `number_of_reviews` from listings.csv. Cast to integer. |
| `host_is_superhost` | `bool` | **Computed:** Map 't' to `True` and 'f' to `False` from the `host_is_superhost` column. |
| `host_response_rate` | `float64` | **Computed:** Parse the string by removing '%' and dividing by 100. Impute NaNs with the column median. |
| `host_acceptance_rate` | `float64` | **Computed:** Parse the string by removing '%' and dividing by 100. Impute NaNs with the column median. |
| `host_identity_verified`| `bool` | **Computed:** Map 't' to `True` and 'f' to `False` from the `host_identity_verified` column. |
| `instant_bookable` | `bool` | **Computed:** Map 't' to `True` and 'f' to `False` from the `instant_bookable` column. |
| **--- Axis 5: Seasonality ---** | | |
| `month` | `int64` | **Computed:** Extract the month number (1-12) from the `last_scraped` date. |
| **--- Auxiliary Columns ---** | | |
| `snapshot_date` | `datetime64[ns]`| Source: `last_scraped` from listings.csv. Convert to datetime object. |
| `reviews_in_last_90_days`| `int64` | **Computed:** As per Step 3.1. |

## 5. Final Output

The final DataFrame must contain **only** the columns listed in the table above, in that order. The script should then save this DataFrame to the specified output path in Parquet format. Ensure the output directory is created before saving.
