# Schema for the Final Modeling Dataset

This document specifies the structure of the final, cleaned dataset that will be the direct input to our model. The entire data pre-processing pipeline (ETL) should be designed to produce a single table conforming to this schema.

The fundamental unit of analysis, or the "grain" of this table, is the **listing-month**. Each row represents a unique `(listing_id, year_month)` combination.

## Table Schema

| Column Name | Final Data Type | Creation Logic / Source |
| :--- | :--- | :--- |
| **--- Keys ---** | | |
| `listing_id` | `integer` | Sourced directly from `listings.csv`. |
| `year_month` | `string` | **Computed** from the `last_scraped` date in `listings.csv` (e.g., '2025-08'). |
| **--- Target & Weighting ---** | | |
| `target_price` | `float` | Sourced from the `price` column in `listings.csv` after cleaning (removing '$'). |
| `estimated_occupancy_rate`| `float` | **Computed** as a sample weight. Calculated from `reviews_in_last_90_days` using a formula with `review_rate` and `avg_length_of_stay` hyperparameters. Rows with a weight of 0 can be dropped. |
| **--- Axis 1: Location ---** | | |
| `latitude` | `float` | Sourced directly from `listings.csv`. |
| `longitude` | `float` | Sourced directly from `listings.csv`. |
| `neighbourhood_cleansed` | `string` | Sourced directly from `listings.csv`. To be treated as a categorical feature. |
| **--- Axis 2: Size & Capacity ---** | | |
| `property_type` | `string` | Sourced directly from `listings.csv`. To be treated as a categorical feature. |
| `room_type` | `string` | Sourced directly from `listings.csv`. To be treated as a categorical feature. |
| `accommodates` | `integer` | Sourced directly from `listings.csv`. |
| `bedrooms` | `float` | Sourced from `listings.csv`. Missing values will be imputed (e.g., with the median). |
| `beds` | `float` | Sourced from `listings.csv`. Missing values will be imputed (e.g., with the median). |
| `bathrooms_numeric` | `float` | **Computed** by parsing the `bathrooms_text` string from `listings.csv` to extract the numerical value. |
| `bathrooms_type` | `string` | **Computed** by parsing `bathrooms_text` to extract the type (e.g., 'private', 'shared'). To be treated as a categorical feature. |
| **--- Axis 3: Amenities ---** | | |
| `amenities` | `string` | Sourced directly from `listings.csv`. The raw string-formatted list is preserved for the model to process and embed. |
| **--- Axis 4: Quality & Reputation ---** | | |
| `review_scores_rating` | `float` | Sourced from `listings.csv`. Missing values will be imputed (e.g., with the mean/median). |
| `review_scores_cleanliness`| `float` | Sourced from `listings.csv`. Missing values will be imputed (e.g., with the mean/median). |
| `review_scores_checkin` | `float` | Sourced from `listings.csv`. Missing values will be imputed (e.g., with the mean/median). |
| `review_scores_communication`| `float` | Sourced from `listings.csv`. Missing values will be imputed (e.g., with the mean/median). |
| `review_scores_location` | `float` | Sourced from `listings.csv`. Missing values will be imputed (e.g., with the mean/median). |
| `review_scores_value` | `float` | Sourced from `listings.csv`. Missing values will be imputed (e.g., with the mean/median). |
| `number_of_reviews_ltm` | `integer` | Sourced directly from `listings.csv`. |
| `host_is_superhost` | `boolean` | Sourced from `listings.csv` after converting 't'/'f' to boolean. |
| `host_response_rate` | `float` | **Computed** by parsing the string from `listings.csv` (e.g., '95%') into a float (0.95). Missing values will be imputed. |
| `host_acceptance_rate` | `float` | **Computed** by parsing the string from `listings.csv` into a float. Missing values will be imputed. |
| `host_identity_verified`| `boolean` | Sourced from `listings.csv` after converting 't'/'f' to boolean. |
| `instant_bookable` | `boolean` | Sourced from `listings.csv` after converting 't'/'f' to boolean. |
| **--- Axis 5: Seasonality ---** | | |
| `month` | `integer` | **Computed** by extracting the month (1-12) from the `last_scraped` date. |
| **--- Auxiliary Columns (For ETL/Debugging) ---** | | |
| `snapshot_date` | `date` | The `last_scraped` date, preserved for reference and temporal joins. |
| `reviews_in_last_90_days`| `integer` | **Computed** by aggregating `reviews.csv` into monthly counts and then calculating a 3-month rolling sum for each listing. This column is the basis for the sample weight but is not a model feature. |
