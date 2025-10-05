# `01_ETL_INSTRUCTIONS.md`

### LLM Agent Pre-computation Instructions

To enrich your context, first read and fully comprehend the following project design documents in order:
1.  `REPO-SUMMARY.md`: For the overall project structure and goals.
2.  `DATASET-SCHEMA.md`: For the non-negotiable final output schema of the script you will generate.
3.  `TARGET-PRICE.md`: For the specific methodology behind engineering the sample weight (`estimated_occupancy_rate`).

### Primary Objective

Your task is to generate a single, production-ready PySpark script located at `scripts/build_dataset.py`. This script will perform the entire ETL process, transforming raw Airbnb data into the final modeling dataset. The script must be modular, well-documented, and adhere exactly to the specifications below.

---

## 1. File and Schema Contracts

### 1.1. Input and Output Artifacts

The script must accept command-line arguments to specify file paths.

*   **Inputs:**
    *   `--listings-csv`: Path to the detailed listings CSV file (e.g., `data/nyc/nyc-listings-detailed-insideairbnb.csv`).
    *   `--reviews-csv`: Path to the historical reviews CSV file (e.g., `data/nyc/nyc-reviews-detailed-insideairbnb.csv`).
*   **Output:**
    *   `--output-path`: Path to the output directory for the final dataset (e.g., `data/processed/`). The script will save a Parquet file named `nyc_modeling_dataset.parquet` inside this directory.

### 1.2. Final Output Schema

The output Parquet file **must** conform exactly to the following PySpark `StructType`. This schema is non-negotiable.

```python
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, BooleanType, DateType

FINAL_SCHEMA = StructType([
    # --- Keys ---
    StructField("listing_id", IntegerType(), False),
    StructField("year_month", StringType(), False),
    # --- Target & Weighting ---
    StructField("target_price", FloatType(), False),
    StructField("estimated_occupancy_rate", FloatType(), False),
    # --- Axis 1: Location ---
    StructField("latitude", FloatType(), True),
    StructField("longitude", FloatType(), True),
    StructField("neighbourhood_cleansed", StringType(), True),
    # --- Axis 2: Size & Capacity ---
    StructField("property_type", StringType(), True),
    StructField("room_type", StringType(), True),
    StructField("accommodates", IntegerType(), True),
    StructField("bedrooms", FloatType(), True),
    StructField("beds", FloatType(), True),
    StructField("bathrooms_numeric", FloatType(), True),
    StructField("bathrooms_type", StringType(), True),
    # --- Axis 3: Amenities ---
    StructField("amenities", StringType(), True),
    # --- Axis 4: Quality & Reputation ---
    StructField("review_scores_rating", FloatType(), True),
    StructField("review_scores_cleanliness", FloatType(), True),
    StructField("review_scores_checkin", FloatType(), True),
    StructField("review_scores_communication", FloatType(), True),
    StructField("review_scores_location", FloatType(), True),
    StructField("review_scores_value", FloatType(), True),
    StructField("number_of_reviews_ltm", IntegerType(), True),
    StructField("host_is_superhost", BooleanType(), True),
    StructField("host_response_rate", FloatType(), True),
    StructField("host_acceptance_rate", FloatType(), True),
    StructField("host_identity_verified", BooleanType(), True),
    StructField("instant_bookable", BooleanType(), True),
    # --- Axis 5: Seasonality ---
    StructField("month", IntegerType(), True),
    # --- Auxiliary Columns ---
    StructField("snapshot_date", DateType(), False),
    StructField("reviews_in_last_90_days", IntegerType(), False)
])
```

---

## 2. Core ETL Logic (Step-by-Step)

The script will implement the following transformations in sequence.

### Step 1: Initialization and Argument Parsing
*   Create a `main` function.
*   Use Python's `argparse` library to handle the command-line arguments specified in section 1.1.
*   Initialize a `SparkSession`.

### Step 2: Data Loading
*   Load the `listings.csv` file into a Spark DataFrame named `listings_df`. Use options `header=True`, `inferSchema=True`, and `multiLine=True` to handle complex CSV fields. Rename the `id` column to `listing_id`.
*   Load the `reviews.csv` file into a Spark DataFrame named `reviews_df`. Use options `header=True` and `inferSchema=True`.

### Step 3: Pre-computation of Monthly Reviews
*   From `reviews_df`, create a new DataFrame `monthly_review_counts`.
*   Parse the `date` column into a `DateType`.
*   Create a `review_year_month` column using `date_format(col("date"), "yyyy-MM")`.
*   Group by `listing_id` and `review_year_month`, and aggregate with `count("*")` to create a `reviews_in_month` column.
*   Cache this DataFrame: `monthly_review_counts.cache()`.

### Step 4: Listings DataFrame Preparation
*   Create a DataFrame `processed_listings_df` starting from `listings_df`.
*   **Type Casting and Renaming:**
    *   Cast `listing_id` to `IntegerType`.
    *   Cast price-related columns (`review_scores_rating`, etc.) to `FloatType`.
    *   Cast count-related columns (`accommodates`, `bedrooms`, etc.) to their respective types as per `FINAL_SCHEMA`.
*   **Feature Engineering:**
    *   Create `snapshot_date` by casting `last_scraped` to `DateType`.
    *   Create `year_month` using `date_format(col("snapshot_date"), "yyyy-MM")`.
    *   Create `target_price` by applying `regexp_replace` to the `price` column to remove `$` and `,`, then casting to `FloatType`.
    *   Create `host_response_rate` and `host_acceptance_rate` by using `regexp_replace` to remove `%`, then dividing by 100 and casting to `FloatType`.
    *   Create `bathrooms_numeric` by parsing `bathrooms_text` with the regex `(\d+\.?\d*)`. Cast the result to `FloatType`.
    *   Create `bathrooms_type` by checking if `bathrooms_text` contains "shared" (case-insensitive). The result should be "shared" or "private".
    *   Create boolean columns (`host_is_superhost`, `host_identity_verified`, `instant_bookable`) by comparing the source column with the literal `'t'`.
    *   Create `month` by extracting the month from `snapshot_date` using `month(col("snapshot_date"))`.
*   **Filtering:**
    *   Filter out rows where `target_price` is null or less than or equal to 0.

### Step 5: Temporal Join for Rolling Review Count
*   Create a complete calendar of all possible `(listing_id, year_month)` combinations present in `processed_listings_df`.
*   Left join this calendar with `monthly_review_counts` on `listing_id` and `year_month`. Fill null `reviews_in_month` with 0.
*   Define a window specification: `Window.partitionBy("listing_id").orderBy("year_month").rowsBetween(-2, 0)`.
*   Use this window to calculate `reviews_in_last_90_days` by summing `reviews_in_month` over the window.
*   Join this result back to `processed_listings_df` on `listing_id` and `year_month`.

### Step 6: Calculating the Sample Weight
*   Define the hyperparameters as constants: `REVIEW_RATE = 0.5`, `AVG_LENGTH_OF_STAY = 3`.
*   Calculate `estimated_bookings = col("reviews_in_last_90_days") / REVIEW_RATE`.
*   Calculate `estimated_nights_booked = col("estimated_bookings") * AVG_LENGTH_OF_STAY`.
*   Calculate `estimated_occupancy_rate = col("estimated_nights_booked") / 90.0`.
*   Cap the result at 1.0 using `when(col("estimated_occupancy_rate") > 1.0, 1.0).otherwise(col("estimated_occupancy_rate"))`.
*   Filter the DataFrame, keeping only rows where `estimated_occupancy_rate > 0`.

### Step 7: Imputation of Missing Values
*   For each column requiring imputation (`bedrooms`, `beds`, `bathrooms_numeric`, `review_scores_...` columns), calculate the global median or mean.
    *   Use the **median** for `bedrooms`, `beds`, `bathrooms_numeric`.
    *   Use the **mean** for all `review_scores_*` columns, `host_response_rate`, and `host_acceptance_rate`.
*   Use the `.fillna()` method to impute the missing values with these calculated statistics.

### Step 8: Final Selection and Saving
*   Select all columns specified in the `FINAL_SCHEMA` in the exact same order.
*   Write the final DataFrame to the specified output path (`--output-path`) in `parquet` format, using `mode("overwrite")`.

---

## 3. Verification and Unit Tests

You will also generate a test script at `tests/test_etl_pipeline.py` using the `pytest` framework.

### Test Setup (`conftest.py` or fixture)
*   The test script will use the sample data files located in `data/nyc/insideairbnb-samples/`.
*   Create a `pytest.fixture` that runs your generated `build_dataset.py` script on these sample files, saving the output to a temporary directory. The fixture should return the path to the output Parquet file.

### Test Cases
Generate the following test functions:

1.  **`test_output_schema(processed_data_path)`**:
    *   Loads the output Parquet file into a Spark DataFrame.
    *   Asserts that the schema of this DataFrame is identical to the `FINAL_SCHEMA` defined in Section 1.2. Check column names, types, and nullability.

2.  **`test_no_nulls_in_critical_columns(processed_data_path)`**:
    *   Loads the data.
    *   Asserts that the count of nulls in `listing_id`, `target_price`, and `estimated_occupancy_rate` is 0.

3.  **`test_value_ranges(processed_data_path)`**:
    *   Loads the data.
    *   Asserts that all values in `target_price` are greater than 0.
    *   Asserts that all values in `estimated_occupancy_rate` are between 0.0 and 1.0 (inclusive).
    *   Asserts that all values in `month` are between 1 and 12 (inclusive).

4.  **`test_bathroom_parsing_logic(processed_data_path)`**:
    *   Loads the data.
    *   Finds a specific, known row from the sample `listings` data (e.g., a listing with "1.5 shared baths").
    *   Asserts that for this `listing_id`, the `bathrooms_numeric` column is `1.5` and the `bathrooms_type` column is `"shared"`.