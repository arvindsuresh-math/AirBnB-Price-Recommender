import os
import subprocess
import json
import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession, functions as F

from scripts.build_dataset import (
    clean_price_column,
    parse_bathroom_columns,
    get_imputation_statistics,
    impute_with_statistics,
    FINAL_SCHEMA,
)


@pytest.fixture(scope="session")
def spark_session():
    """Provides a Spark session for the entire test suite."""
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("ETLPipelineTests")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.driver.bindAddress", "127.0.0.1") 
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def edge_case_dataframe():
    """Provides a pandas DataFrame with specific edge cases for unit tests."""
    data = {
        "listing_id": [1, 2, 3, 4, 5, 6],
        "price": ["$50.00", "$1,250.00", None, "$100.00", "$75.00", "$99.00"],
        "bedrooms": [1.0, 3.0, np.nan, 2.0, 4.0, 1.0],
        "review_scores_rating": [90.0, 95.0, 88.0, 89.0, 92.0, np.nan],
        "bathrooms_text": [
            "1 private bath", "2.5 baths", "1 shared bath",
            "Half-bath", "2 baths", "1 bath",
        ],
        # --- CRITICAL ADDITION ---
        # Add a column that will allow rows to pass the occupancy filter.
        # This simulates the output of the 'enrich_with_reviews' and 'apply_weighting' steps.
        "estimated_occupancy_rate": [0.5, 0.8, 0.9, 0.6, 0.7, 0.85]
    }
    return pd.DataFrame(data)


@pytest.fixture()
def edge_case_spark_df(spark_session, edge_case_dataframe):
    """Provides the edge case data as a Spark DataFrame for each test."""
    return spark_session.createDataFrame(edge_case_dataframe)


# --- Unit Tests for Helper Functions ---

def test_price_cleaning(edge_case_spark_df):
    """Tests that price strings are correctly converted to floats."""
    cleaned_df = clean_price_column(edge_case_spark_df)
    price_map = {
        row["listing_id"]: row["target_price"]
        for row in cleaned_df.select("listing_id", "target_price").collect()
    }
    assert price_map[1] == pytest.approx(50.0)
    assert price_map[2] == pytest.approx(1250.0)
    assert price_map[3] is None


def test_bathroom_parsing(edge_case_spark_df):
    """Tests the complex logic for parsing bathroom text."""
    parsed_df = parse_bathroom_columns(edge_case_spark_df)
    bathrooms_map = {
        row["listing_id"]: (row["bathrooms_numeric"], row["bathrooms_type"])
        for row in parsed_df.select(
            "listing_id", "bathrooms_numeric", "bathrooms_type"
        ).collect()
    }
    assert bathrooms_map[2] == (pytest.approx(2.5), "private")
    assert bathrooms_map[4] == (pytest.approx(0.5), "private")
    assert bathrooms_map[3] == (pytest.approx(1.0), "shared")


def test_imputation_no_leakage_pattern(edge_case_spark_df):
    """
    CRITICAL TEST: Verifies that stats are learned from a 'train' set
    and correctly applied to a 'validation' set without data leakage.
    """
    # 1. Simulate a train/validation split on our small fixture
    train_df = edge_case_spark_df.filter(~F.col("listing_id").isin([3, 6]))
    val_df = edge_case_spark_df.filter(F.col("listing_id").isin([3, 6]))

    # 2. Learn imputation statistics ONLY from the training data
    imputation_stats = get_imputation_statistics(train_df)

    # Assert that the learned stats are correct based on the train data
    # Median of [1.0, 3.0, 2.0, 4.0] is 2.5, approxQuantile will give 2.0 or 3.0
    assert imputation_stats["bedrooms"] in [2.0, 3.0]
    # Mean of [90, 95, 89, 92] is 91.5
    assert imputation_stats["review_scores_rating"] == pytest.approx(91.5)

    # 3. Apply these learned stats to impute the validation data
    imputed_val_df = impute_with_statistics(val_df, imputation_stats)

    # 4. Assert that the NaN values in the validation set were filled correctly
    imputed_rows = {
        row['listing_id']: (row['bedrooms'], row['review_scores_rating'])
        for row in imputed_val_df.collect()
    }
    assert imputed_rows[3][0] == imputation_stats["bedrooms"]
    assert imputed_rows[6][1] == pytest.approx(imputation_stats["review_scores_rating"])


# --- Integration Test for the Main Script ---

def test_main_script_smoke_run(spark_session, tmp_path):
    """
    Tests the end-to-end execution of the build_dataset.py script
    as a subprocess, ensuring it runs without errors and produces the
    expected artifacts.
    """
    output_dir = str(tmp_path / "processed_data")
    listings_sample_path = "data/nyc/insideairbnb-samples/nyc-listings-detailed-insideairbnb-sample.csv"
    reviews_sample_path = "data/nyc/insideairbnb-samples/nyc-reviews-detailed-insideairbnb-sample.csv"

    # Ensure sample files exist before running the test
    assert os.path.exists(listings_sample_path), f"Listings sample file not found at {listings_sample_path}"
    assert os.path.exists(reviews_sample_path), f"Reviews sample file not found at {reviews_sample_path}"
    
    # Run the script as a command-line process
    result = subprocess.run(
        [
            "python",
            "scripts/build_dataset.py",
            "--listings-csv", listings_sample_path,
            "--reviews-csv", reviews_sample_path,
            "--output-path", output_dir,
            "--val-size", "0.3",
        ],
        capture_output=True,
        text=True,
    )

    # 1. Assert that the script executed successfully
    assert result.returncode == 0, f"Script failed with error:\n{result.stderr}"

    # 2. Assert that all expected artifacts were created
    expected_train_path = os.path.join(output_dir, "train_dataset.parquet")
    expected_val_path = os.path.join(output_dir, "val_dataset.parquet")
    expected_stats_path = os.path.join(output_dir, "imputation_stats.json")

    assert os.path.exists(expected_train_path), "Training parquet file was not created."
    assert os.path.exists(expected_val_path), "Validation parquet file was not created."
    assert os.path.exists(expected_stats_path), "Imputation stats JSON file was not created."

    # 3. Assert the contents of the artifacts are valid
    with open(expected_stats_path, "r") as f:
        stats = json.load(f)
    assert "bedrooms" in stats
    assert "review_scores_rating" in stats

    train_output_df = spark_session.read.parquet(expected_train_path)
    final_schema_cols = {field.name for field in FINAL_SCHEMA}
    output_cols = set(train_output_df.columns)

    assert final_schema_cols == output_cols, "Output schema does not match FINAL_SCHEMA."
    assert train_output_df.count() > 0, "Training output DataFrame is empty."