import argparse
import os
import json
from typing import Dict, List

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# --- SCHEMA AND CONSTANTS ---
FINAL_SCHEMA = StructType([
    StructField("listing_id", IntegerType(), False),
    StructField("year_month", StringType(), False),
    StructField("target_price", FloatType(), False),
    StructField("estimated_occupancy_rate", FloatType(), False),
    StructField("latitude", FloatType(), True),
    StructField("longitude", FloatType(), True),
    StructField("neighbourhood_cleansed", StringType(), True),
    StructField("property_type", StringType(), True),
    StructField("room_type", StringType(), True),
    StructField("accommodates", IntegerType(), True),
    StructField("bedrooms", FloatType(), True),
    StructField("beds", FloatType(), True),
    StructField("bathrooms_numeric", FloatType(), True),
    StructField("bathrooms_type", StringType(), True),
    StructField("amenities", StringType(), True),
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
    StructField("month", IntegerType(), True),
    StructField("snapshot_date", DateType(), False),
    StructField("reviews_in_last_90_days", IntegerType(), False),
])

REVIEW_RATE = 0.5
AVG_LENGTH_OF_STAY = 3


# --- HELPER FUNCTIONS ---
def init_spark(app_name: str = "BuildDatasetETL") -> SparkSession:
    """Initializes and returns a Spark session."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.bindAddress", "127.0.0.1")  # <-- THE FIX
        .getOrCreate()
    )


def load_listings(spark: SparkSession, listings_path: str) -> DataFrame:
    """Loads the listings CSV file into a Spark DataFrame."""
    listings_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(listings_path)
    )
    if "id" in listings_df.columns:
        listings_df = listings_df.withColumnRenamed("id", "listing_id")
    return listings_df


def load_reviews(spark: SparkSession, reviews_path: str) -> DataFrame:
    """Loads the reviews CSV file into a Spark DataFrame."""
    return (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)  
        .option("escape", '"') 
        .csv(reviews_path)
    )


def compute_monthly_review_counts(reviews_df: DataFrame) -> DataFrame:
    """Aggregates reviews to get a monthly count per listing."""
    review_dates = reviews_df.withColumn("review_date", F.to_date("date"))
    monthly_review_counts = (
        review_dates.filter(F.col("review_date").isNotNull())
        .withColumn("review_year_month", F.date_format("review_date", "yyyy-MM"))
        .groupBy("listing_id", "review_year_month")
        .agg(F.count("*").alias("reviews_in_month"))
        .cache()
    )
    return monthly_review_counts


def clean_price_column(
    df: DataFrame, source_column: str = "price", output_column: str = "target_price"
) -> DataFrame:
    """Cleans a currency string column and converts it to a float."""
    cleaned = F.regexp_replace(F.col(source_column), r"[$,]", "")
    return df.withColumn(output_column, cleaned.cast(FloatType()))


def convert_percentage_to_ratio(df: DataFrame, column_name: str) -> DataFrame:
    """Converts a percentage string to a float ratio."""
    cleaned = F.regexp_replace(F.col(column_name), "%", "")
    ratio = cleaned.cast(FloatType()) / F.lit(100.0)
    return df.withColumn(column_name, ratio)


def parse_bathroom_columns(df: DataFrame) -> DataFrame:
    """Parses the 'bathrooms_text' column into numeric and type columns."""
    lower_text = F.lower(F.col("bathrooms_text"))
    numeric_match = F.regexp_extract(F.col("bathrooms_text"), r"(\d+\.?\d*)", 1)
    df = df.withColumn(
        "bathrooms_numeric",
        F.when(F.col("bathrooms_text").isNull(), None)
        .when(lower_text.contains("half"), F.lit(0.5))
        .when(F.length(numeric_match) > 0, numeric_match.cast(FloatType()))
        .otherwise(None),
    )
    df = df.withColumn(
        "bathrooms_type",
        F.when(F.col("bathrooms_text").isNull(), None)
        .when(lower_text.contains("shared"), F.lit("shared"))
        .otherwise(F.lit("private")),
    )
    return df


def prepare_listings_dataframe(listings_df: DataFrame) -> DataFrame:
    """Applies all cleaning, parsing, and feature engineering to the raw listings data."""
    df = listings_df
    df = clean_price_column(df)
    if "host_response_rate" in df.columns:
        df = convert_percentage_to_ratio(df, "host_response_rate")
    if "host_acceptance_rate" in df.columns:
        df = convert_percentage_to_ratio(df, "host_acceptance_rate")
    df = parse_bathroom_columns(df)

    # Apply various type castings
    df = df.withColumn("listing_id", F.col("listing_id").cast(IntegerType()))
    df = df.withColumn("snapshot_date", F.to_date("last_scraped"))
    df = df.withColumn("year_month", F.date_format("snapshot_date", "yyyy-MM"))
    df = df.withColumn("month", F.month("snapshot_date"))
    
    # Cast boolean-like 't'/'f' strings to actual Booleans
    for col_name in ["host_is_superhost", "host_identity_verified", "instant_bookable"]:
        if col_name in df.columns:
            df = df.withColumn(col_name, (F.col(col_name) == F.lit("t")).cast(BooleanType()))
            
    df = df.filter(F.col("target_price").isNotNull() & (F.col("target_price") > 0))
    return df


def enrich_with_reviews(
    processed_listings_df: DataFrame, monthly_review_counts: DataFrame
) -> DataFrame:
    """
    Correctly calculates a 90-day rolling review count and enriches the listings data.
    """
    # 1. Create a complete calendar of all (listing_id, year_month) pairs that exist
    #    in either the listings or the reviews to build a full timeline.
    listing_months = processed_listings_df.select("listing_id", "year_month").distinct()
    review_months = monthly_review_counts.select(
        "listing_id", F.col("review_year_month").alias("year_month")
    ).distinct()
    
    full_calendar = listing_months.union(review_months).distinct()

    # 2. Join the full calendar with the monthly review counts. This ensures all months
    #    (even those with no reviews) are present for the window function.
    calendar_with_counts = full_calendar.join(
        monthly_review_counts,
        (full_calendar["listing_id"] == monthly_review_counts["listing_id"])
        & (full_calendar["year_month"] == monthly_review_counts["review_year_month"]),
        "left",
    ).select(
        full_calendar["listing_id"],
        full_calendar["year_month"],
        F.col("reviews_in_month"),
    )
    calendar_with_counts = calendar_with_counts.fillna({"reviews_in_month": 0})

    # 3. Apply the rolling window function on the complete timeline.
    #    This correctly sums the counts from the preceding 2 months + the current month.
    review_window = Window.partitionBy("listing_id").orderBy("year_month").rowsBetween(-2, 0)
    rolling_counts_df = calendar_with_counts.withColumn(
        "reviews_in_last_90_days", F.sum("reviews_in_month").over(review_window)
    )

    # 4. Finally, join the original listings data to this new DataFrame which now
    #    contains the correctly calculated rolling sum for every month.
    enriched_df = processed_listings_df.join(
        rolling_counts_df.select("listing_id", "year_month", "reviews_in_last_90_days"),
        ["listing_id", "year_month"],
        "inner",  # Use inner join to only keep listings we have snapshot data for
    )
    
    return enriched_df


def apply_weighting(enriched_df: DataFrame) -> DataFrame:
    """Calculates the estimated occupancy rate to be used as a sample weight."""
    df = enriched_df.withColumn(
        "estimated_nights_booked",
        (F.col("reviews_in_last_90_days") / F.lit(REVIEW_RATE)) * F.lit(AVG_LENGTH_OF_STAY),
    )
    df = df.withColumn(
        "estimated_occupancy_rate",
        F.col("estimated_nights_booked") / F.lit(90.0),
    )
    df = df.withColumn(
        "estimated_occupancy_rate",
        F.when(F.col("estimated_occupancy_rate") > 1.0, F.lit(1.0)).otherwise(
            F.col("estimated_occupancy_rate")
        ),
    )
    df = df.filter(F.col("estimated_occupancy_rate") > 0)
    df = df.drop("estimated_nights_booked")
    return df


def get_imputation_statistics(df: DataFrame) -> Dict[str, float]:
    """Calculates mean/median statistics for imputation from a DataFrame (TRAINING DATA ONLY)."""
    median_columns = ["bedrooms", "beds", "bathrooms_numeric"]
    mean_columns = [
        "review_scores_rating", "review_scores_cleanliness", "review_scores_checkin",
        "review_scores_communication", "review_scores_location", "review_scores_value",
        "host_response_rate", "host_acceptance_rate",
    ]
    imputation_stats: Dict[str, float] = {}
    
    for column in median_columns:
        if column in df.columns:
            quantiles = df.approxQuantile(column, [0.5], 0.01)
            if quantiles and quantiles[0] is not None:
                imputation_stats[column] = float(quantiles[0])

    for column in mean_columns:
        if column in df.columns:
            mean_value = df.select(F.avg(F.col(column))).first()[0]
            if mean_value is not None:
                imputation_stats[column] = float(mean_value)
                
    return imputation_stats


def impute_with_statistics(df: DataFrame, imputation_stats: Dict[str, float]) -> DataFrame:
    """Fills missing values in a DataFrame using pre-calculated statistics."""
    return df.fillna(imputation_stats)


def finalize_dataset(df: DataFrame, imputation_stats: Dict[str, float]) -> DataFrame:
    """Applies final imputation and selects/casts columns to conform to the final schema."""
    imputed_df = impute_with_statistics(df, imputation_stats)
    
    selected_columns: List[Column] = []
    for field in FINAL_SCHEMA:
        if field.name in imputed_df.columns:
            selected_columns.append(F.col(field.name).cast(field.dataType).alias(field.name))
            
    return imputed_df.select(*selected_columns)


def save_artifacts(
    train_df: DataFrame, val_df: DataFrame, imputation_stats: Dict[str, float], output_path: str
) -> None:
    """Saves the processed datasets and imputation statistics."""
    os.makedirs(output_path, exist_ok=True)
    
    train_df.write.mode("overwrite").parquet(os.path.join(output_path, "train_dataset.parquet"))
    val_df.write.mode("overwrite").parquet(os.path.join(output_path, "val_dataset.parquet"))
    
    with open(os.path.join(output_path, "imputation_stats.json"), "w") as f:
        json.dump(imputation_stats, f, indent=4)
        
    print(f"Artifacts successfully saved to {output_path}")


def main() -> None:
    """Main ETL orchestration script."""
    parser = argparse.ArgumentParser(description="Build modeling dataset from Airbnb data")
    parser.add_argument("--listings-csv", required=True, dest="listings_csv")
    parser.add_argument("--reviews-csv", required=True, dest="reviews_csv")
    parser.add_argument("--output-path", required=True, dest="output_path")
    parser.add_argument("--val-size", type=float, default=0.2, dest="val_size")
    args = parser.parse_args()

    spark = init_spark()
    try:
        listings_df = load_listings(spark, args.listings_csv)
        reviews_df = load_reviews(spark, args.reviews_csv)

        monthly_review_counts = compute_monthly_review_counts(reviews_df)
        processed_listings_df = prepare_listings_dataframe(listings_df)
        enriched_df = enrich_with_reviews(processed_listings_df, monthly_review_counts)
        weighted_df = apply_weighting(enriched_df)
        
        train_df, val_df = weighted_df.randomSplit(
            [(1.0 - args.val_size), args.val_size], seed=42
        )
        
        imputation_stats = get_imputation_statistics(train_df)
        
        final_train_df = finalize_dataset(train_df, imputation_stats)
        final_val_df = finalize_dataset(val_df, imputation_stats)
        
        save_artifacts(final_train_df, final_val_df, imputation_stats, args.output_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()