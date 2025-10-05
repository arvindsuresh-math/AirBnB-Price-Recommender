# Explainable Pricing for Short-Term Rentals — Project Proposal (Updated)

## 1) Problem Statement

Short-term rental hosts lack an affordable, **explainable** pricing tool. Existing products are often opaque and costly. This project aims to build a practical, transparent system that gives hosts useful, defensible pricing guidance based on public data.

## 2) Proposed Solution (Web App)

A lightweight web app where a host provides their property details. The app then:

1. Produces a **recommended price for each month** of the year.
2. Provides a short, plain-English explanation that **decomposes the price** into intuitive parts (location, size, amenities, quality, seasonality).
3. Presents a **comparable set (comp set)** of similar listings to serve as market evidence for the recommendation.

## 3) Stakeholders & KPIs

**Stakeholders:** Individual hosts and small property managers.

**Primary KPIs (measured offline):**

* **Weighted Mean Absolute Error (WMAE):** Model performance metric, weighted by market activity.
* **Explanation Coverage:** Every recommendation is accompanied by a full price breakdown and top-K comparables.

## 4) Modeling Design & Strategy (High-Level)

We prioritize **interpretability** and structure the model around four principles:

**(i) Split the total into a sum (additivity).**
Recommended price is modeled as a **sum of contributions** from distinct “axes” (feature groups):
**price ≈** (location) + (size/capacity) + (amenities) + (quality) + (seasonality) + baseline.
This additive view lets us explain “what added how much.”

**(ii) Embed/encode each axis as a vector.**
Each axis is turned into a numeric vector representation (**embedding**). This is detailed in `EMBEDDINGS.md`.

**(iii) One small model per axis; sum the outputs.**
Each axis feeds a small model (**sub-network**) that outputs a **price contribution**. The final recommendation is the **sum** of these contributions. The model is trained end-to-end.

**(iv) Choice of “true” value (target): Market-Accepted Price via Weighted Learning.**
Instead of inferring a target from other listings, we model the listing's own observed price. To account for market acceptance, we **weight each training sample by its estimated occupancy rate.** A listing with high recent booking activity (proxied by review velocity) will have a stronger influence on the model than a listing with no activity. This directly teaches the model to prioritize prices that are proven to succeed in the market. This robust methodology is detailed in `TARGET-PRICE.md`.

## 5) Data Requirements (Final Dataset Shape)

We will build a **listing-month panel** per city with the following key columns:

**Keys:** `(listing_id, year_month)`

**Features by axis (See `FEATURE-AXES.md`):**

* **Location:** `latitude`, `longitude`, `neighbourhood_cleansed`.
* **Size/Capacity:** `property_type`, `room_type`, `accommodates`, `bedrooms`, `bathrooms_numeric`, etc.
* **Amenities:** Raw `amenities` string for embedding.
* **Quality/Reputation:** `review_scores_rating`, `number_of_reviews_ltm`, `host_is_superhost`, etc.
* **Seasonality:** `month`.

**Target & Weight:**

* `target_price`: The listing's own observed price from `listings.csv`.
* `estimated_occupancy_rate`: The **sample weight** calculated from recent review velocity, used to weight the loss function during training.

## 6) Pre-Processing & Cleaning

The ETL pipeline, built with PySpark, will ingest raw data and produce the final `listing-month` panel, including the critical `estimated_occupancy_rate` sample weight.

## 7) Proposed 5-Week Timeline (LLM-Accelerated)

This timeline leverages an LLM coding agent to generate the initial codebase from detailed instructions, allowing the team to focus on review, integration, and refinement.

### Week 1 — Foundations & Blueprints

* Finalize all project design documents (`SCHEMA`, `MODELING`, etc.).
* Write the complete, detailed set of LLM instruction markdowns (`01` through `06`).
* Generate and commit a small, schema-compliant dummy dataset to unblock parallel development.
* **Deliverables:** Finalized project design; a complete set of instruction markdowns; a dummy dataset.

### Week 2 — Code Generation & Unit Testing

* Feed instruction markdowns to the LLM agent to generate all Python scripts (`etl`, `dataset`, `model`, `train`, `deploy`, `app`).
* Feed instruction markdowns to generate all corresponding test scripts.
* Run all unit tests against the generated code using the dummy dataset and mocks. Debug and refine the generated code until all tests pass.
* **Deliverables:** A full, working codebase; a passing unit test suite.

### Week 3 — Integration & Model Training v1

* Run the generated ETL script on the full, multi-month dataset.
* Run the generated training script to produce the first trained model (`best_model.pth`) and feature processor (`feature_processor.joblib`).
* Analyze the model's performance and the learned price breakdowns for sanity.
* **Deliverables:** The final modeling dataset; Model v1 artifacts; initial evaluation report.

### Week 4 — Deployment & End-to-End Testing

* Use the generated deployment script to package the artifacts and deploy the model to a live AWS SageMaker endpoint.
* Connect the Streamlit application to the live endpoint.
* Perform end-to-end testing: enter data in the UI, confirm a valid prediction is returned and displayed correctly.
* **Deliverables:** A deployed SageMaker endpoint; a fully functional web application.

### Week 5 — Polish, Validation & Delivery

* Refine the Streamlit app's UI/UX.
* Finalize documentation, including the `README.md` and a final project report detailing the methodology, results, and limitations.
* Prepare a final presentation and demo of the application.
* **Deliverables:** A polished app demo; end-to-end reproducible pipeline; final model and report.
