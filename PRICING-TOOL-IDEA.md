# Explainable Pricing for Short-Term Rentals — Project Proposal (5-Week Plan)

## 1) Problem Statement

Short-term rental hosts lack an affordable, **explainable** pricing tool. Existing products are often opaque and costly. Public data (e.g., InsideAirbnb) is available as periodic snapshots, but nightly prices may be missing and calendar “unavailable” can mean *booked or host-blocked*. The goal is a practical, transparent system that remains honest about these limits while still giving hosts useful, defensible pricing guidance.

## 2) Proposed Solution (Web App)

A lightweight web app where a host provides property details (location, property/room type, bedrooms/bathrooms/capacity, amenities; optionally an existing listing ID). The app then:

1. Builds a **comparable set (comp set)** of nearby, similar listings and shows a ranked comparison.
2. Presents a side-by-side view of the host’s property vs. the closest competitors (attributes and historical price bands by month).
3. Produces a **recommended price for each month** of the year, together with a short, plain-English explanation that decomposes the price into intuitive parts (location, size, amenities, reviews, seasonality).

## 3) Stakeholders & KPIs

**Stakeholders:** individual hosts and small property managers.

**Primary KPIs (measured offline):**

- **Peer-band hit rate:** % of monthly recommendations that fall within the interquartile range (P25–P75) of the comp-set prices for that month.
- **Distance to peer median:** absolute or % deviation from the comp-set median price for that month.
- **Explanation coverage:** each recommendation includes a factorized breakdown and top K comparables with evidence (attributes and distances).

(If later adding offline counterfactual evaluation, it should be secondary and clearly caveated given support/identifiability limits.)

## 4) Modeling Design & Strategy (High-Level)

We prioritize **interpretability** and structure the model around four principles:

**(i) Split the total into a sum (additivity).**  
Recommended price is modeled as a **sum of contributions** from distinct “axes” (feature groups):  
**price ≈** (location) + (size/capacity) + (amenities) + (reviews) + (seasonality) + baseline.  
This additive view lets us explain “what added how much.”

**(ii) Embed/encode each axis as a vector.**  
Each axis is turned into a numeric vector representation (**embedding**). Examples:  

- Location → spatial cell/coordinates encoded as a vector.  
- Size/capacity → bedrooms, bathrooms, accommodates as a small numeric vector.  
- Amenities → multi-hot list transformed into an embedding vector.  
- Reviews → counts/scores as a numeric vector.  
- Seasonality → month encoded as cyclic features or a small learned vector.

**(iii) One small model per axis; sum the outputs.**  
Each axis feeds a small model (**sub-network**) that outputs a **price contribution** for that axis. The final recommendation is the **sum** of axis contributions (the **additive model**). Training is end-to-end so each axis learns to predict its own piece.

**(iv) Choice of “true” value (target): market-accepted median.**  
For each listing and month, the target is the **median price among its comp set on nights marked “unavailable”** in that month (a practical proxy for market-accepted prices). This reduces the impact of idiosyncratic host snapshots. We will be explicit that “unavailable” mixes booked and blocked.

## 5) Data Requirements (Final Dataset Shape)

We will build a **listing-month panel** per city:

**Keys:** `(listing_id, year_month)`

**Features by axis:**

- **Location:** latitude/longitude → spatial index (e.g., H3 at a fixed resolution); neighborhood labels.
- **Size/Capacity:** bedrooms, bathrooms, accommodates; property_type; room_type.
- **Amenities:** binary flags and grouped counts (e.g., climate control, workspace, parking, kitchen).
- **Reviews:** overall and sub-scores; total and recent review counts (e.g., last 90–180 days).
- **Seasonality:** month; optional event markers if available.

**Targets & evaluation fields:**

- `target_price_month`: comp-set **median** price for nights marked unavailable that month.  
- `peer_band_p25/p75_month`: comp-set interquartile band (for evaluation/guardrails).  
- `comp_ids_topK`, `similarity_scores`: diagnostics for transparency and auditability.

**Supporting nightly cache (for target construction):**

- `(listing_id, date, available, price, adjusted_price)` where present; used to compute monthly comp-set medians and bands.

## 6) Pre-Processing & Cleaning (From InsideAirbnb Snapshots)

1. **Ingest & standardize:** Load monthly `listings.csv`, `calendar.csv`, `reviews.csv`; add `scrape_month`. Conform schemas across months/cities.  
2. **Spatial features:** Convert lat/lon to a consistent spatial index (e.g., H3); keep neighborhood text labels.  
3. **Comparable set (comp set) construction:** For each listing-month, retrieve top-K comparables based on a mixed similarity score combining:  
   - spatial proximity (same/near cell),  
   - similar property/room type and capacity,  
   - amenity overlap,  
   - optional text similarity (if listing text available).  
4. **Monthly price targets:** Within each listing’s comp set and month, collect nightly records with `available == 'f'` and compute the **median** and P25–P75 price band. (State clearly that “unavailable” conflates booked and blocked.)  
5. **Assemble listing-month panel:** Join axis features, the target price, and evaluation fields (bands, comp IDs, similarity metrics).  
6. **Modeling data splits:** Time-based splits (train on earlier months, validate on recent, test on the most recent) and, optionally, geographic hold-outs to check generalization.

## 7) Proposed 5-Week Timeline (Parallelized for 3 Contributors)

**Roles (workstreams run in parallel):**  

- **Data & ETL Lead** — builds the pipelines, schemas, and comp-set cache.  
- **Modeling Lead** — defines the additive architecture, trains/validates models, and prepares explanations.  
- **App Lead** — develops the web UI, API contracts, and integrates placeholders → live endpoints.

### Week 1 — Foundations & Prototyping

- **Data & ETL:** Ingest 1–2 cities; standardize schemas; implement spatial indexing; draft comp-set retrieval (baseline k-NN by tabular features + distance).  
- **Modeling:** Define axes and target (market-accepted median); design additive architecture and metrics (peer-band hit rate, distance to median).  
- **App:** Build a clickable prototype (host input form → mock comp table → mock monthly price chart and factor breakdown). Define API contract (`/recommend` request/response).

**Deliverables:** data schema docs; first comp-set prototype; app skeleton with placeholder outputs.

### Week 2 — Targets & Featureization

- **Data & ETL:** Compute nightly → monthly comp-set medians/bands; materialize the **listing-month panel**.  
- **Modeling:** Implement axis encoders (location, size, amenities, reviews, seasonality); train a simple additive baseline (e.g., linear/GAM) on one city.  
- **App:** Wire real comp-set retrieval and real monthly price bands into the UI; replace placeholders for comparison views.

**Deliverables:** first real target table; baseline additive model; app showing real comps and peer bands.

### Week 3 — Additive Model v1 & Explanations

- **Data & ETL:** Optimize comp retrieval (e.g., approximate nearest neighbors); add diagnostics (comp IDs, similarity scores).  
- **Modeling:** Train the **multi-axis additive model v1**; produce per-axis contributions and evaluation (peer-band hit, distance to median).  
- **App:** Integrate contribution breakdown and top-K comparables with evidence; add simple CSV/PNG export of results.

**Deliverables:** model v1 with contributions; evaluation report draft; app with explanations and exports.

### Week 4 — Scale, Robustness & Integration

- **Data & ETL:** Extend to additional months/cities; harden pipelines; add data quality checks (missing price rates, comp coverage).  
- **Modeling:** Improve encodings (amenity/text embeddings if available); tune for stability; finalize metrics and plots.  
- **App:** Connect to a lightweight model API (local or hosted); implement guardrails (don’t recommend outside peer band unless clearly justified).

**Deliverables:** multi-city dataset; stabilized model; app running against a live endpoint with guardrails.

### Week 5 — Polish, Validation & Delivery

- **Data & ETL:** Produce a reproducible run (Makefile/notebooks/scripts) for one full city from raw to app-ready tables.  
- **Modeling:** Final evaluation; ablations (drop an axis to show impact); calibration by neighborhood/month; documentation of assumptions/limits.  
- **App:** UX polish; onboarding copy; README and usage guide; optional deployment (containerized API + simple hosting, or notebook demo if deployment is out-of-scope).

**Deliverables:** end-to-end reproducible pipeline, final model and report, polished app demo, documentation.

---

## Glossary

- **Axis:** a coherent group of features that influences price (e.g., location, size/capacity, amenities, reviews, seasonality).  
- **Comparable set (comp set):** the nearby, most similar listings used as market evidence for pricing; similarity considers location proximity, property/room type, capacity, and amenities (optionally text).  
- **Embedding:** a numeric vector representation of features (e.g., amenities list) learned or engineered to capture similarity.  
- **Sub-network:** a small model that takes one axis’s features and outputs that axis’s **price contribution**.  
- **Additive model:** final price is the **sum** of axis contributions (plus a baseline), enabling a clear explanation of “what added how much.”  
- **Market-accepted median (target):** for a listing and month, the comp-set **median price** computed only over nights marked “unavailable”; used as the training target with clear caveats.
