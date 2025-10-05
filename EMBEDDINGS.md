# Feature Representation and Embedding Strategy

## 1. Objective

The goal of this stage is to transform the raw, heterogeneous features from our pre-processed dataset into dense, numerical tensors. This is a critical pre-requisite for our multi-axis neural network, as each sub-network requires a fixed-size, continuous vector as its input. Our strategy is to tailor the representation method to the specific statistical properties of each feature—be it continuous, categorical, ordinal, or text-based.

## 2. Core Methodologies

We will employ a variety of standard and state-of-the-art techniques to create meaningful representations:

*   **Learned Embeddings (`nn.Embedding`):** Used for high-cardinality categorical features.
*   **One-Hot Encoding (OHE):** Used for low-cardinality categorical and boolean features.
*   **Numerical Transformations:** Standardization, Logarithmic Transform, and Cyclical Encoding.
*   **Fine-tuning Pre-trained Transformers:** Used for complex text features (`amenities`).

## 3. Axis-by-Axis Implementation Plan

The following table provides a detailed, feature-by-feature breakdown of the embedding strategy for each of the five axes.

| Axis | Feature(s) | Data Type | Chosen Method | Output Dim | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Location** | `latitude`, `longitude` | Continuous | Positional (Cyclical) Encoding | 32 | Captures the non-linear & cyclical nature of geographic coordinates. |
| | `neighbourhood_cleansed`| High-Card. Categorical | Learned Embedding | 16 | Efficiently learns price similarities between many neighborhoods. |
| **2. Size & Capacity** | `property_type` | High-Card. Categorical | Learned Embedding | 8 | Efficiently handles the large vocabulary of property types. |
| | `room_type` | Low-Card. Categorical | One-Hot Encoding | 4 | Simplicity and clarity for a very small set of categories. |
| | `bathrooms_type` | Low-Card. Categorical | One-Hot Encoding | 2 | Simplicity and clarity for a binary-like category. |
| | `bedrooms`, `beds`, `bathrooms_numeric` | Ordinal (as Categorical)| Learned Embedding | 4 (each) | Provides flexibility to learn non-linear price impacts. |
| | `accommodates` | Continuous Integer | Standardization | 1 | Treats as a standard numerical feature. |
| **3. Quality & Reputation** | **`review_scores_rating`, `_cleanliness`, `_checkin`, `_communication`, `_location`, `_value`, `host_response_rate`, `host_acceptance_rate`** | Numerical (Ratings) | Standardization | 8 | Centers the 8 rating/rate features and puts them on a common scale. |
| | `number_of_reviews` | Numerical (Count) | Log Transform + Standardization | 1 | Mitigates the high skew of the total review count distribution. |
| | `host_is_superhost` | Boolean | One-Hot Encoding | 2 | Simple, unambiguous representation for a key binary flag. |
| | `host_identity_verified` | Boolean | One-Hot Encoding | 2 | Simple, unambiguous representation. |
| | `instant_bookable` | Boolean | One-Hot Encoding | 2 | Simple, unambiguous representation. |
| **4. Amenities**| `amenities` | Text | Fine-tune Pre-trained Transformer | 384 | Leverages a state-of-the-art language model (`bge-small-en-v1.5`) for semantic understanding. |
| **5. Seasonality**| `month` | Cyclical Integer | Cyclical Encoding | 2 | Correctly represents the cyclical nature of the year. |

## 4. Final Sub-Network Inputs

After applying the strategies above, each of the five sub-networks will receive a single, concatenated tensor with the following dimensions:

*   **Location Sub-net Input Dimension:** `32 + 16` = **48**
*   **Size & Capacity Sub-net Input Dimension:** `8 + 4 + 2 + 4 + 4 + 4 + 1` = **27**
*   **Quality & Reputation Sub-net Input Dimension:** `8 + 1 + 2 + 2 + 2` = **15**
*   **Amenities Sub-net Input Dimension:** **384**
*   **Seasonality Sub-net Input Dimension:** **2**