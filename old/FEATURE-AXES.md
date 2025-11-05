# Feature Architecture: Decomposing Price into Interpretable "Axes"

## 1. The Additive Axis Model

To achieve the project's primary goal of explainability, we will not treat the features as a single monolithic block. Instead, we will partition them into logically coherent subsets, which we term **"axes."** An axis is a group of related features that represents a distinct, intuitive driver of a property's price.

The core architectural idea is to model the final price as an **additive combination** of contributions from each of these axes:

`Price ≈ Baseline + Contribution(Location) + Contribution(Size) + ...`

Each axis will be fed into its own dedicated neural sub-network, which learns to output the marginal price contribution for that specific axis. The final model output is the sum of these contributions. This design allows us to directly inspect the output of each sub-network and explain the final price recommendation in simple terms (e.g., "$150 base price + $50 for the prime location + $30 for the amenities...").

## 2. The Five Core Axes (Phase 1)

For the initial implementation, we will use the following five axes. This feature set has been carefully curated to be comprehensive, robust, and directly aligned with the additive modeling goal.

| Axis Name | Features Included |
| :--- | :--- |
| **1. Location** | `latitude`, `longitude`, `neighbourhood_cleansed` |
| **2. Size & Capacity** | `property_type`, `room_type`, `accommodates`, `bedrooms`, `beds`, `bathrooms_numeric`, `bathrooms_type` |
| **3. Amenities** | `amenities` |
| **4. Quality & Reputation** | `review_scores_rating`, `review_scores_cleanliness`, `review_scores_checkin`, `review_scores_communication`, `review_scores_location`, `review_scores_value`, `number_of_reviews_ltm`, `host_is_superhost`, `host_response_rate`, `host_acceptance_rate`, `host_identity_verified`, `instant_bookable` |
| **5. Seasonality** | `month` |

## 3. Notes on Feature Selection

The selection of these features is deliberate. Features were excluded for several key reasons to ensure model robustness and validity:

* **Target Leakage:** Features that are an *outcome* of price and booking activity (e.g., `availability_365`, `reviews_per_month`) have been excluded to prevent the model from learning spurious correlations.
* **High Complexity / Deferred:** Unstructured text features (`name`, `description`, `comments`) are highly valuable but require complex NLP techniques. They are intentionally deferred to a potential Phase 2 to ensure the successful delivery of the core model first.
* **Redundancy or Low Signal:** Features that are redundant (e.g., using `neighbourhood_cleansed` over the less reliable `neighbourhood`) or administrative (e.g., `license`) have been omitted to simplify the model and focus on the primary drivers of price.

## 4. Future Enhancements: An Attention-Infused Model (Phase 2 Stretch Goal)

Should time permit after the successful implementation of the core additive model, we can enhance its predictive power by incorporating unstructured text data. The review `comments` are particularly valuable, as they often contain specific context about the other features.

The proposed enhancement is to use a **cross-modal attention mechanism**:

* **Concept:** The unstructured text from reviews (e.g., "The location was perfect, right next to the subway!") can be used to dynamically modulate the inputs or outputs of the other sub-networks.
* **Implementation:**
    1. First, we would aggregate and embed all review comments for a listing into a single representative vector.
    2. This review vector would then "attend to" each of the other axes. For example, the attention mechanism could learn that the phrase "next to the subway" is highly relevant to the **Location** axis and use that information to boost its output contribution.
* **Impact:** This would create a more sophisticated model where the axes are not entirely independent but can influence one another through the shared context of the review text. While this would likely improve accuracy, it comes at the cost of increased complexity in both implementation and interpretation.
