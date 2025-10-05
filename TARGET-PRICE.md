# Engineering the Target Variable: A Data-Driven Proxy for the Revenue-Maximizing Price

## 1. The Strategic Objective: From Revenue Maximization to a Market-Price Proxy

The ultimate goal of a pricing tool is to recommend a price that **maximizes a host's revenue**. Revenue is a function of both nightly price and occupancy: `Revenue = Price × Occupancy`. Critically, occupancy is itself a function of price, `Occupancy(Price)`, representing the listing's demand curve. The true optimization problem is to find the price `P` that maximizes `P × Occupancy(P)`.

Solving this directly would require building a causal model of the demand curve, which is not feasible with our observational dataset. We cannot easily isolate the causal effect of price from the thousands of other confounding variables across different listings.

Therefore, we will pivot to a more robust, data-driven objective: we will build a model to predict the **market-accepted price**, operating under a key strategic assumption:

**Assumption:** The observed equilibrium price in the market is a strong and reliable proxy for the theoretical revenue-maximizing price.

This assumption is justified by the "wisdom of the crowds." The market consists of thousands of hosts and guests making rational decisions. Prices that are too high result in zero revenue, while prices that are too low represent missed opportunities. The resulting market-accepted prices are the product of a large-scale, real-world optimization process. By modeling this price, we leverage the collective intelligence of the market to guide our recommendations.

## 2. Measuring Market Price: The Ideal Dataset

To perfectly measure the market-accepted price, an ideal dataset would include:

* **Dynamic Daily Pricing:** The exact price a listing was offered at on any given day.
* **Unambiguous Calendar Status:** A daily status for each listing, clearly distinguishing between `AVAILABLE` (open for booking), `BOOKED` (successfully transacted at a specific price), and `BLOCKED` (made unavailable by the host).

With this data, we could directly and accurately identify confirmed bookings and the prices at which they occurred.

## 3. Data Limitations and Engineering Challenges

Our available dataset from Inside Airbnb presents two primary challenges:

* **Snapshot Price:** The `price` column in `listings.csv` is a single value captured at the time of the monthly data scrape, not a confirmed historical transaction price.
* **Ambiguous Unavailability:** The `calendar.csv` data does not differentiate between a `BOOKED` night and a `BLOCKED` night. Both are labeled as `available = 'f'`. This makes it impossible to reliably calculate true occupancy from the calendar alone.

## 4. The Engineering Solution: Estimating Occupancy via a Recent Review Window

To overcome these limitations, we will engineer a robust proxy for market acceptance using the full historical `reviews.csv` dataset. Instead of using a simple filter, we will calculate a continuous measure of recent market success for each listing and use it to weight the importance of each data point during model training.

The process is as follows:

**Step 1: Pre-process Historical Reviews**
First, we will create a monthly summary table from the full `reviews.csv` dataset. This table will have the schema `(listing_id, year_month, reviews_in_month)`, counting the number of reviews each listing received in a given month.

**Step 2: Engineer a Rolling Activity Metric**
We will join this monthly review count to our main `listing-month` dataset. For each record, we will calculate a new feature, `reviews_in_last_90_days`, by summing the `reviews_in_month` for the current and two preceding months. This rolling window provides a timely signal of recent activity while being robust to the natural lag and sparsity of guest reviews.

**Step 3: Estimate Occupancy Rate**
Using the rolling review count, we will estimate the occupancy rate based on the "San Francisco Model" methodology:

* `estimated_bookings = reviews_in_last_90_days / review_rate`
    *(Where `review_rate` is a hyperparameter, e.g., 0.5 for a 50% review rate).*
* `estimated_nights_booked = estimated_bookings * avg_length_of_stay`
    *(Where `avg_length_of_stay` is a city-specific hyperparameter, e.g., 3 nights).*
* `estimated_occupancy_rate = min(1.0, estimated_nights_booked / 90)`
    *(The rate is capped at 100%).*

**Step 4: Apply as a Sample Weight**
This `estimated_occupancy_rate` will be a column in our final dataset. It will not be a feature for the model, but rather a **sample weight** passed to the loss function during training. This instructs the model to prioritize learning from listings with higher demonstrated market success. A listing with 80% estimated occupancy will have a much stronger influence on the model's learned parameters than one with 10% occupancy. Any listing with zero recent reviews will have a weight of zero and will be effectively ignored.

**Step 5: Define Final Target**
The target variable the model learns to predict remains the simple, interpretable nightly price.
`target_price = price`

This methodology provides a robust and theoretically sound solution. It formally connects our high-level business objective to a concrete and defensible data engineering process.
