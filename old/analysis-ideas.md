### **Category 1: Deeper Error Analysis**

These plots help you understand *where* and *why* your model is making mistakes, which is the most direct path to improving it.

#### **1. Residuals vs. Predicted Price Plot**

* **What:** A scatter plot with the predicted price on the x-axis and the residual (error: `true_price - predicted_price`) on the y-axis.
* **Why:** This is a classic diagnostic plot that helps you check for **heteroscedasticity**—a situation where the error margin systematically changes as the prediction value increases. If you see a cone shape (errors get larger for more expensive listings), it might indicate that a log transform of the target was a good idea, or that the model has trouble with very high-end properties. A random cloud of points around y=0 is the ideal outcome.
* **How:** On your validation results dataframe, compute `error = price - predicted_price` and create a scatter plot of `predicted_price` vs. `error`.

#### **2. Performance by Categorical Features**

* **What:** Bar plots showing the Mean Absolute Error (MAE) in dollars for different categories within key features like `room_type`, `property_type`, or even top N `neighbourhood_cleansed`.
* **Why:** This helps you identify if the model is systematically worse for certain types of listings. For example, you might discover that your model is excellent for "Entire home/apt" but struggles with "Shared room". This could guide future feature engineering or data collection efforts.
* **How:** Group your validation results dataframe by `room_type` (or another categorical column) and calculate the mean of the absolute error for each group. Then, create a bar plot of the results.

#### **3. Geospatial Error Map**

* **What:** An interactive Folium map where each listing is a dot, colored by its prediction error. For example, blue for under-predictions (`predicted < true`), red for over-predictions (`predicted > true`), with color intensity representing the magnitude of the error.
* **Why:** This can reveal geographic patterns in your model's performance. You might find that the model consistently underprices listings near a newly popular area or overprices them in a neighborhood with older housing stock that the features don't fully capture.
* **How:** Using your validation results, create a Folium map. For each listing, add a `CircleMarker` whose `fill_color` is determined by the value of the error column.

---

### **Category 2: Model Interpretability and Feature Impact**

These visualizations leverage the unique `p_*` contribution columns to explain *how* the model is making its decisions.

#### **4. Distribution of Price Contributions**

* **What:** A box plot or violin plot showing the distribution of each of the additive log-price contributions (`p_location`, `p_size_capacity`, `p_amenities`, etc.) across the entire dataset.
* **Why:** While you've already seen the *average* contribution, this shows the full picture. It answers questions like: Which factor has the widest range of impact on price? Is the impact of `quality` generally small but with a few strong outliers? This reveals the variance and importance of each feature axis in the model's decision-making process.
* **How:** Select all `p_*` columns from your results dataframe and use `seaborn.boxplot` or `seaborn.violinplot`.

#### **5. Correlating Raw Features with Their Learned Contributions**

* **What:** A series of scatter plots that correlate a key input feature with its corresponding axis contribution. For example:
  * `review_scores_rating` (x-axis) vs. `p_quality` (y-axis)
  * `accommodates` (x-axis) vs. `p_size_capacity` (y-axis)
* **Why:** This is a powerful "sanity check" to ensure your sub-networks are learning logical relationships. You would expect to see a positive correlation in the examples above. If you don't, it might indicate a problem in that sub-network's architecture or the features being fed into it.
* **How:** Create a scatter plot for each pair using your results dataframe. You can add a regression line (`sns.regplot`) to make the trend clearer.

#### **6. Analyzing the "Most Misunderstood" Listings**

* **What:** A simple table displaying the top 5-10 listings from the validation set with the highest absolute error. The table should show the `price`, `predicted_price`, `error`, and the full breakdown of all six `p_*` contributions.
* **Why:** This is a fantastic debugging tool. By examining the individual price breakdowns of your biggest failures, you can often find patterns. For instance, you might find that the model consistently fails on listings where the `p_description` contribution is extremely high, suggesting it's over-relying on certain keywords.
* **How:** Sort your validation results dataframe by the absolute error in descending order and display the top rows (`.head(10)`).

---

### **Category 3: Price Statistics and Seasonal Analysis**

These plots use the augmented (all-months) dataset to provide strategic insights.

#### **7. Learned City-Wide Seasonality Curve**

* **What:** A line plot showing the average multiplicative price factor (`pm_seasonality`) for each month of the year (1 through 12).
* **Why:** This explicitly visualizes the seasonal pricing trend the model has learned for the city as a whole. It will clearly show the peak season (e.g., a factor of 1.15 in July, meaning a 15% price increase) and the low season (e.g., a factor of 0.90 in February, a 10% discount).
* **How:** Using your final `_app_database.parquet` file, group by `month` and calculate the mean of the `pm_seasonality` column. Then, plot the result.

#### **8. Top 10 "Underpriced" Opportunities**

* **What:** A table of the top 10 listings from the validation set that have the largest "opportunity gap" (`predicted_price - price`).
* **Why:** This is a direct, business-relevant application of your model. It provides a list of properties that, according to the model, could potentially increase their prices without being out of line with the market. This is a powerful feature for a host-facing tool.
* **How:** On your validation results, calculate `opportunity_gap = predicted_price - price`. Sort by this column in descending order and display the top 10 listings, including their name, neighborhood, price, and predicted price.
