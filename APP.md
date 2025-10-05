# Application Technical Specification: An Explainable Airbnb Pricing Tool

## 1. System Architecture: Decoupled Cloud-Native Deployment on AWS

The application will be deployed as a decoupled system on AWS, separating the user interface from the machine learning model. This is a standard, scalable, and robust architecture for MLOps.

### 1.1. Core Components

* **ML Model Serving (Back-End): Amazon SageMaker Endpoint**
  * **Description:** The trained PyTorch model will be deployed to a real-time SageMaker Endpoint. SageMaker will manage the underlying compute infrastructure, provide a RESTful API for inference, and handle auto-scaling.
  * **API Contract:** The endpoint will accept `POST` requests with a `Content-Type` of `application/json`.
    * **Request Body:** A JSON object containing all the raw feature values required by the model, matching the schema of our final dataset. (See Appendix A for a sample JSON request).
    * **Response Body:** A JSON object containing the final recommended price, the individual price contributions from each of the 5 axes, the global bias, and the `listing_ids` of the top 3 comparable listings. (See Appendix B for a sample JSON response).

* **Web Application (Front-End): AWS App Runner with a Containerized Application**
  * **Description:** The user interface will be built as a Python web application using the **Streamlit** framework. The app will be packaged as a **Docker** container and deployed on AWS App Runner. App Runner will manage the build, deployment, and scaling of the web service.
  * **Functionality:** The Streamlit app's Python code is responsible for rendering the UI, collecting user inputs, validating them, constructing the JSON payload for the SageMaker endpoint, making the API call, and displaying the results returned by the endpoint.

* **Artifact & Data Storage: Amazon S3**
  * **Description:** An S3 bucket will act as the central repository for all project assets, including the final trained model artifacts (`model.pth`), the datasets, and any other required files (e.g., the vocabulary mapping for categorical features). The SageMaker Endpoint will be configured to load the model directly from this S3 bucket.

### 1.2. Data Flow Diagram

`User (Web Browser) -> AWS App Runner (Streamlit UI) -> API Gateway (optional) -> Amazon SageMaker Endpoint (PyTorch Model) -> Returns JSON -> Streamlit UI renders results`

## 2. User Interface (UI) Design & Layout

The application will be a simple, single-page interface built with Streamlit, designed for clarity and ease of use.

### 2.1. App Title

`Explainable Airbnb Price Recommender`

### 2.2. Layout

The layout will use a two-column design: a sidebar for user inputs and a main area for displaying the results.

### 2.3. Input Panel (Sidebar)

This panel, created with `st.sidebar`, will contain all the necessary input fields for a user to describe their listing. Dropdown menus should be populated from a file containing the unique values from the training dataset (e.g., `feature_vocab.json`).

* **Section Header:** `st.sidebar.header("Enter Your Listing's Details")`

* **Location Inputs:**
  * `latitude = st.sidebar.number_input("Latitude", value=40.7128)`
  * `longitude = st.sidebar.number_input("Longitude", value=-74.0060)`
  * `neighbourhood = st.sidebar.selectbox("Neighborhood", options=NEIGHBORHOOD_LIST)`

* **Size & Capacity Inputs:**
  * `property_type = st.sidebar.selectbox("Property Type", options=PROPERTY_TYPE_LIST)`
  * `room_type = st.sidebar.selectbox("Room Type", options=ROOM_TYPE_LIST)`
  * `accommodates = st.sidebar.slider("Accommodates", 1, 16, 2)`
  * `bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 1)`
  * `bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 1)`

* **Amenities Input:**
  * `amenities_str = st.sidebar.text_area("Amenities (comma-separated)", "Wifi, Kitchen, Heating, Air conditioning")`

* **Action Button:**
  * `if st.sidebar.button("Recommend Price"):`

### 2.4. Results Panel (Main Area)

This area will be conditionally rendered after the "Recommend Price" button is clicked and the API call to SageMaker is successful.

* **Recommendation Display:**
  * `st.metric(label="Recommended Nightly Price", value=f"${response['recommended_price']:.2f}")`

* **Price Decomposition Display:**
  * `st.subheader("Price Breakdown")`
  * The contributions from the API response (`response['breakdown']`) will be visualized using a **horizontal bar chart** (e.g., using `st.bar_chart` or Plotly Express). The chart should clearly label each axis and its dollar contribution. Both positive and negative contributions should be handled gracefully.

* **Market Comparables Display:**
  * `st.subheader("Top 3 Market Comparables")`
  * The top 3 `comp_ids` from the API response will be used to look up the details of those listings from a reference file (e.g., a `listings_summary.csv` containing price, key features, and URL for all listings).
  * The results will be displayed in three columns using `st.columns(3)`. Each column will act as a "card" for one comparable listing.
  * Within each column (card):
    * Display the listing's name (`st.text`).
    * Display its actual price (`st.metric`).
    * Display 2-3 key features (e.g., Bedrooms, Accommodates).
    * Provide a clickable link to its Airbnb page (optional, if available).

---

## Appendix A: Sample SageMaker Request Payload (JSON)

```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "neighbourhood_cleansed": "Williamsburg",
  "property_type": "Entire rental unit",
  "room_type": "Entire home/apt",
  "accommodates": 2,
  "bedrooms": 1,
  "bathrooms_numeric": 1.0,
  "bathrooms_type": "private",
  "amenities": "Wifi, Kitchen, Heating, Air conditioning",
  "review_scores_rating": 4.9,
  "review_scores_cleanliness": 4.8,
  "review_scores_checkin": 5.0,
  "review_scores_communication": 5.0,
  "review_scores_location": 4.7,
  "review_scores_value": 4.8,
  "number_of_reviews": 150,
  "host_is_superhost": true,
  "host_identity_verified": true,
  "instant_bookable": false,
  "month": 7
}
```

## Appendix B: Sample SageMaker Response Payload (JSON)

```json
{
  "recommended_price": 185.50,
  "breakdown": {
    "global_bias": 110.00,
    "location": 55.20,
    "size_capacity": 25.30,
    "amenities": 15.00,
    "quality_reputation": -10.00,
    "seasonality": -10.00
  },
  "comparable_listing_ids": [
    7801,
    6848,
    2595
  ]
}
```
