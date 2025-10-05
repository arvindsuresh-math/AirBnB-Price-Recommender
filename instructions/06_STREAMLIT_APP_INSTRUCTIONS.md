# `06_STREAMLIT_APP_INSTRUCTIONS.md`

### LLM Agent Pre-computation Instructions

To enrich your context, first read and fully comprehend the following project design documents in order:

1. `APP.md`: This is your **primary blueprint**. The User Interface Design, layout, component selection, and API contract (Request/Response payloads) described within are non-negotiable and must be implemented exactly.
2. `DATASET-SCHEMA.md`: To understand the full set of features required by the model. This will inform the UI inputs you need to create.
3. `EMBEDDINGS.md`: To find the lists of unique values for categorical features like `neighbourhood_cleansed`, `property_type`, and `room_type`. You will need to hardcode these lists to populate the `selectbox` options in the UI.

### Primary Objective

Your task is to generate a single, user-facing Python script located at `app.py`. This script will create a web application using the Streamlit framework. The application will serve as the front-end for the pricing tool, collecting user inputs, invoking the deployed SageMaker endpoint, and displaying the results in a clear and interpretable format.

---

## 1. Script Structure and Setup

### 1.1. Required Imports

`streamlit`, `pandas`, `json`, `boto3`, and `plotly.express`.

### 1.2. Page Configuration

* At the top of the script, set the page configuration: `st.set_page_config(layout="wide", page_title="Explainable Airbnb Price Recommender")`.

### 1.3. Hardcoded Categorical Feature Options

* Create constant Python lists containing the unique string values for the following categorical features. These will populate the `selectbox` widgets. You can source these from a quick analysis of the sample data or `EMBEDDINGS.md`.
  * `NEIGHBORHOOD_LIST`
  * `PROPERTY_TYPE_LIST`
  * `ROOM_TYPE_LIST`
  * `BATHROOMS_TYPE_LIST`

### 1.4. SageMaker Endpoint Configuration

* Define a constant for the SageMaker endpoint name: `SAGEMAKER_ENDPOINT_NAME = "your-endpoint-name-here"`. This should be easily configurable at the top of the file.

---

## 2. User Interface (UI) Layout

The UI must follow the two-column layout described in `APP.md`.

### 2.1. Main Area: Titles and Placeholders

* `st.title("Explainable Airbnb Price Recommender")`
* Use `st.expander` to create a collapsible section explaining the purpose of the tool.
* Initialize two empty containers that will be populated with results later: `results_container = st.container()` and `comparables_container = st.container()`.

### 2.2. Sidebar: Input Panel (`st.sidebar`)

* Create a header: `st.sidebar.header("Enter Your Listing's Details")`.
* Create a Python function `get_user_inputs()` that contains all the `st.sidebar` widgets and returns a dictionary of the user's selections.

**Input Widgets within `get_user_inputs()`:**

* **Location:**
  * `latitude = st.sidebar.number_input("Latitude", value=40.7128, format="%.4f")`
  * `longitude = st.sidebar.number_input("Longitude", value=-74.0060, format="%.4f")`
  * `neighbourhood_cleansed = st.sidebar.selectbox("Neighborhood", options=NEIGHBORHOOD_LIST)`
* **Size & Capacity:**
  * `property_type = st.sidebar.selectbox("Property Type", options=PROPERTY_TYPE_LIST)`
  * `room_type = st.sidebar.selectbox("Room Type", options=ROOM_TYPE_LIST)`
  * `accommodates = st.sidebar.slider("Accommodates", 1, 16, 2)`
  * `bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 1)`
  * `bathrooms_numeric = st.sidebar.slider("Bathrooms (Numeric)", 0.0, 5.0, 1.0, 0.5)`
  * `bathrooms_type = st.sidebar.selectbox("Bathroom Type", options=BATHROOMS_TYPE_LIST)`
* **Amenities:**
  * `amenities_str = st.sidebar.text_area("Amenities (comma-separated)", "Wifi, Kitchen, Heating, Air conditioning")`
* **Quality & Reputation (with defaults for new listings):**
  * `review_scores_rating = st.sidebar.slider("Expected Rating", 1.0, 5.0, 4.8, 0.1)`
  * `number_of_reviews = st.sidebar.number_input("Current Number of Reviews", value=0)`
  * `host_is_superhost = st.sidebar.checkbox("Is Superhost?", value=False)`
  * `host_identity_verified = st.sidebar.checkbox("Is Host Identity Verified?", value=True)`
  * `instant_bookable = st.sidebar.checkbox("Is Instant Bookable?", value=False)`
* **Seasonality:**
  * `month = st.sidebar.slider("Month for Pricing", 1, 12, 7)`
* **Action Button:**
  * `recommend_button = st.sidebar.button("Recommend Price")`
* The function should return a dictionary containing all these values and the button's state.

---

## 3. Backend Logic: SageMaker Invocation

### 3.1. API Payload Construction

* Create a function `construct_payload(user_inputs: dict) -> str`.
* This function takes the dictionary from `get_user_inputs()`.
* It must construct a new dictionary whose keys and value types **exactly match** the sample request in `APP.md`, Appendix A.
* **Crucial Transformation**: The `amenities_str` (a single string) must be converted into the format expected by the model in the payload (which is also a single string).
* Add default values for features not collected in the UI but required by the model (e.g., `review_scores_cleanliness: 4.8`, `review_scores_checkin: 5.0`, etc.). The defaults should be sensible averages.
* Return the dictionary serialized as a JSON string using `json.dumps()`.

### 3.2. SageMaker Invocation

* Create a function `invoke_sagemaker_endpoint(payload: str) -> dict`.
* This function should be decorated with `st.cache_data` to prevent re-running the API call on every UI interaction.
* **Logic**:
    1. Initialize the Boto3 SageMaker runtime client: `client = boto3.client('sagemaker-runtime', region_name='your-aws-region')`.
    2. Use a `try...except` block to handle potential `EndpointConnectionError` or other Boto3 exceptions.
    3. Call `client.invoke_endpoint()` with `EndpointName=SAGEMAKER_ENDPOINT_NAME`, `ContentType='application/json'`, and `Body=payload`.
    4. Read and parse the JSON response from the endpoint's response body.
    5. Return the parsed dictionary. If there was an error, return an error dictionary (e.g., `{'error': 'Could not connect to endpoint.'}`).

---

## 4. Main Application Flow

* Call `get_user_inputs()` to render the sidebar and get the current inputs.
* Check if the "Recommend Price" button was clicked: `if user_inputs['recommend_button']:`.
* Inside the `if` block:
    1. Display a spinner: `with st.spinner('Fetching recommendation...'):`.
    2. Call `construct_payload()` with the user inputs.
    3. Call `invoke_sagemaker_endpoint()` with the payload.
    4. Check the response for an error key. If an error exists, display it using `st.error()`.
    5. If successful, store the response dictionary in `st.session_state['api_response']`.

## 5. Results Display

* Check if the API response exists in the session state: `if 'api_response' in st.session_state:`.
* Inside this `if` block, unpack the response and render the results using the containers defined in Section 2.1.
* **Recommended Price (`with results_container:`):**
  * `st.metric(label="Recommended Nightly Price", value=f"${response['recommended_price']:.2f}")`
* **Price Breakdown (`with results_container:`):**
  * `st.subheader("Price Breakdown")`
  * Extract the `breakdown` dictionary from the response.
  * Convert it into a pandas DataFrame with columns `['Axis', 'Contribution']`.
  * Use `plotly.express.bar()` to create a horizontal bar chart. Configure it to have a title and appropriate labels. Use `st.plotly_chart(fig, use_container_width=True)`.
* **Market Comparables (`with comparables_container:`):**
  * `st.subheader("Top 3 Market Comparables")`
  * `st.info("Market comparables feature is under development.")` (Placeholder as per project plan).

---

## 6. Verification and Unit Tests

Generate a test script at `tests/test_app_helpers.py`. The focus is on testing the data transformation logic, not the Streamlit UI rendering itself.

### Test Cases

1. **`test_payload_construction()`**:
    * Create a sample dictionary of user inputs as would be returned by `get_user_inputs()`.
    * Call `construct_payload()` with this dictionary.
    * Parse the returned JSON string.
    * Assert that the resulting dictionary contains all the keys expected by the SageMaker endpoint (as per `APP.md`, Appendix A).
    * Assert that the data types are correct (e.g., `month` is an integer, `latitude` is a float).

2. **`test_sagemaker_invocation_mocked(monkeypatch)`**:
    * Use `pytest`'s `monkeypatch` fixture.
    * Create a mock Boto3 client and a mock `invoke_endpoint` function that returns a predictable, correctly formatted JSON response.
    * Use `monkeypatch.setattr(boto3, 'client', lambda *args, **kwargs: mock_client)`.
    * Call `invoke_sagemaker_endpoint()` with a dummy payload.
    * Assert that the function returns the parsed dictionary from your mock response.
    * Assert that the mock `invoke_endpoint` method was called exactly once.
