# STREAMLIT-APP-INSTRUCTIONS.md

## 1. Objective

This specification details the creation of a Python script for a Streamlit web application. The application will serve as the user interface for the Explainable Airbnb Pricing Tool. It will collect user inputs, make a real-time inference request to a deployed SageMaker model endpoint, and display the returned price recommendation, decomposition, and comparable listings in a clear and intuitive manner.

## 2. File and Path Specifications

### 2.1. Script Location and Name

* **Location:** `/Fall-2025-Team-Big-Data/`
* **Name:** `app.py`

### 2.2. Required Input Artifacts

The application requires two artifacts to function correctly. These should be located in a new, dedicated directory.

* **App Artifacts Directory:** `/Fall-2025-Team-Big-Data/app_artifacts/`

* **Artifact 1: Feature Vocabularies**
  * **Source:** This file is a simplified version of the main feature artifacts file. It should be created by a small helper script.
  * **Location:** `/Fall-2025-Team-Big-Data/app_artifacts/feature_vocab.json`
  * **Content:** A JSON object containing lists of unique values for each dropdown menu in the UI. Example:

        ```json
        {
          "neighbourhood_cleansed": ["Williamsburg", "Midtown", ...],
          "property_type": ["Entire rental unit", "Private room in condo", ...],
          "room_type": ["Entire home/apt", "Private room", "Shared room"]
        }
        ```

* **Artifact 2: Comparables Data Lookup**
  * **Source:** A subset of the main listings CSV.
  * **Location:** `/Fall-2025-Team-Big-Data/app_artifacts/listings_summary.csv`
  * **Content:** A CSV file containing, at a minimum, the following columns for **all** listings in the dataset: `id`, `name`, `price`, `bedrooms`, `accommodates`, `listing_url`. This file is used to retrieve details for the comparable listings returned by the model.

## 3. Environment and Configuration

### 3.1. AWS Configuration

The application must communicate with a deployed SageMaker endpoint. This requires AWS credentials.

* **Method:** The script will use the `boto3` library. `boto3` will automatically search for credentials (e.g., from environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, or from an EC2 instance role).
* **Configuration:** The script must define two constants at the top:
  * `SAGEMAKER_ENDPOINT_NAME = "your-sagemaker-endpoint-name-here"` (This must be updated with the actual endpoint name).
  * `AWS_REGION = "us-east-1"` (Or the appropriate AWS region).

### 3.2. Mocking for Local Development

To allow UI development before the SageMaker endpoint is live, the script MUST include a "mocking" mode.

* **Logic:** A global boolean flag, `MOCK_API_CALL = True`, should be defined at the top of the script. The function that calls SageMaker must check this flag. If `True`, it should skip the `boto3` call and immediately return the hardcoded sample JSON response from `APP.md`.

## 4. Step-by-Step UI and Logic Implementation

The script should be implemented using the `streamlit` library.

### Step 1: Initialization and Data Loading

1. Import necessary libraries: `streamlit`, `pandas`, `json`, `boto3`.
2. Define the constants: `MOCK_API_CALL`, `SAGEMAKER_ENDPOINT_NAME`, `AWS_REGION`.
3. Load the required artifacts into memory:
    * Load `feature_vocab.json` into a dictionary.
    * Load `listings_summary.csv` into a Pandas DataFrame and set the `id` column as the index for fast lookups.

### Step 2: UI Implementation (Input Sidebar)

1. Set the page title: `st.set_page_config(layout="wide")`.
2. Render the main app title: `st.title("Explainable Airbnb Price Recommender")`.
3. Implement the input panel in the sidebar as specified in `APP.md`.
    * `st.sidebar.header(...)`
    * Use the loaded vocabulary lists to populate the `options` for `st.selectbox`.
    * Store the return value of each Streamlit widget in a variable.

### Step 3: Backend Communication Function

1. Define a function: `get_price_recommendation(payload)`.
2. Inside this function, check the `MOCK_API_CALL` flag.
    * If `True`, `time.sleep(1)` to simulate a network call and return the hardcoded sample JSON response.
    * If `False`, implement the `boto3` logic:
        a. Create a SageMaker runtime client: `sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=AWS_REGION)`.
        b. Convert the Python payload dictionary to a JSON string.
        c. Invoke the endpoint:

           ```python
           response = sagemaker_runtime.invoke_endpoint(
               EndpointName=SAGEMAKER_ENDPOINT_NAME,
               ContentType='application/json',
               Body=json.dumps(payload)
           )
           ```        d. Read the response body, decode it from UTF-8, and parse it back into a Python dictionary using `json.loads`.
        e. Return the parsed dictionary.
3. This function should include error handling (e.g., a `try...except` block) for the API call.

### Step 4: Main Application Logic

1. Wrap the main logic in an `if st.sidebar.button("Recommend Price"):` block.
2. Inside the block:
    a. **Collect Inputs:** Gather all the values from the sidebar widgets.
    b. **Construct Payload:** Create a Python dictionary that exactly matches the structure of the **Sample SageMaker Request Payload** in `APP.md`.
    c. **Make API Call:** Call the `get_price_recommendation` function with the payload.
    d. **Display Results:** Use the returned dictionary to render the entire results panel as specified in `APP.md`.
        *Use `st.metric` for the main price.
        *   Convert the `breakdown` dictionary into a Pandas DataFrame and use `st.bar_chart` to display it.
        *Retrieve the `comparable_listing_ids`, look them up in the `listings_summary` DataFrame using `.loc`.
        *   Use `st.columns(3)` to create the three cards and populate them with the data for the comparable listings.

## 5. Dockerization

A `Dockerfile` must be created in the root directory to containerize the application for deployment on AWS App Runner.

* **Dockerfile:**

    ```dockerfile
    # Use an official Python runtime as a parent image
    FROM python:3.10-slim

    # Set the working directory in the container
    WORKDIR /app

    # Copy the requirements file and install dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy the application code and artifacts
    COPY app.py .
    COPY app_artifacts/ ./app_artifacts/

    # Make port 8501 available to the world outside this container
    EXPOSE 8501

    # Define the command to run the app
    CMD ["streamlit", "run", "app.py"]
    ```*   A corresponding `requirements.txt` file must be created, containing `streamlit`, `pandas`, and `boto3`.
