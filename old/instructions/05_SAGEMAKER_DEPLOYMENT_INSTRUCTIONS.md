# `05_SAGEMAKER_DEPLOYMENT_INSTRUCTIONS.md`

### LLM Agent Pre-computation Instructions

To enrich your context, first read and fully comprehend the following project design documents in order:

1. `APP.md`: For the non-negotiable API contract. The JSON request/response structures in Appendix A and B are the ground truth for the serialization/deserialization logic you will write.
2. `02_PYTORCH_DATASET_INSTRUCTIONS.md`: To understand the `FeatureProcessor` class, as you will be loading and using a saved instance of it.
3. `03_PYTORCH_MODEL_INSTRUCTIONS.md`: To understand the `AdditiveAxisModel` class, as you will be loading its saved weights.
4. `04_MODEL_TRAINING_INSTRUCTIONS.md`: To know the exact filenames of the artifacts produced by the training script (`best_model.pth`, `feature_processor.joblib`) which must be packaged for deployment.

### Primary Objective

Your task is to generate two distinct Python scripts:

1. **`sagemaker/inference.py`**: The core Python script containing the logic for model loading, data pre-processing, prediction, and post-processing. This script will be executed by the SageMaker managed PyTorch container.
2. **`scripts/deploy.py`**: A command-line utility script that packages the model artifacts and deploys them to a SageMaker real-time endpoint using the SageMaker Python SDK.

You will also specify a `sagemaker/requirements.txt` file.

---

## 1. Specification for `sagemaker/inference.py`

This script must define the four handler functions required by the SageMaker PyTorch serving container.

### 1.1. Required Imports

`json`, `os`, `pandas`, `torch`, `torch.nn`, `joblib`, `numpy`, and the custom classes `AdditiveAxisModel` and `FeatureProcessor` from `src.model` and `src.dataset`.

### 1.2. Function: `model_fn(model_dir)`

* **Purpose**: Loads the saved model artifacts from disk.
* **Input**: `model_dir` (string): The directory path where SageMaker has unzipped the `model.tar.gz` file.
* **Logic**:
    1. Load the `FeatureProcessor`: `processor = FeatureProcessor.load(os.path.join(model_dir, "feature_processor.joblib"))`.
    2. Extract the `vocab_sizes` and `embedding_dims` from the loaded `processor` object.
    3. Instantiate the model architecture: `model = AdditiveAxisModel(vocab_sizes, embedding_dims)`.
    4. Load the trained weights: `model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pth"), map_location=torch.device('cpu')))`
    5. Set the model to evaluation mode: `model.eval()`.
* **Output**: A dictionary containing the initialized model and processor: `{'model': model, 'processor': processor}`.

### 1.3. Function: `input_fn(request_body, request_content_type)`

* **Purpose**: Deserializes the incoming HTTP request payload.
* **Inputs**: `request_body` (string), `request_content_type` (string).
* **Logic**:
    1. Assert that `request_content_type` is `'application/json'`. If not, raise a `ValueError`.
    2. Parse the JSON string: `data = json.loads(request_body)`.
* **Output**: The parsed Python dictionary.

### 1.4. Function: `predict_fn(input_object, model_and_processor)`

* **Purpose**: Performs the prediction.
* **Inputs**: `input_object` (dict from `input_fn`), `model_and_processor` (dict from `model_fn`).
* **Logic**:
    1. Extract the model and processor from `model_and_processor`.
    2. Convert the single input dictionary into a pandas DataFrame with one row: `input_df = pd.DataFrame([input_object])`.
    3. Use the processor to transform the DataFrame: `processed_data = processor.transform(input_df)`.
    4. Convert the `processed_data` (a dictionary of lists) into a batch of tensors suitable for the model. Iterate through the dictionary, convert each list to a tensor, and add a batch dimension of 1.
    5. Wrap the prediction in `with torch.no_grad():`.
    6. Perform inference: `prediction = model(tensor_batch)`.
* **Output**: The raw prediction dictionary returned by the model's `forward` method (which contains tensors).

### 1.5. Function: `output_fn(prediction, accept)`

* **Purpose**: Serializes the prediction result into the HTTP response payload.
* **Inputs**: `prediction` (dict from `predict_fn`), `accept` (string).
* **Logic**:
    1. Assert that `accept` is `'application/json'`. If not, raise a `ValueError`.
    2. Get the `predicted_price` tensor and the `breakdown` dictionary from the `prediction` object.
    3. Convert all tensor values to standard Python floats using the `.item()` method. This is critical for JSON serialization.
    4. Construct the final response dictionary, ensuring its structure exactly matches `Appendix B` in `APP.md`. The `comparable_listing_ids` key can be an empty list `[]` for now.
    5. Serialize the dictionary to a JSON string: `return json.dumps(response_dict)`.

---

## 2. Specification for `sagemaker/requirements.txt`

Create this file with the following exact contents:

```
pandas==2.2.0
scikit-learn==1.4.1.post1
joblib==1.3.2
sentence-transformers==2.6.1
```

---

## 3. Specification for `scripts/deploy.py`

This is a command-line utility for the user to execute.

### 3.1. Imports

`argparse`, `sagemaker`, `tarfile`, `os`, `shutil`.

### 3.2. Command-Line Arguments

* `--artifact-path`: Required. Path to the directory containing the training outputs (`best_model.pth`, `feature_processor.joblib`).
* `--endpoint-name`: Required. The desired name for the SageMaker endpoint.
* `--aws-role`: Required. The AWS IAM Role ARN that SageMaker can assume to access model artifacts and create resources.
* `--instance-type`: Optional. The EC2 instance type for the endpoint. Default: `'ml.t2.medium'`.

### 3.3. Core Logic

1. **Initialization**: Parse args, get the SageMaker session and default S3 bucket.
2. **Artifact Packaging**:
    * Create a temporary directory named `package`.
    * Copy the `best_model.pth` and `feature_processor.joblib` from `--artifact-path` into `package/`.
    * Create `model.tar.gz` using the `tarfile` library.
    * Add the files from `package/` to the root of the tarball.
    * Add the `sagemaker/` directory (containing `inference.py` and `requirements.txt`) into the tarball under a `code/` prefix.
3. **Upload to S3**: Upload `model.tar.gz` to `s3://{sagemaker-default-bucket}/{endpoint-name}/model.tar.gz`.
4. **Model Creation**:
    * Instantiate `sagemaker.pytorch.model.PyTorchModel`.
    * Provide the S3 model data path, the `--aws-role`, `entry_point='inference.py'`, `source_dir='./code'`, framework versions (e.g., `py_version='py310'`, `framework_version='2.1'`).
5. **Deployment**:
    * Call the `.deploy()` method on the `PyTorchModel` object.
    * Pass `initial_instance_count=1`, `instance_type=args.instance_type`, and `endpoint_name=args.endpoint_name`.
6. **Cleanup**: Remove the local `package/` directory and `model.tar.gz`.

---

## 4. Verification and Unit Tests

Generate a test script at `tests/test_inference.py`. The goal is to test the inference logic *locally* without needing AWS credentials.

### 4.1. Mocks and Fixtures

* Use `unittest.mock` to create mock versions of `AdditiveAxisModel` and `FeatureProcessor`.
* Create a `pytest.fixture` named `mock_model_dir(tmp_path)` that creates a temporary directory and saves fake `best_model.pth` and `feature_processor.joblib` files inside it.

### 4.2. Test Cases

1. **`test_model_fn(mock_model_dir)`**:
    * Call `model_fn` with the path from the fixture.
    * Assert that it correctly loads the mock artifacts and returns a dictionary with mock model and processor objects.

2. **`test_input_fn()`**:
    * Get the sample request JSON string from `APP.md`, Appendix A.
    * Call `input_fn` with this string and `'application/json'`.
    * Assert that the output is a correctly parsed Python dictionary.

3. **`test_predict_fn()`**:
    * Create mock model and processor instances. The mock model's `forward` method should be configured to return a predictable dictionary of tensors.
    * Create a sample input object (dict).
    * Call `predict_fn` with the input object and a dictionary containing the mock model and processor.
    * Assert that the processor's `transform` method was called once.
    * Assert that the model's `forward` method was called once.
    * Assert that the return value is the predictable dictionary of tensors from the mock model.

4. **`test_output_fn()`**:
    * Create a sample prediction dictionary containing `torch.Tensor` values.
    * Call `output_fn` with this dictionary and `'application/json'`.
    * Assert that the output is a JSON string.
    * Parse the output JSON and assert that the values are floats, not tensors, and the structure matches `APP.md`, Appendix B.
