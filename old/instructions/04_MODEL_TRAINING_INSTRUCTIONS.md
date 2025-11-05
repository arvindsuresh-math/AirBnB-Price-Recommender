# `04_MODEL_TRAINING_INSTRUCTIONS.md`

### LLM Agent Pre-computation Instructions

To enrich your context, first read and fully comprehend the following project design documents in order:

1. `01_ETL_INSTRUCTIONS.md`: To understand the origin and schema of the input data Parquet file.
2. `02_PYTORCH_DATASET_INSTRUCTIONS.md`: To know the exact class names (`FeatureProcessor`, `AirbnbDataset`) and methods you will need to import and instantiate.
3. `03_PYTORCH_MODEL_INSTRUCTIONS.md`: To know the exact model class name (`AdditiveAxisModel`) and constructor arguments you will need to use.
4. `TARGET-PRICE.md`: For the critical rationale behind using the `estimated_occupancy_rate` column as a sample weight in the loss function.

### Primary Objective

Your task is to generate a single, comprehensive Python script located at `scripts/train.py`. This script will serve as the main entry point for training and evaluating the model. It must orchestrate data loading, pre-processing, model training, validation, and artifact saving in a clean, reproducible manner.

---

## 1. Script Structure and Argument Parsing

The script must be executable from the command line and configurable via arguments.

### 1.1. Imports

* Import necessary libraries: `argparse`, `pandas`, `torch`, `torch.nn`, `torch.optim`, `sklearn.model_selection`, `joblib`, and others.
* Import the custom classes you will use from other modules:
  * `from src.dataset import FeatureProcessor, AirbnbDataset`
  * `from src.model import AdditiveAxisModel`

### 1.2. Command-Line Arguments

Use Python's `argparse` to define the following command-line arguments:

* `--data-path`: Required. Path to the input Parquet file (e.g., `data/processed/nyc_modeling_dataset.parquet`).
* `--output-dir`: Required. Directory where all output artifacts (model, processor, logs) will be saved (e.g., `artifacts/`).
* `--amenities-model-name`: Optional. The HuggingFace name of the sentence transformer. Default: `'all-MiniLM-L6-v2'`.
* `--epochs`: Optional. Number of training epochs. Default: `10`.
* `--batch-size`: Optional. Batch size for training and validation. Default: `64`.
* `--learning-rate`: Optional. Learning rate for the Adam optimizer. Default: `1.0e-3`.
* `--test-size`: Optional. Fraction of data to use for the validation set. Default: `0.2`.
* `--smoke-test`: Optional. A boolean flag (`action='store_true'`) to run a quick test on a small subset of data.

---

## 2. Core Orchestration Logic in `main` function

### Step 1: Setup and Configuration

* Parse the command-line arguments.
* Set the device: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
* Create the output directory if it doesn't exist (`os.makedirs(args.output_dir, exist_ok=True)`).

### Step 2: Data Loading and Splitting

* Load the Parquet file specified by `--data-path` into a pandas DataFrame.
* **Handle Smoke Test:** If the `--smoke-test` flag is set, reduce the DataFrame to its first 1000 rows.
* Perform a train/validation split using `sklearn.model_selection.train_test_split`. Use `--test-size` and a fixed `random_state=42` for reproducibility.

### Step 3: Feature Processing

* Instantiate `FeatureProcessor`.
* Fit the processor on the **training** DataFrame: `processor.fit(train_df)`.
* Save the fitted processor to the output directory: `processor.save(os.path.join(args.output_dir, "feature_processor.joblib"))`.
* Transform both the training and validation DataFrames:
  * `train_processed_data = processor.transform(train_df)`
  * `val_processed_data = processor.transform(val_df)`

### Step 4: PyTorch Data Loaders

* Instantiate the `AirbnbDataset` for both training and validation sets, passing the respective processed data dictionaries and the `--amenities-model-name`.
* Create `torch.utils.data.DataLoader` for both datasets.
  * `train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)`
  * `val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)`

### Step 5: Model, Loss, and Optimizer Initialization

* **Model:**
  * Prepare the `vocab_sizes` and `embedding_dims` dictionaries required by the `AdditiveAxisModel` constructor by inspecting the fitted `processor.vocabs`.
  * Instantiate the model: `model = AdditiveAxisModel(vocab_sizes, embedding_dims).to(device)`.
* **Loss Function:**
  * Instantiate the Mean Squared Error loss with no reduction: `criterion = nn.MSELoss(reduction='none')`.
* **Optimizer:**
  * Instantiate the Adam optimizer: `optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)`.

### Step 6: The Training and Validation Loop

* Loop for the specified number of epochs.
* Inside the loop, call separate functions for the training and validation phases.

#### 6.1. Training Phase (within a `train_one_epoch` function)

* Set the model to training mode: `model.train()`.
* Initialize a running loss tracker.
* Iterate over the `train_loader`. For each batch:
    1. Move all tensors in the batch dictionary to the `device`.
    2. Zero the gradients: `optimizer.zero_grad()`.
    3. Perform the forward pass: `outputs = model(batch)`.
    4. Extract predictions, target, and sample weights: `predictions = outputs['predicted_price']`, `targets = batch['target_price']`, `weights = batch['estimated_occupancy_rate']`.
    5. **Calculate Weighted Loss (Crucial Step):**
        * `raw_loss = criterion(predictions, targets)`
        * `weighted_loss = (raw_loss * weights).mean()`
    6. Perform backpropagation: `weighted_loss.backward()`.
    7. Update weights: `optimizer.step()`.
    8. Update the running loss.
* Return the average training loss for the epoch.

#### 6.2. Validation Phase (within an `evaluate` function)

* Set the model to evaluation mode: `model.eval()`.
* Initialize a running validation loss tracker.
* Wrap the entire evaluation in `with torch.no_grad():`.
* Iterate over the `val_loader`. For each batch:
    1. Move all tensors to the `device`.
    2. Perform the forward pass.
    3. Calculate the weighted loss using the same logic as in training.
    4. Update the running validation loss.
* Return the average validation loss for the epoch.

### Step 7: Model Checkpointing and Saving

* After each epoch, compare the current validation loss to the best validation loss seen so far.
* If the current loss is better, save the model's state dictionary: `torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))`.
* Update the best validation loss.
* Print a log of the epoch's training loss and validation loss.

---

## 3. Verification and Smoke Testing

The primary verification method is integrated directly into the script.

### Smoke Test Logic

* As specified in Section 2, if the `--smoke-test` command-line flag is provided, the script's logic must handle it.
* The test is defined as follows: the script must successfully load a small subset of the data, initialize all components, and complete **one full training epoch and one full validation pass** without raising any errors.
* The test is successful if the script exits with code 0.

### Test Instructions

Generate a test script at `tests/test_training.py`.

1. **`test_smoke_run(tmp_path)`**:
    * `tmp_path` is a `pytest` fixture for a temporary directory.
    * Use `subprocess.run` to execute the `scripts/train.py` script from the command line.
    * Pass the following arguments:
        * `--data-path` pointing to the sample Parquet file (you may need a fixture to generate this first by running the ETL script on sample data).
        * `--output-dir` pointing to the `tmp_path`.
        * `--epochs 1`.
        * `--batch-size 16`.
        * `--smoke-test`.
    * Assert that the subprocess completes with a return code of `0`.
    * Assert that the expected output artifacts (`best_model.pth`, `feature_processor.joblib`) are present in the temporary output directory.
