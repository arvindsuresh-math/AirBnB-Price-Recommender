# MODEL-TRAINING-INSTRUCTIONS.md

## 1. Objective

This specification details the creation of a Python script that orchestrates the end-to-end model training and evaluation process. The script will use the components defined in `data_loader.py` and `model.py` to train the **Phase 1 Baseline Model**, log metrics, and save the best performing model artifact.

## 2. File and Path Specifications

### 2.1. Script Location and Name

* **Location:** `/Fall-2025-Team-Big-Data/scripts/`
* **Name:** `02_train_model.py`

### 2.2. Input Artifacts

* **Modeling Data:** `/Fall-2025-Team-Big-Data/data/processed/nyc/nyc-modeling-data-YYYY-MM.parquet`
* **Feature Artifacts:** `/Fall-2025-Team-Big-Data/data/processed/nyc/nyc-feature-artifacts-YYYY-MM.json`
    *(The script should accept the `YYYY-MM` version string as a command-line argument to locate these files dynamically.)*

### 2.3. Output Artifacts

* **Output Directory:** `/Fall-2025-Team-Big-Data/models/YYYY-MM/`
  * *The script must create this versioned directory.*
* **Output 1: Best Model Checkpoint**
  * **Name:** `best_model.pth`
  * **Content:** The `state_dict` of the model that achieved the lowest validation loss.
* **Output 2: Training Log**
  * **Name:** `training_log.csv`
  * **Content:** A CSV file logging the performance for each epoch, with columns: `epoch`, `train_loss`, `val_loss`, `train_mae`, `val_mae`.
* **Output 3: TensorBoard Logs (Optional but Recommended)**
  * **Directory:** `/Fall-2025-Team-Big-Data/logs/YYYY-MM/`

## 3. Configuration and Hyperparameters

The script must be configurable via command-line arguments (using `argparse`).

* `--version-string`: (Required) The `YYYY-MM` string.
* `--batch-size`: Default `64`.
* `--learning-rate`: Default `1e-3`.
* `--epochs`: Default `20`.
* `--device`: Default `'cuda'` if `torch.cuda.is_available()` else `'cpu'`.

## 4. Step-by-Step Execution Plan

### Step 1: Initialization and Setup

1. Parse command-line arguments.
2. Set up logging (e.g., Python's `logging` module) to print progress to the console.
3. Set the random seed for reproducibility: `torch.manual_seed(42)`.
4. Define the device (CUDA or CPU).
5. Create the output directories.

### Step 2: Data Loading

1. Construct the full paths to the `.parquet` and `.json` files using the `version-string`.
2. Import the `create_dataloaders` function from `src.data_loader`.
3. Call `train_loader, val_loader, test_loader = create_dataloaders(...)` to get the data loaders.

### Step 3: Model, Optimizer, and Loss Function Setup

1. Import the `ExplainablePriceModel` class from `src.model`.
2. Instantiate the model: `model = ExplainablePriceModel(artifacts_path=...)`.
3. Move the model to the specified device: `model.to(device)`.
4. Instantiate the optimizer: `optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)`.
5. Define the loss function. Since we are using sample weights, we must use a **reduction of 'none'**.
    * `loss_fn = nn.MSELoss(reduction='none')`

### Step 4: The Main Training Loop

1. Loop for `args.epochs` number of epochs.
2. **Training Phase (per epoch):**
    * Set the model to training mode: `model.train()`.
    * Initialize running totals for loss and Mean Absolute Error (MAE).
    * Iterate through the `train_loader` with `tqdm` for a progress bar. For each `batch`:
        a. Move all tensors in the batch to the `device`.
        b. Perform the forward pass: `outputs = model(batch['inputs'])`.
        c. Calculate the un-reduced loss: `unweighted_loss = loss_fn(outputs['predicted_price'], batch['target'])`.
        d. **Apply the sample weights:** `weighted_loss = unweighted_loss * batch['weight']`.
        e. Calculate the final batch loss: `final_loss = weighted_loss.mean()`.
        f. Perform backpropagation: `final_loss.backward()`.
        g. Update the weights: `optimizer.step()`.
        h. Zero the gradients: `optimizer.zero_grad()`.
        i. Update running totals for logging.
3. **Validation Phase (per epoch):**
    * Set the model to evaluation mode: `model.eval()`.
    * Use a `with torch.no_grad():` block to disable gradient calculations.
    * Initialize running totals for validation loss and MAE.
    * Iterate through the `val_loader`. For each `batch`:
        a. Perform the forward pass.
        b. Calculate the weighted loss (same logic as training).
        c. Update validation running totals.
4. **Logging and Checkpointing (per epoch):**
    * Calculate the average train/val loss and MAE for the epoch.
    * Print the results to the console.
    * Append the results to the `training_log.csv` file.
    * Log metrics to TensorBoard (if using).
    * Compare the current `val_loss` with the best `val_loss` seen so far. If it is lower, update the best loss and save the model's `state_dict` to `/path/to/best_model.pth`.

### Step 5: Final Evaluation (After Training Loop)

1. Load the best performing model checkpoint: `model.load_state_dict(torch.load(...))`.
2. Perform a final evaluation pass on the `test_loader` using the same logic as the validation loop.
3. Print the final test loss and test MAE to the console. This is the final reported performance of the model.

## 5. Phase 2 Enhancements (To Be Implemented Later)

The script must include comments indicating where to make changes for advanced training procedures.

* **For Fine-Tuning (Enhancement 2.i):**
  * Add a comment in **Step 3** where the optimizer is defined.
  * `# PHASE 2.i: To enable fine-tuning, replace the standard optimizer with the discriminative learning rate setup.`
  * `# See PYTORCH-MODEL-INSTRUCTIONS.md for the parameter grouping logic.`

## 6. Debugging and Robustness Instructions

To ensure the training process is stable and easy to debug, the following features must be included in the script.

1. **Gradient Clipping:** In the training loop, immediately after the `loss.backward()` call, add a gradient clipping step to prevent exploding gradients, which can lead to training instability (NaN losses).
    * `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
2. **Sanity Check Logging:** At the beginning of the script's execution, use the `logging` module to print a summary of all key hyperparameters and configurations (learning rate, batch size, epochs, device, version string).
3. **Fast Fail on Data Loading:** Immediately after the `create_dataloaders` call, add an assertion to verify that the datasets are not empty. This prevents the script from running a full epoch before failing on a data loading issue.
    * `assert len(train_loader.dataset) > 0 and len(val_loader.dataset) > 0`
