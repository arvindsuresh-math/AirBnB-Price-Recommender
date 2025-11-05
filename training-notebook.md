### 0. Setup and Installations

This cell prepares the Google Colab environment by mounting Google Drive, changing the working directory to our project folder to ensure all custom modules can be imported, installing the required Python packages, and handling Hugging Face authentication.

```python
# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Change Directory to Project Folder ---
# This is a crucial step that makes all local imports work seamlessly
import os
# IMPORTANT: Make sure this path matches the location of your project folder in Google Drive
PROJECT_PATH = '/content/drive/MyDrive/Airbnb_Price_Project'
os.chdir(PROJECT_PATH)
print(f"Current working directory: {os.getcwd()}")

# --- Install Dependencies ---
!pip install pandas
!pip install pyarrow
!pip install sentence-transformers
!pip install scikit-learn
!pip install torch
!pip install tqdm
!pip install transformers
!pip install matplotlib
!pip install seaborn

# --- Hugging Face Authentication ---
from google.colab import userdata
from huggingface_hub import login
print("\nAttempting Hugging Face login...")
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
    login(token=HF_TOKEN)
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Could not log in. Please ensure 'HF_TOKEN' is a valid secret. Error: {e}")

```

### 1. Imports and Global Configuration

Here, we import all necessary functions and classes from our custom Python scripts. We also load the central `config` dictionary, set the global random seed for reproducibility, and confirm which compute device (`cuda` or `cpu`) is being used for the session.

```python
import torch
import torch.optim as optim
import pandas as pd

# Import from our custom scripts
from config import config
from utils import set_seed, plot_target_distributions, plot_training_history
from data_processing import load_and_split_data, FeatureProcessor, create_dataloaders
from model import AdditiveAxisModel
from train import train_model, save_artifacts
from build_app_dataset import build_dataset

# Set the seed for the entire notebook
set_seed(config['SEED'])

# Confirm the device
print(f"\nUsing device: {config['DEVICE']}")
if config['DEVICE'] == 'cuda':
    print(f"GPU Name: {torch.cuda.get_name(0)}")
```

### 2. Step 1: Load and Split Data

We now load the raw dataset and perform the stratified group split using the `load_and_split_data` function. This ensures that all records for a given `listing_id` are isolated to either the training or validation set, preventing data leakage.

```python
# Load and split the data according to the new stratified group logic
train_df, val_df, neighborhood_log_means, train_ids, val_ids = load_and_split_data(config)

# Print the shapes to confirm the split
print(f"\nTraining DataFrame shape: {train_df.shape}")
print(f"Validation DataFrame shape: {val_df.shape}")
```

### 3. Step 2: Visualize Target Distributions

This cell calls the `plot_target_distributions` function to generate a set of visualizations. This serves as a critical sanity check to confirm that our stratified split has produced training and validation sets with similar distributions for price, log-price, and our final target variable (log-price deviation).

```python
# Plot the distributions of the target variable and its components
plot_target_distributions(train_df, val_df, neighborhood_log_means)
```

### 4. Step 3: Process Features

With the data split and verified, we now prepare it for the model. We instantiate the `FeatureProcessor`, fit it exclusively on the training data to learn vocabularies and scaling parameters, and then use the fitted processor to transform both the training and validation sets into numerical features.

```python
# Instantiate and fit the feature processor
processor = FeatureProcessor(config)
processor.fit(train_df)

# Transform both datasets into feature dictionaries
train_features = processor.transform(train_df, neighborhood_log_means)
val_features = processor.transform(val_df, neighborhood_log_means)
```

### 5. Step 4: Instantiate Model and DataLoaders

We create the core PyTorch objects for training. First, we instantiate our `AdditiveAxisModel`, then call the `count_parameters()` method to get a detailed summary of its architecture. Finally, we create the `train_loader` and `val_loader` which will handle batching and data shuffling.

```python
# Instantiate the model
model = AdditiveAxisModel(processor, config)

# Print the breakdown of trainable and frozen parameters
model.count_parameters()

# Create the DataLoaders
train_loader, val_loader = create_dataloaders(train_features, val_features, config)
```

### 6. Step 5: Define Optimizer and Scheduler

Here, we define the optimization components. We create two separate parameter groups to apply a much lower learning rate to the pre-trained text transformer's fine-tuned layer. We then instantiate the AdamW optimizer and the `ReduceLROnPlateau` scheduler, which will automatically reduce the learning rate if validation performance stagnates.

```python
# Create parameter groups for differential learning rates
transformer_params = model.text_transformer.parameters()
other_params = [p for n, p in model.named_parameters() if 'text_transformer' not in n]

# Instantiate the optimizer
optimizer = optim.AdamW([
    {'params': other_params, 'lr': config['LEARNING_RATE']},
    {'params': transformer_params, 'lr': config['TRANSFORMER_LEARNING_RATE']}
])

# Instantiate the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config['SCHEDULER_FACTOR'],
    patience=config['SCHEDULER_PATIENCE']
)
print("Optimizer and Scheduler have been defined.")
```

### 7. Step 6: Train the Model

This is the main training step. We call the `train_model` function, which encapsulates the entire training loop, including forward/backward passes, optimization, evaluation, and early stopping. The best performing model state and a history of performance metrics are returned.

```python
# Run the training loop
trained_model, history_df = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config
)
```

### 8. Step 7: Analyze Training History

Immediately after training, we visualize the results by passing the `history_df` to the `plot_training_history` function. This dual-axis plot allows us to inspect the training/validation loss curves and the validation MAPE to assess for overfitting and confirm that the model converged properly.

```python
# Plot the training and validation loss/MAPE curves
plot_training_history(history_df)
```

### 9. Step 8: Save Model Artifacts

Now that the model is trained, we save all the essential components—the trained model's state dictionary, the fitted feature processor, and the configuration—to a single timestamped `.pt` file. This allows us to easily load and reuse the entire pipeline for inference later.

```python
# Save the trained model, processor, and config to a file
saved_artifacts_path = save_artifacts(trained_model, processor, config)
```

### 10. Step 9: Build and Save Final Application Dataset

This is the final "production" step. We package the necessary objects from our session into a dictionary and pass them to the `build_dataset` function. This function will perform the computationally expensive tasks of augmenting the dataset for all months and running inference on the entire panel, creating the self-contained database for our Streamlit application.

```python
# Package the necessary objects for the build process
artifacts_for_build = {
    'model': trained_model,
    'processor': processor,
    'train_ids': train_ids,
    'val_ids': val_ids
}

# Run the build process
build_dataset(artifacts_for_build)
```

### 11. Step 10: Verify Final Dataset

As a final sanity check, we load the application database that was just created. We then display its schema using `.info()` and a few sample rows with `.head()` to confirm that it has the correct structure, includes all the new `p_*`, `pm_*`, and `h_*` columns, and is ready for use in the analysis notebook and the final web application.

```python
# Construct the path to the newly created app database
app_data_path = os.path.join(config['DRIVE_SAVE_PATH'], 'app_data', f"{config['CITY']}_app_database.parquet")

# Load and inspect the final dataset
print(f"Loading final app database from: {app_data_path}")
final_app_df = pd.read_parquet(app_data_path)

print("\n--- Final App Database Info ---")
final_app_df.info()

print("\n--- Final App Database Head ---")
display(final_app_df.head())
```