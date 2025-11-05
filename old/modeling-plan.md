### 1. Required Scripts and Their Responsibilities

This structure promotes the **Don't Repeat Yourself (DRY)** principle. The scripts will contain all the core, reusable logic, while the notebooks will be clean, readable narratives that import and execute this logic.

**`config.py`**
*   **Purpose:** The single source of truth for all hyperparameters and settings.
*   **Contents:** A single Python dictionary named `config`.
*   **Functions:** None.

---

**`data_processing.py`**
*   **Purpose:** Handles all data loading, splitting, and feature transformation.
*   **Functions/Classes:**
    *   `load_and_split_data(config)`: Loads the dataset, filters, and performs the stratified group-based train/validation split. Returns the dataframes, neighborhood means, and listing ID sets.
    *   `class FeatureProcessor`: The same class you developed. It learns transformations from the training data (`.fit()`) and applies them to any dataset (`.transform()`). This is crucial for preventing data leakage.
    *   `create_dataloaders(train_features, val_features, config)`: Takes the transformed feature dictionaries and prepares the PyTorch `DataLoader` objects for training and validation.

---

**`model.py`**
*   **Purpose:** Defines all neural network architectures.
*   **Functions/Classes:**
    *   `class AirbnbPriceDataset(Dataset)`: The PyTorch Dataset class that handles on-the-fly tokenization.
    *   `class BaselineModel(nn.Module)`: The final, regularized, fully-connected baseline network (128 -> 32, with dropout and batch norm).
    *   `class AdditiveModel(nn.Module)`: The final, regularized additive axis model. It will be renamed from `AdditiveAxisModelV2` and will have dropout and regularization built-in, but the ablation/freezing logic will be removed to keep the final version clean and focused.
    *   `class AblationAdditiveModel(nn.Module)`: A separate model class used *only* in `ablations.ipynb`. It will contain the logic to dynamically exclude axes during its forward pass.

---

**`train.py`**
*   **Purpose:** Contains all logic related to the model training and evaluation loop.
*   **Functions/Classes:**
    *   `train_model(...)`: The main training loop. It will take a model, data loaders, optimizer, and config, and handle the epoch loop, forward/backward passes, optimization, early stopping, and LR scheduling. It returns the best trained model and a history dataframe.
    *   `evaluate_model(...)`: Calculates and returns loss and MAPE for a given model and data loader. Used by `train_model` for validation and for final evaluation.
    *   `run_ablation_experiment(...)`: A high-level wrapper function used *only* in `ablations.ipynb`. It will instantiate `AblationAdditiveModel`, train it, evaluate it, and return a dictionary of results. This keeps the ablation notebook extremely clean.

---

**`inference.py`**
*   **Purpose:** Contains functions for generating predictions from trained models.
*   **Functions/Classes:**
    *   `run_inference(model, data_loader, device)`: A generic function that takes any trained model and a data loader, and returns a DataFrame with predictions. Perfect for the `BaselineModel`.
    *   `run_inference_with_details(model, data_loader, device)`: The function you already have. It's specifically for the `AdditiveModel` and returns the final prediction *plus* all the intermediate price contributions (`p_*`) and hidden states (`h_*`).

---

**`plotting.py`**
*   **Purpose:** A new script to house all visualization functions. This makes the `results.ipynb` notebook incredibly clean and focused on analysis rather than plotting code.
*   **Functions/Classes:**
    *   `plot_predictions_vs_actual(results_df, model_name)`: Generates a single scatter plot of true vs. predicted prices.
    *   `plot_mape_distribution(results_df, model_name)`: Generates a histogram or KDE plot of the Mean Absolute Percentage Error, which is more informative than a single number.
    *   `plot_ablation_results(ablation_df)`: Creates a bar chart visualizing the drop in performance when each axis is removed.
    *   `plot_additive_contributions(listing_data)`: Creates a waterfall or bar chart showing how the price of a *single listing* is built from the base price plus/minus the contributions of each axis.

---
### 2. Skeleton of Each Jupyter Notebook

Here is a cell-by-cell outline for each notebook. Each `[Markdown]` cell is for exposition, and each `[Code]` cell is for execution.

#### 1. `01_baseline_nn_model.ipynb`

*   **[Markdown] Cell 1: Title and Introduction**
    *   Title: `Baseline Neural Network Model`
    *   Objective: "In this notebook, we train and evaluate a fully-connected neural network. This model will serve as a robust baseline to measure the performance of our more complex, interpretable additive model. It uses all available features concatenated into a single vector."
*   **[Code] Cell 2: Setup and Imports**
    *   Mount Google Drive, `os.chdir`.
    *   Import all necessary functions and classes from your `.py` scripts.
*   **[Markdown] Cell 3: Data Loading and Preprocessing**
    *   "We begin by loading the dataset and performing the stratified group split to ensure no data leakage between training and validation sets. We then use our `FeatureProcessor` to convert the raw data into numerical tensors."
*   **[Code] Cell 4: Execute Data Pipeline**
    *   Call `load_and_split_data`, `processor.fit/transform`, `create_dataloaders`.
*   **[Markdown] Cell 5: Model Initialization**
    *   "Here, we instantiate our `BaselineModel`, which consists of a simple MLP with regularization. We also define the AdamW optimizer and a learning rate scheduler."
*   **[Code] Cell 5: Instantiate Model and Optimizer**
    *   `model = BaselineModel(...)`, `optimizer = optim.AdamW(...)`, `scheduler = ...`
*   **[Markdown] Cell 6: Model Training**
    *   "We now pass all the components to our `train_model` function, which handles the entire training loop, including validation, early stopping, and saving the best model state."
*   **[Code] Cell 7: Run Training**
    *   `trained_model, history = train_model(...)`
*   **[Markdown] Cell 8: Final Evaluation and Prediction Generation**
    *   "With the best model checkpoint loaded, we perform a final evaluation on the validation set and then run inference on the full dataset to generate predictions for later analysis."
*   **[Code] Cell 9: Evaluate and Predict**
    *   `final_metrics = evaluate_model(...)`, print metrics.
    *   `predictions_df = run_inference(...)`
*   **[Markdown] Cell 10: Save Artifacts**
    *   "Finally, we save the trained model weights, the fitted feature processor, the final metrics, and the predictions DataFrame to disk. These artifacts will be loaded in the `04_results.ipynb` notebook for analysis."
*   **[Code] Cell 11: Save All Outputs**
    *   `torch.save(...)`, `predictions_df.to_parquet(...)`

---
#### 2. `02_additive_nn_model.ipynb`

*This notebook will be nearly identical in structure to the baseline notebook, which is a strength, showing a consistent workflow.*

*   **[Markdown] Cell 1: Title and Introduction**
    *   Title: `Final Additive Explainable Model`
    *   Objective: "This notebook details the training of our primary model: the `AdditiveModel`. Its architecture is designed for explainability, predicting price as a sum of contributions from different feature axes (Location, Quality, Amenities, etc.)."
*   **...Cells 2-7...**
    *   These will mirror the baseline notebook, but will instantiate `AdditiveModel` and use the optimizer with differential learning rates.
*   **[Markdown] Cell 8: Final Evaluation and Detailed Prediction Generation**
    *   "We run a final evaluation and then use the `run_inference_with_details` function to get not only the final price prediction but also the individual contribution of each axis (`p_*`), which is the core of our model's explainability."
*   **[Code] Cell 9: Evaluate and Predict with Details**
    *   `final_metrics = evaluate_model(...)`, print metrics.
    *   `predictions_with_details_df = run_inference_with_details(...)`
*   **[Markdown] & [Code] Cells 10-11: Save Artifacts**
    *   Save the model, processor, and the detailed predictions DataFrame.

---
#### 3. `03_ablation_study.ipynb`

*   **[Markdown] Cell 1: Title and Introduction**
    *   Title: `Additive Model Ablation Study`
    *   Objective: "To quantify the importance of each feature axis in our `AdditiveModel`, we conduct an ablation study. We systematically remove each axis one by one, retrain the model, and record the impact on performance. This tells us which components are most critical to the model's accuracy."
*   **[Code] Cell 2: Setup and Imports**
    *   Standard imports, but this time, `from model import AblationAdditiveModel` and `from train import run_ablation_experiment`.
*   **[Markdown] Cell 3: Data Loading**
    *   "First, we prepare the data loaders. They will be reused for every experiment in the study."
*   **[Code] Cell 4: Load Data**
    *   Call `load_and_split_data`, `processor.fit/transform`, `create_dataloaders`.
*   **[Markdown] Cell 5: Running the Experiments**
    *   "We first establish a baseline by training the full model. Then, we loop through each of the six axes, calling our `run_ablation_experiment` function to handle the training and evaluation for each variation."
*   **[Code] Cell 6: Experiment Loop**
    *   `axes_to_ablate = ['location', 'size_capacity', ...]`
    *   `results = []`
    *   Run baseline (exclude `[]`).
    *   `for axis in axes_to_ablate: ... results.append(run_ablation_experiment(exclude_axes=[axis], ...))`
*   **[Markdown] Cell 7: Saving Results**
    *   "We compile the results into a single DataFrame and save it to a CSV file. This file will be loaded in the final analysis notebook to visualize the findings."
*   **[Code] Cell 8: Compile and Save Results**
    *   `results_df = pd.DataFrame(results)`
    *   Display `results_df.head()`.
    *   `results_df.to_csv(...)`

---
#### 4. `04_results_and_analysis.ipynb`

*   **[Markdown] Cell 1: Title and Introduction**
    *   Title: `Model Comparison and Results Analysis`
    *   Objective: "This notebook brings everything together. We load the saved predictions and results from our three models (Random Forest, Baseline NN, Additive NN) to perform a comprehensive comparison and dive deep into the interpretability of the final `AdditiveModel`."
*   **[Code] Cell 2: Setup and Imports**
    *   Import `pandas`, `matplotlib`, `seaborn`, and all functions from `plotting.py`.
*   **[Markdown] Cell 3: Load All Artifacts**
    *   "We load the prediction DataFrames generated by the training notebooks and the results of the ablation study."
*   **[Code] Cell 4: Load Files**
    *   `baseline_preds = pd.read_parquet(...)`
    *   `additive_preds = pd.read_parquet(...)`
    *   `rf_preds = pd.read_parquet(...)`
    *   `ablation_results = pd.read_csv(...)`
*   **[Markdown] Cell 5: Overall Performance Comparison**
    *   "Let's compare the models on two fronts: the relationship between true and predicted prices, and the distribution of their prediction errors (MAPE)."
*   **[Code] Cell 6: Plot Performance**
    *   Create a 1x3 subplot.
    *   Call `plot_predictions_vs_actual` for each model on one of the subplots.
    *   Create another 1x3 subplot.
    *   Call `plot_mape_distribution` for each model.
*   **[Markdown] Cell 7: Ablation Study Insights**
    *   "The ablation study reveals which features are most important to the Additive Model. A larger drop in performance when an axis is removed indicates higher importance."
*   **[Code] Cell 8: Plot Ablation Results**
    *   `plot_ablation_results(ablation_results)`
*   **[Markdown] Cell 9: Deep Dive: Explaining a Single Prediction**
    *   "The standout feature of the `AdditiveModel` is its explainability. Here, we select a single expensive listing and visualize how its final predicted price is constructed from the neighborhood baseline and the contributions of each axis."
*   **[Code] Cell 10: Plot Single Contribution**
    *   Select one row from `additive_preds`.
    *   `plot_additive_contributions(listing_data=that_row)`
*   **[Markdown] Cell 11: Conclusion**
    *   Summarize the key findings. Which model performed best overall? Which features were most important? How does the explainability of the additive model provide value beyond a simple prediction?