# Model Architecture: A Phased Approach from a Simple Baseline to an Interactive Model

This document outlines a multi-phase plan for our model's architecture. We will start with a simple, robust baseline to verify our pipeline and establish initial performance. We will then detail several incremental enhancements that can be explored to increase model complexity and predictive power.

## Phase 1: The Simple Additive Baseline Model

**Purpose:** To create a minimal, explainable model that is fast to train and easy to debug. This will serve as our performance baseline. In this phase, all pre-trained components are **frozen**.

### 1.1. Architecture

The architecture is a simple, parallel set of sub-networks whose outputs are summed. There is no interaction between the axes.

1. **Input Embedding Stage:** Features for each axis are embedded as defined in the Feature Representation document. The pre-trained Amenities transformer is **FROZEN**.
2. **MLP Sub-Networks:** Each axis's embedded vector is passed through its own independent MLP to produce a scalar contribution `P_i`.
3. **Aggregation Stage:** The final price is the sum of all contributions and a global bias term.
    `Predicted_Price = Global_Bias + P_location + P_size + P_amenities + P_quality + P_seasonality`

### 1.2. Data Flow, Shapes, and Trainable Parameters

| Axis | Input Dim | Sub-Network Architecture | Output Dim | Trainable Parameters (From Scratch) |
| :--- | :--- | :--- | :--- | :--- |
| **Location** | 48 | `Linear(48, 32) -> ReLU -> Linear(32, 1)` | 1 | `(48*32+32) + (32*1+1)` = 1,601 |
| **Size & Capacity**| 27 | `Linear(27, 32) -> ReLU -> Linear(32, 1)` | 1 | `(27*32+32) + (32*1+1)` = 929 |
| **Quality** | 15 | `Linear(15, 32) -> ReLU -> Linear(32, 1)` | 1 | `(15*32+32) + (32*1+1)` = 545 |
| **Amenities** | 384 | `Linear(384, 1)` *(Frozen Transformer)* | 1 | `(384*1+1)` = 385 |
| **Seasonality** | 2 | `Linear(2, 16) -> ReLU -> Linear(16, 1)` | 1 | `(2*16+16) + (16*1+1)` = 65 |
| **Embeddings** | - | Various `nn.Embedding` layers | - | `(250*16)+(80*8)+(10*4)+(15*4)+(8*4)` = 4,772 |
| **Global Bias** | - | `nn.Parameter(1)` | - | 1 |
| **Total** | | | | **~8,300** |

## Phase 2: Incremental Enhancements

After successfully training and evaluating the baseline, we will explore the following enhancements.

### Enhancement 2.i: Fine-Tuning the Pre-trained Text Model

**Purpose:** To specialize the general linguistic knowledge of the amenities transformer for our specific price prediction task.

* **Change:** Unfreeze the weights of the pre-trained `bge-small-en-v1.5` model.
* **Implementation:** The model will be trained end-to-end with **discriminative learning rates** (a low learning rate for the transformer, a higher rate for all other components).
* **Impact on Parameters:** The number of **trainable** parameters increases dramatically.
  * **New Trainable Parameters:** ~33,500,000 (from the transformer).
  * **Total Trainable Parameters:** `8,300 + 33,500,000` = **~33.51 Million**.

### Enhancement 2.ii: Adding a Transformer Interaction Layer

**Purpose:** To move from a purely additive model to an interactive one, allowing the model to learn complex relationships between the feature axes.

* **Change:** Insert a **Transformer Encoder Layer** between the embedding stage and the prediction heads.
* **Architectural Data Flow & Shapes (`d_model = 64`):**

| Step | Process | Tensor Shape Transformation | Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **1. Projection** | Project each of the 5 embedded axis vectors to `d_model=64`. | `(batch, D_axis) -> (batch, 64)` | `(48*64+64) + (27*64+64) + ...` = ~7,000 |
| **2. Stack** | Stack the 5 vectors into a sequence. | `5x (batch, 64) -> (batch, 5, 64)` | 0 |
| **3. Transformer Encoder** | Self-attention and FFN layers process the sequence. | `(batch, 5, 64) -> (batch, 5, 64)` | **~33,500** |
| **4. Prediction Heads** | Project each of the 5 contextualized vectors to a scalar. | `5x (batch, 64) -> 5x (batch, 1)` | `5 * (64*1+1)` = 325 |
| **5. Aggregation** | Sum contributions with the Global Bias. | `5x (batch, 1) -> (batch, 1)` | 1 |

* **Impact on Parameters:**
  * If added to the **baseline model**, this enhancement adds `7000 + 33500 + 325 + 1 =` **~41,000** new parameters.
  * The model becomes significantly more complex, and its explainability becomes contextual rather than purely additive.

### Enhancement 2.iii: Incorporating Richer Textual Data (Stretch Goal)

**Purpose:** To leverage the valuable but unstructured text from listing descriptions and guest reviews.

* **Change:** Add new "text" axes to the model (e.g., `Description` axis, `Reviews Text` axis).
* **Implementation:**
    1. Each new text axis would require its own pre-trained transformer to create an embedding (e.g., another `bge-small` model, adding **~33.5M** parameters per axis if fine-tuned).
    2. The resulting text embeddings would be incorporated as additional inputs into the model, ideally into the Transformer Interaction Layer (Enhancement 2.ii).
* **Impact:** This would represent a major increase in model size, complexity, and data processing requirements. It is a significant research extension to the core project.
