# Methodology for Generating Market Comparables

## 1. Problem Definition and Baselines

Generating a relevant set of comparable listings (a **comp set**) is a nearest neighbor search problem. The primary challenge lies in defining an appropriate distance metric that quantifies the "similarity" between two listings from a market pricing perspective.

### 1.1. Limitations of Price-Blind Metrics

Standard unsupervised methods, such as k-Nearest Neighbors on scaled features or clustering within a Variational Autoencoder's (VAE) latent space, are "price-blind." These methods typically rely on an L2 norm in the feature space, which implicitly assigns equal importance to all features. This is suboptimal, as it fails to account for the heterogeneous contributions of different feature groups (e.g., location, size, amenities) to the final market price.

### 1.2. The Pitfall of Circular Reasoning

An alternative is to explicitly incorporate price into the similarity learning process. However, this constitutes a form of **cheating** and creates **circular reasoning**. The purpose of the comp set is to *justify* a recommended price. If we generate the comp set by explicitly searching for listings that are already similar in price, we are tainting the evidence. The user is shown listings that are similar in price simply because we asked the model to find them, not because they are truly comparable based on their intrinsic features. This undermines the tool's transparency and trustworthiness.

Our goal is to use the model's understanding of price to guide our *search strategy*, not to be the *subject* of the search itself.

## 2. A Two-Stage, Dynamically Weighted Retrieval System

To address these limitations, we propose a two-stage retrieval and ranking system. This system leverages the internal representations learned by our primary additive pricing model to construct a similarity metric that is both "price-aware" and avoids the circularity of direct price conditioning.

### 2.1. Stage 1: Candidate Generation via Multi-Index Retrieval

This stage is designed for efficient, large-scale retrieval with high recall. Its objective is to generate a small but relevant subset of candidate listings from the entire database.

1. **Offline Indexing:** After the main additive model is trained, we extract the latent vectors (`z_i`) from the penultimate layer of each of the **5** sub-networks. We then construct **5** separate Approximate Nearest Neighbor (ANN) indexes, one for each of the **5** latent spaces.
2. **Online Retrieval:** For a given query listing `A`, we retrieve its **5** latent vectors, `(z_1(A), ..., z_5(A))`. We then query each of the **5** ANN indexes in parallel, retrieving the top `m` nearest neighbors from each index (e.g., `m=200`).
3. **Candidate Pooling:** The retrieved sets of `listing_ids` are aggregated and deduplicated to form a single candidate pool of size at most `5 * m`.

### 2.2. Stage 2: Re-ranking with a Normalized Dynamic Metric

This stage applies a more computationally intensive and precise scoring function to the limited candidate pool to ensure high precision in the final result.

1. **The Normalized Dynamic Metric:** The distance between the query listing `A` and a candidate listing `B` is a weighted sum of the Euclidean distances within each latent space. The weights are **normalized** to represent the *relative importance* of each axis for the query listing `A`.

    The calculation for a given query `A` is as follows:
    * First, calculate the absolute price contribution for each axis: `|P_i(A)|`.
    * Next, calculate the sum of all absolute contributions: `Total_P(A) = Σ |P_j(A)|`.
    * The final normalized weight for axis `i` is: `w_i(A) = |P_i(A)| / Total_P(A)`. These weights sum to 1.
    * The final distance formula is: `Distance(A, B) = Σ w_i(A) * ||z_i(A) - z_i(B)||` (from i=1 to 5).

2. **Justification:** This metric uses the model's learned market knowledge (`|P_i(A)|`) to define similarity. Normalizing the weights makes the distance metric **scale-invariant** with respect to the listing's absolute price. The metric focuses purely on the *relative importance* of the feature axes, making the resulting distance scores more stable and directly comparable across different queries.

3. **Example:** Consider finding comps for **Listing A**, a 1-bedroom apartment in a prime downtown neighborhood. The model predicts its price is driven heavily by location, accounting for **60%** of its value (normalized weight `w_loc = 0.6`), but less so by its modest size, which accounts for only **8%** (`w_size = 0.08`).
    * A **price-blind metric** might consider **Listing B**—a 4-bedroom house in a remote suburb—a reasonable comp if they share similar amenities or review scores, which is incorrect.
    * Our **dynamic weighted metric**, when comparing `A` to `B`, will compute a distance dominated by the location term: `0.6 * ||z_loc(A) - z_loc(B)||`. Since their locations are very different, this term will be large, correctly identifying `B` as a poor comparable. When comparing `A` to **Listing C**—another 1-bedroom in the same neighborhood—the location distance will be small, resulting in a low overall distance and correctly identifying `C` as a strong comparable.

4. **Final Selection:** The dynamic distance is calculated for all candidates in the pool. The candidates are then sorted by this distance, and the top `N` listings are selected as the final comp set.
