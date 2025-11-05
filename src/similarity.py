"""
Contains all logic for the nearest neighbor similarity search.
"""

import numpy as np
from scipy.spatial.distance import cdist

def haversine_distance(latlon1_rad: np.ndarray, latlon2_rad: np.ndarray) -> np.ndarray:
    """Calculates the Haversine distance in miles between a point and an array of points."""
    dlon = latlon2_rad[:, 1] - latlon1_rad[1]
    dlat = latlon2_rad[:, 0] - latlon1_rad[0]
    a = np.sin(dlat / 2.0)**2 + np.cos(latlon1_rad[0]) * np.cos(latlon2_rad[:, 0]) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3959 * c  # Earth radius in miles

def euclidean_distance(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Calculates the Euclidean distance from a vector to every row in a matrix."""
    return cdist(vector.reshape(1, -1), matrix, 'euclidean').flatten()

def cosine_distance(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Calculates the Cosine distance from a vector to every row in a matrix."""
    return cdist(vector.reshape(1, -1), matrix, 'cosine').flatten()

def calculate_axis_importances(p_contributions: dict, exclude_axes: list = None) -> dict:
    """Calculates normalized importance weights for each model axis."""
    exclude_axes = exclude_axes or []
    filtered_contributions = {k: v for k, v in p_contributions.items() if k not in exclude_axes}

    abs_contributions = {k: abs(v) for k, v in filtered_contributions.items()}
    total_abs_contribution = sum(abs_contributions.values())

    if total_abs_contribution == 0:
        num_axes = len(abs_contributions)
        return {k: 1.0 / num_axes for k in abs_contributions} if num_axes > 0 else {}

    return {k: v / total_abs_contribution for k, v in abs_contributions.items()}

def find_nearest_neighbors(query_idx: int, all_listing_ids: np.ndarray, lat_lon_rad: np.ndarray, 
                           price_contributions: dict, hidden_states: dict, 
                           top_k: int = 5, radius_miles: float = 2.0):
    """
    Finds the top K nearest neighbors for a listing within a geographic radius.
    """
    # 1. Geospatial Filtering
    query_lat_lon_rad = lat_lon_rad[query_idx]
    distances_miles = haversine_distance(query_lat_lon_rad, lat_lon_rad)
    candidate_indices = np.where((distances_miles > 0) & (distances_miles <= radius_miles))[0]

    query_id = all_listing_ids[query_idx]
    mask = (all_listing_ids[candidate_indices] != query_id)
    candidate_indices = candidate_indices[mask]

    if len(candidate_indices) == 0:
        return [], {}

    # 2. Calculate Axis-Importance Weights
    query_p_contribs = {name: p_vec[query_idx] for name, p_vec in price_contributions.items()}
    weights = calculate_axis_importances(query_p_contribs, exclude_axes=['location'])

    # 3. Calculate and combine weighted distances for candidates
    final_scores = np.zeros(len(candidate_indices))
    search_axes = [axis for axis in hidden_states.keys() if axis != 'location']

    for axis in search_axes:
        h_matrix = hidden_states[axis]
        query_h = h_matrix[query_idx]
        candidate_h = h_matrix[candidate_indices]

        dist_func = cosine_distance if axis in ["amenities", "description"] else euclidean_distance
        raw_dists = dist_func(query_h, candidate_h)

        min_dist, max_dist = raw_dists.min(), raw_dists.max()
        norm_dists = (raw_dists - min_dist) / (max_dist - min_dist + 1e-6)
        final_scores += weights.get(axis, 0) * norm_dists

    # 4. Find and return top_k results
    nearest_indices_in_candidates = np.argsort(final_scores)
    top_original_indices = candidate_indices[nearest_indices_in_candidates]
    
    return top_original_indices[:top_k], weights