# inference.py

"""
Contains functions for running inference with the trained model.
"""
import torch
from tqdm.notebook import tqdm
from app.src.model import AdditiveAxisModel


def run_inference_with_details(model: AdditiveAxisModel, data_loader: torch.utils.data.DataLoader, device: str) -> dict:
    """
    Runs inference and returns the full decomposition for each listing.

    This function returns not just the final prediction, but also the hidden
    states and additive price contributions from each of the model's sub-networks.

    Args:
        model (AdditiveAxisModel): The trained model.
        data_loader (DataLoader): A DataLoader for the dataset to be processed.
        device (str): The device to run inference on ('cuda' or 'cpu').

    Returns:
        dict: A dictionary where keys are output names (e.g., 'p_location',
              'h_amenities') and values are numpy arrays of these outputs for
              the entire dataset.
    """
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Running Inference", leave=False):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
                else: batch[k] = {sk: sv.to(device) for sk, sv in v.items()}

            with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                batch_outputs = model.forward_with_hidden_states(batch)

            batch_outputs['neighborhood_log_mean'] = batch['neighborhood_log_mean']
            outputs.append({k: v.cpu() for k, v in batch_outputs.items()})

    # Concatenate results from all batches into single numpy arrays
    final_outputs = {key: torch.cat([o[key] for o in outputs]).numpy() for key in outputs[0].keys()}

    # Reconstruct final predicted price from the log-space components
    predicted_log = final_outputs['predicted_log_deviation'] + final_outputs['neighborhood_log_mean']
    final_outputs['predicted_price'] = np.expm1(predicted_log)
    return final_outputs