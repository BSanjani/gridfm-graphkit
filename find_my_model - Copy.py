import torch

try:
    data = torch.load('predictions_118.pt')
    print("\n--- PREDICTION RESULTS (IEEE 118 Bus) ---\n")

    # GridFM usually outputs a dictionary or tensor
    if isinstance(data, dict):
        # Often keys are 'pred' (prediction) and 'target' (ground truth)
        preds = data.get('pred', data) 
    else:
        preds = data

    print(f"Data Shape: {preds.shape}")
    print("First 5 Predicted Voltages/Angles:")
    print(preds[:5])

except Exception as e:
    print(f"Error reading file: {e}")