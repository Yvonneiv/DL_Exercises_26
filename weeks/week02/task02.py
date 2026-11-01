import numpy as np #for the arrays and calculate the errors between actual and predicted values
import torch
import torch.nn.functional as F

from dl_lib.losses.functional import mae

y_true = np.array([3, 5, 7, 9], dtype=float)
y_pred = np.array([2, 6, 8, 7], dtype=float)

result = mae(y_true, y_pred)

torch_result = F.l1_loss(
    torch.tensor(y_pred),
    torch.tensor(y_true)
)

print("y_true:", y_true)
print("y_pred:", y_pred)
print(f"MAE: {result:.4f}")
print(f"PyTorch MAE: {torch_result.item():.4f}")