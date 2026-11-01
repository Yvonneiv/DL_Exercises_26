import numpy as np
import torch
import torch.nn.functional as F

from dl_lib.losses.functional import mae, mse

#Using the same actual values and predictions as in Task01
#Specifying float to work with floating-point numbers
y_true = np.array([3, 5, 7, 9], dtype=float)
y_pred = np.array([2, 6, 8, 7,], dtype=float)

#Calculating MSE using my own function from functional.py file
mse_result =mse(y_true, y_pred)

print("y_true:", y_true)
print("y_pred:", y_pred)
print(f"MSE: {mse_result:.4f}")

#Creating a copy of the original predictions
#Use .copY() because I do not want to change y_pred itself

y_pred_outlier = y_pred.copy()

#Replacing the last prediction with 100. This creates an extreme prediction error called an outlier
y_pred_outlier[-1] = 100

mae_outlier = mae(y_true, y_pred_outlier)
#Calculating MAE again using the predictions with the outlier.
#I want to see how much the average absolute error increases.
mse_outlier = mse(y_true, y_pred_outlier)

print()
print("y_pred with one extreme outlier:", y_pred_outlier)
print(f"MAE with outlier: {mae_outlier:.4f}")
print(f"MSE with outlier: {mse_outlier:.4f}")

#Converting the original NumPy arrays into PyTorch tensors
true_tensor = torch.tensor(y_true)
pred_tensor = torch.tensor(y_pred)

torch_mse = F.mse_loss(pred_tensor, true_tensor)

print(f"PyTorch MSE: {torch_mse.item():.4f}")


