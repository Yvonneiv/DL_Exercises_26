import numpy as np

y_true = np.array([3, 5, 7, 9]) #real values
y_pred = np.array([2, 6, 8, 7]) #model predictions

errors = y_pred - y_true #calculating errors

absolute_errors = np.abs(errors) #calculating the absolute errors 
squared_errors = np.square(errors) #calculating the squared errors
mean_errors = np.mean(errors) #calculating the mean 

#Calculating MAE, MSE and RMSE


mae = np.mean(absolute_errors)
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)

#Printing all results
print ("y_true:", y_true)
print("y_pred:", y_pred)
print("errors:", errors)
print("absolute errors:", absolute_errors)
print("squared errors:", squared_errors)

print()
print(f"mean error: {mean_errors:.4f}")
print(f"MAE (mean absolute error): {mae:.4f}")
print(f"MSE (mean squared error): {mse:.4f}")
print(f"RMSE (root mean squared error): {rmse:.4f}")