import numpy as np

#created my own function for calculating MAE - giving it 2 inputs the real and the predicted values
def mae(y_true, y_pred):
     #converting both inputs into NumPy arrays so I can subtract their values directly
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        
        raise ValueError("Arrays must have the same shape")
    
    absolute_errors = np.abs(y_true - y_pred)
    result = np.mean(absolute_errors)
    return float(result)

#Creating my own function to calculate MSE (measuring the average of the squared differences between actual and predicted values)
def mse(y_true, y_pred):
    
    #Converting both inputs into NumPy arrays so I can perform the mathematical operations on their  values
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    #Check if the 2 arrays have the same shape 
    
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    
    errors = y_true - y_pred
    
    #Squaring makes all errors positive and gives larger errors a much bigger impact on the  result
    squared_errors = np.square(errors)
    
    result = np.mean(squared_errors)
    
    return float(result)

#Creating my own function to calculate RMSE (the square root og the MSE)
def rmse(y_true, y_pred):
    
    #Using my existing mse() function to calculate the average of the squared prediction errors
    mse_result = mse(y_true, y_pred)
    
    #Calculating the square root pf the MSE
    result = np.sqrt(mse_result)
    
    return float(result)