from wsgiref.types import ErrorStream

import numpy as np 
import matplotlib.pyplot as plt

from dl_lib.losses.functional import mae, mse, rmse

#Keeping the three prediction errors fixed throughout the experiment and change only the fourth error
fixed_errors = np.array([1.0, -1.0, 0.5])

#Generate 101 values between 0 and 10 to make curves in my graph smooth
growing_errors = np.linspace(0, 10, 101)

#Create empty lists to store the calculated MAE, MSE, RMSE values for each experiment
mae_values = []
mse_values = []
rmse_values = []

#Going through every value of the growing error
for growing_error in growing_errors:
    
    #Combining the three fixed errors with the fourth error that changes in each iteration
    errors = np.append(fixed_errors, growing_error)
    
    #Use zeros as the actual values. This means the prediction values themselves represent the errors in the experiment
    y_true = np.zeros(4)
    y_pred = errors
    
    mae_result = mae(y_true, y_kakpred)
    mse_result = mse(y_true, y_pred)
    rmse_result = rmse(y_true, y_pred)
    
    mae_values.append(mae_result)
    mse_values.append(mse_result)
    rmse_values.append(rmse_result)
    
    print("Growing errors:", len(growing_errors))
    print("MAE values:", len(mae_values))
    print("MSE values:", len(mse_values))
    print("RMSE values:", len(rmse_values))
    
    for error in [1.0, 5.0, 10.0]:
        
        #Finding the position of the selected error in the array of the growing error values
        
        index = np.argmin(np.abs(growing_errors - error))
        
        print(
            f"Growing error={error:.1f}"
            f"MAE={mae_values[index]:.4f}"
            f"MSE={mse_values[index]:.4f}"
            f"RMSE={rmse_values[index]:.4f}"
        )
        
        #Create a new figure for my graph
        plt.figure(figsize=(9,6))
        
        #Drawing the MAE curve
        plt.plot(growing_errors, mae_values, label="MAE")
        
        #MSE curve
        
        plt.plot(growing_errors, mse_values, label="MSE")
        
        #RMSE curve 
        
        plt.plot(growing_errors, rmse_values, label="RMSE")
        
        plt.title("MAE, MSE, and RMSE as one prediction error grows")
        
        #Label the horizontal axis with the growing error
        plt.xlabel("Growing prediction error")
        
        plt.ylabel("Loss value")
        
        plt.legend()
        
        plt.grid(True)
        
        plt.tight_layout()
        
        #Display the finished graph
        plt.show()
        
        