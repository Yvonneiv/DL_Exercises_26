import numpy as np
from dl_lib.metrics.functional import confusion_matrix

y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 1])

result = confusion_matrix(y_true, y_pred)

print("True Positives:", result["tp"])
print("True Negatives:", result["tn"])
print("False Positives:", result["fp"])
print("False Negatives:", result["fn"])