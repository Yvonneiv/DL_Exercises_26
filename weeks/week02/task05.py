import numpy as np
from dl_lib.metrics.functional import apply_threshold

probabilities = np.array([0.1, 0.8, 0.4, 0.7, 0.9])

thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

print("Probabilities:", probabilities)

for threshold in thresholds:
    predictions = apply_threshold(probabilities, threshold)
    print(f"threshold={threshold}: predictions={predictions}")
    