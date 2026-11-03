import numpy as np
from dl_lib.metrics.functional import accuracy, precision, recall, f1_score # type: ignore

y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([1, 0, 0, 1, 1])

print("Balanced dataset:")
print(f"accuracy: {accuracy(y_true, y_pred):.4f}")
print(f"precision: {precision(y_true, y_pred):.4f}")
print(f"recall: {recall(y_true, y_pred):.4f}")
print(f"f1: {f1_score(y_true, y_pred):.4f}")

rng = np.random.default_rng(42)

y_true = np.array([0] * 990 + [1] * 10)
rng.shuffle(y_true)

y_pred = y_true.copy()

positive_indices = np.where(y_true == 1) [0]

missed_indices = rng.choice(positive_indices, size=3, replace=False)

y_pred[missed_indices] = 0

print("\nImbalanced dataset (990 negative, 10 positive):")
print(f"accuracy: {accuracy(y_true, y_pred):.4f}")
print(f"precision: {precision(y_true, y_pred):.4f}")
print(f"recall: {recall(y_true, y_pred):.4f}")
print(f"f1: {f1_score(y_true, y_pred):.4f}")