import numpy as np
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
labels = ["Joy", "Sadness", "Anger", "Fear", "Love", "Surprise"]

# Mean and Std (row-normalized confusion matrix)
mean_cm = np.array([
    [0.797, 0.092, 0.020, 0.046, 0.038, 0.007],
    [0.071, 0.801, 0.061, 0.030, 0.027, 0.009],
    [0.081, 0.355, 0.484, 0.054, 0.021, 0.005],
    [0.151, 0.102, 0.017, 0.678, 0.049, 0.002],
    [0.121, 0.086, 0.010, 0.064, 0.682, 0.038],
    [0.091, 0.264, 0.009, 0.052, 0.182, 0.403]
])

std_cm = np.array([
    [0.01, 0.008, 0.003, 0.004, 0.002, 0.003],
    [0.005, 0.007, 0.002, 0.003, 0.003, 0.001],
    [0.006, 0.024, 0.016, 0.006, 0.009, 0.003],
    [0.024, 0.011, 0.004, 0.024, 0.01, 0.002],
    [0.015, 0.012, 0.003, 0.015, 0.017, 0.007],
    [0.025, 0.038, 0.007, 0.015, 0.021, 0.015]
])

# =========================
# Plot
# =========================
plt.figure(figsize=(8, 6))
# Use a white-to-light-blue palette; fix colorbar upper bound to the max value in mean_cm

im = plt.imshow(mean_cm, cmap="Blues", vmin=0, vmax=np.max(mean_cm))

plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)

# Annotate cells with mean ± std
for i in range(mean_cm.shape[0]):
    for j in range(mean_cm.shape[1]):
        plt.text(
            j, i,
            f"{mean_cm[i, j]:.2f}\n±{std_cm[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Row-normalized Confusion Matrix (Mean ± Std)")

plt.tight_layout()
plt.show()
