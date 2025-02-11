import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Default": {
        "steps": [1, 3, 4, 5, 10],
        "values": [27.64, 12.69, 17.66, None, 15.34],
    },
    "Default + Time Embedding": {
        "steps": [1, 3, 4, 5, 10],
        "values": [27.64, 42.71, 50.80, 46.71, 32.88],
    },
    "Default + Time Embedding + Gating": {
        "steps": [1, 3, 4, 5, 10],
        "values": [27.64, 2.76, None, None, None],
    },
    "nGPT": {"steps": [1, 3, 4, 5, 10], "values": [32.85, 33.5, 9.27, None, None]},
    "nGPT + TE": {
        "steps": [1, 3, 4, 5, 10],
        "values": [32.85, 31.05, 29.67, None, None],
    },
    "nGPT + TE + Gating": {
        "steps": [1, 3, 4, 5, 10],
        "values": [32.85, 0.0, 5.87, None, None],
    },
    "nGPT + different eigen lrs for each step": {
        "steps": [1, 3, 4, 5, 10],
        "values": [32.85, 28.74, 33.25, 17.27, 17.61],  # Using nGPT value for step 1
    },
}

# Set up the plot
plt.figure(figsize=(12, 8))
plt.grid(True, linestyle="--", alpha=0.7)

# Colors for each line
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# Plot each line
for (name, series), color in zip(data.items(), colors):
    # Convert to numpy arrays and handle None values
    steps = np.array(series["steps"])
    values = np.array(series["values"], dtype=float)

    # Find non-None values
    mask = ~np.isnan(values)

    # Plot the line
    plt.plot(
        steps[mask],
        values[mask],
        "o-",
        label=name,
        color=color,
        linewidth=2,
        markersize=8,
    )

# Customize the plot
plt.xlabel("Number of Steps", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Model Performance Across Steps", fontsize=14, pad=20)

# Set axis ranges
plt.xlim(0, 11)
plt.ylim(
    0,
    max(
        [
            max([v for v in series["values"] if v is not None])
            for series in data.values()
        ]
    )
    + 5,
)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=10)

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Show the plot
plt.show()
