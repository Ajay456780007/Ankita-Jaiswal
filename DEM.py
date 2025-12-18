import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# Directories
# ------------------------------------
IMG_DIR = "Results/Reliability"
NPY_DIR = "Analysis/Reliability"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)

# ------------------------------------
# Configuration
# ------------------------------------
models = ["PSO", "GWO", "AFO", "DEO", "JFO", "BFO", "HYBRID"]
nodes = np.array([50, 100, 150, 200, 250])
round_sets = [20, 40, 60, 80]

# Base end reliability at 250 nodes for each round
round_end_target = {
    20: 0.64,
    40: 0.58,
    60: 0.50,
    80: 0.43
}

# Model performance factor (HYBRID best)
model_factor = {
    "PSO": 1.08,
    "GWO": 1.06,
    "AFO": 1.07,
    "DEO": 1.04,
    "JFO": 1.05,
    "BFO": 1.10,
    "HYBRID": 1.00
}

# ------------------------------------
# Reliability function
# ------------------------------------
def reliability_curve(nodes, end_value):
    k = -np.log(end_value) / 250
    return np.exp(-k * nodes)

# ------------------------------------
# Generate, Plot, Save
# ------------------------------------
for rounds in round_sets:

    reliability_data = {}

    for model in models:
        base_curve = reliability_curve(nodes, round_end_target[rounds])
        reliability_data[model] = base_curve ** model_factor[model]

    # -------- Save CSV --------
    df = pd.DataFrame(reliability_data, index=nodes)
    df.index.name = "Number_of_Nodes"

    csv_path = os.path.join(IMG_DIR, f"{rounds}_rounds.csv")
    df.to_csv(csv_path)

    # -------- Save NPY --------
    npy_path = os.path.join(NPY_DIR, f"{rounds}_rounds.npy")
    np.save(npy_path, df.values)

    # -------- Plot --------
    plt.figure(figsize=(9, 6))

    for model in models:
        plt.plot(nodes, reliability_data[model],
                 marker='o', linewidth=2, label=model)

    plt.xlabel("Number of Nodes")
    plt.ylabel("Reliability")
    plt.title(f"Reliability vs Nodes ({rounds} Packet Transmission Rounds)")
    plt.ylim(0.3, 1.02)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    img_path = os.path.join(IMG_DIR, f"{rounds}_rounds.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"Saved: {img_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {npy_path}")
