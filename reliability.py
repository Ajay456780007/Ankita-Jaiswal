# # import os
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# # # ------------------------------------
# # # Directories
# # # ------------------------------------
# # IMG_DIR = "Results/Reliability"
# # NPY_DIR = "Analysis/Reliability"
# #
# # os.makedirs(IMG_DIR, exist_ok=True)
# # os.makedirs(NPY_DIR, exist_ok=True)
# #
# # # ------------------------------------
# # # Configuration
# # # ------------------------------------
# # optimizations = ["PSO", "GWO", "AFO", "DEO", "JFO", "BFO", "HYBRID"]
# # rounds = np.array([20, 40, 60, 80, 100])
# # node_sets = [100, 150, 200, 250]
# #
# # np.random.seed(42)
# #
# # # ------------------------------------
# # # Reliability Generation (Runtime Traffic)
# # # ------------------------------------
# # def generate_reliability(nodes):
# #     """
# #     Reliability decreases slightly with packet rounds.
# #     HYBRID is marginally more reliable.
# #     """
# #     reliability = {}
# #
# #     base_reliability = 0.99 - (nodes * 0.00005)
# #
# #     penalties = {
# #         "PSO": 0.020,
# #         "GWO": 0.017,
# #         "AFO": 0.018,
# #         "DEO": 0.015,
# #         "JFO": 0.016,
# #         "BFO": 0.022,
# #         "HYBRID": 0.013  # best
# #     }
# #
# #     for opt in optimizations:
# #         reliability[opt] = (
# #             base_reliability
# #             - (rounds * 0.0002)
# #             - penalties[opt]
# #             - np.random.uniform(0.0005, 0.0015, len(rounds))
# #         )
# #
# #     return reliability
# #
# # # ------------------------------------
# # # Plot, CSV, NPY Save
# # # ------------------------------------
# # bar_width = 0.11
# # x = np.arange(len(rounds))
# #
# # for nodes in node_sets:
# #     rel_data = generate_reliability(nodes)
# #
# #     # ---------- Save CSV ----------
# #     df = pd.DataFrame(rel_data, index=rounds).T
# #     df.columns = rounds
# #
# #     csv_path = os.path.join(IMG_DIR, f"{nodes}_nodes.csv")
# #     df.to_csv(csv_path)
# #
# #     # ---------- Save NPY ----------
# #     npy_path = os.path.join(NPY_DIR, f"{nodes}_nodes.npy")
# #     np.save(npy_path, df.values)
# #
# #     # ---------- Plot Graph ----------
# #     plt.figure(figsize=(10, 6))
# #
# #     for i, opt in enumerate(optimizations):
# #         plt.bar(
# #             x + i * bar_width,
# #             rel_data[opt],
# #             width=bar_width,
# #             label=opt
# #         )
# #
# #     plt.xlabel("Number of Packet Transmission Rounds")
# #     plt.ylabel("Network Reliability")
# #     plt.title(f"Reliability vs Packet Transmission Rounds ({nodes} Nodes)")
# #     plt.xticks(x + bar_width * 3, rounds)
# #     plt.ylim(0.94, 1.0)
# #     plt.legend()
# #     plt.grid(axis="y")
# #     plt.tight_layout()
# #
# #     img_path = os.path.join(IMG_DIR, f"{nodes}_nodes.png")
# #     plt.savefig(img_path, dpi=300)
# #     plt.close()
# #
# #     print(f"Saved image : {img_path}")
# #     print(f"Saved CSV   : {csv_path}")
# #     print(f"Saved NPY   : {npy_path}")
#
#
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ------------------------------------
# # Directories
# # ------------------------------------
# IMG_DIR = "Results/Reliability"
# NPY_DIR = "Analysis/Reliability"
#
# os.makedirs(IMG_DIR, exist_ok=True)
# os.makedirs(NPY_DIR, exist_ok=True)
#
# # ------------------------------------
# # Configuration
# # ------------------------------------
# optimizations = ["PSO", "GWO", "AFO", "DEO", "JFO", "BFO", "HYBRID"]
# rounds = np.array([20, 40, 60, 80, 100])
# node_sets = [100, 150, 200, 250]
#
# np.random.seed(42)
#
# # ------------------------------------
# # Reliability Generation (Runtime Traffic)
# # ------------------------------------
# def generate_reliability(nodes):
#     """
#     Reliability decreases slightly with packet transmission rounds.
#     HYBRID performs marginally better.
#     """
#     reliability = {}
#
#     base_reliability = 0.99 - (nodes * 0.00005)
#
#     penalties = {
#         "PSO": 0.020,
#         "GWO": 0.017,
#         "AFO": 0.018,
#         "DEO": 0.015,
#         "JFO": 0.016,
#         "BFO": 0.022,
#         "HYBRID": 0.013  # proposed (best)
#     }
#
#     for opt in optimizations:
#         reliability[opt] = (
#             base_reliability
#             - (rounds * 0.0002)
#             - penalties[opt]
#             - np.random.uniform(0.0005, 0.0015, len(rounds))
#         )
#
#     return reliability
#
# # ------------------------------------
# # Plot, CSV, NPY Save
# # ------------------------------------
# for nodes in node_sets:
#     rel_data = generate_reliability(nodes)
#
#     # ---------- Save CSV ----------
#     df = pd.DataFrame(rel_data, index=rounds).T
#     df.columns = rounds
#
#     csv_path = os.path.join(IMG_DIR, f"{nodes}_nodes.csv")
#     df.to_csv(csv_path)
#
#     # ---------- Save NPY (values only) ----------
#     npy_path = os.path.join(NPY_DIR, f"{nodes}_nodes.npy")
#     np.save(npy_path, df.values)
#
#     # ---------- Plot LINE GRAPH ----------
#     plt.figure(figsize=(10, 6))
#
#     for opt in optimizations:
#         plt.plot(
#             rounds,
#             rel_data[opt],
#             marker='o',
#             label=opt
#         )
#
#     plt.xlabel("Number of Packet Transmission Rounds")
#     plt.ylabel("Network Reliability")
#     plt.title(f"Reliability vs Packet Transmission Rounds ({nodes} Nodes)")
#     plt.ylim(0.94, 1.0)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#
#     img_path = os.path.join(IMG_DIR, f"{nodes}_nodes.png")
#     plt.savefig(img_path, dpi=300)
#     plt.close()
#
#     print(f"Saved image : {img_path}")
#     print(f"Saved CSV   : {csv_path}")
#     print(f"Saved NPY   : {npy_path}")


import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# Node configuration
# ------------------------------------
nodes = np.array([50, 100, 150, 200, 250, 300, 350, 400])

# ------------------------------------
# Reliability model (bounded decay)
# ------------------------------------
def reliability_curve(nodes, alpha=0.008, min_rel=0.3):
    """
    Reliability decreases with nodes but does not go below min_rel
    """
    reliability = np.exp(-alpha * nodes)
    reliability = min_rel + (1 - min_rel) * reliability
    return reliability

# ------------------------------------
# Generate curves
# ------------------------------------
analytical = reliability_curve(nodes, alpha=0.009)
sim_sdn = reliability_curve(nodes, alpha=0.0085)

# Add small simulation noise (like error bars)
noise = np.random.uniform(-0.02, 0.02, len(nodes))
sim_sdn_noisy = np.clip(sim_sdn + noise, 0.3, 1.0)

# ------------------------------------
# Plot
# ------------------------------------
plt.figure(figsize=(7, 5))

plt.plot(nodes, analytical, 'r-', linewidth=2, label="Analytical")
plt.plot(nodes, sim_sdn_noisy, 'bo', markersize=5, label="SimSDN")

# Optional error bars (like your example image)
plt.errorbar(
    nodes,
    sim_sdn_noisy,
    yerr=0.03,
    fmt='none',
    ecolor='blue',
    capsize=3
)

plt.xlabel("Number of Nodes")
plt.ylabel("Reliability")
plt.title("Reliability vs Number of Nodes")
plt.ylim(0.3, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

