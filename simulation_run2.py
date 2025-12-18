import cv2

def split_video(video_path, packet_size=5):
    """
    Read a video, resize each frame to 28x28, and group frames into packets.

    Args:
        video_path (str): Path to the input video file.
        packet_size (int): Number of frames per packet.

    Returns:
        packets (list): List of packets, each packet is a list of `packet_size` frames.
        packet_ids (list): List of integer IDs (0..num_packets-1) corresponding to packets.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [], []

    frames = []

    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to 28x28 (BGR)
        frame_resized = cv2.resize(frame, (28, 28))
        frames.append(frame_resized)

    cap.release()

    # If no frames read, return empty
    if not frames:
        print("Warning: No frames read from video.")
        return [], []

    # Group frames into packets
    packets = []
    packet_ids = []

    frame_count = len(frames)
    usable_frames = frame_count - (frame_count % packet_size)  # drop leftover frames

    for i in range(0, usable_frames, packet_size):
        packet = frames[i:i + packet_size]
        packets.append(packet)
        packet_ids.append(len(packet_ids))  # sequential IDs

    return packets, packet_ids


import os
import random

from .video_utils import split_video   # or from your file import split_video


class HostSender:
    def __init__(self, video_dir="Videos/", packet_size=5):
        """
        Host-side video traffic generator.

        Args:
            video_dir (str): Directory containing input video files.
            packet_size (int): Number of frames per packet.
        """
        self.video_dir = video_dir
        self.packet_size = packet_size
        self.video_list = []
        self.current_round = 0

        # Load videos at initialization
        if os.path.exists(video_dir):
            # filter only common video extensions if needed
            all_files = os.listdir(video_dir)
            self.video_list = sorted(
                f for f in all_files
                if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
            )
            if not self.video_list:
                print(f"Warning: No video files found in {video_dir}")
        else:
            print(f"Error: Directory {video_dir} not found")

    def get_next_video(self):
        """
        Automatically select the next video based on the round number.

        Returns:
            str or None: Full path of the next video, or None if not available.
        """
        if not self.video_list:
            print("No videos found in directory.")
            return None

        index = self.current_round % len(self.video_list)
        video_name = self.video_list[index]
        video_path = os.path.join(self.video_dir, video_name)

        self.current_round += 1
        return video_path

    def generate_packets(self):
        """
        Main function to call each round to get packetized data.

        Returns:
            packets (list): List of packets (each packet is list of frames).
            packet_ids (list): List of packet IDs in shuffled order.
        """
        video_path = self.get_next_video()
        if video_path is None:
            return [], []

        print(f"\n🎬 Round {self.current_round}: Processing video → {video_path}")

        packets, packet_ids = split_video(video_path, self.packet_size)

        if not packets:
            print("No packets created from this video.")
            return [], []

        print(f"Total Packets Created: {len(packets)}")

        # Shuffle packets (random arrival order simulation)
        combined = list(zip(packets, packet_ids))
        random.shuffle(combined)
        packets, packet_ids = zip(*combined)

        return list(packets), list(packet_ids)




import networkx as nx
import matplotlib.pyplot as plt


class NetworkTopology:
    def __init__(self, num_nodes):
        """
        num_nodes = total nodes = 1 host + 1 receiver + S switches
        S = num_nodes - 2

        Switches: S1, S2, ..., S_S
        Host: H
        Receiver: R

        Topology: simple line
            H - S1 - S2 - ... - S_S - R
        """
        assert num_nodes >= 3, "Need at least 3 nodes (H, R, 1 switch)."
        self.G = nx.Graph()

        self.host = "H"
        self.receiver = "R"

        self.num_nodes = num_nodes
        self.num_switches = num_nodes - 2
        self.switches = [f"S{i+1}" for i in range(self.num_switches)]

        # Controller can be placed on any switch
        self.controller_candidates = self.switches.copy()

        # Add nodes
        self.G.add_node(self.host, type="host")
        self.G.add_node(self.receiver, type="receiver")
        for sw in self.switches:
            self.G.add_node(sw, type="switch")

        # Add edges: line topology H - S1 - ... - S_S - R
        prev = self.host
        for sw in self.switches:
            self.G.add_edge(prev, sw, weight=1)
            prev = sw
        self.G.add_edge(prev, self.receiver, weight=1)

        # Positions for drawing (x-axis line)
        self.pos = {}
        x = 0
        self.pos[self.host] = (x, 0)
        x += 1
        for sw in self.switches:
            self.pos[sw] = (x, 0)
            x += 1
        self.pos[self.receiver] = (x, 0)

    def draw_topology(self, path_switches=None, on_switches=None):
        """
        Draw the full topology.

        Args:
            path_switches (list or None): switches on the selected path (green).
            on_switches (list or set or None): switches that are ON from stage 1.
        """
        if path_switches is None:
            path_switches = []
        if on_switches is None:
            on_switches = set()
        else:
            on_switches = set(on_switches)

        path_set = set(path_switches)

        node_colors = []
        for node in self.G.nodes():
            if node == self.host:
                node_colors.append("blue")
            elif node == self.receiver:
                node_colors.append("purple")
            elif node in path_set:
                node_colors.append("green")        # on path
            elif node in on_switches:
                node_colors.append("lightgreen")   # ON but not used in path
            elif node.startswith("S"):
                node_colors.append("red")          # OFF switch
            else:
                node_colors.append("gray")

        # Build edges that are on the path (H -> path_switches -> R)
        edges = self.G.edges()
        edge_colors = []
        path_edges = set()
        if path_switches:
            seq = [self.host] + list(path_switches) + [self.receiver]
            for i in range(len(seq) - 1):
                path_edges.add((seq[i], seq[i+1]))
                path_edges.add((seq[i+1], seq[i]))

        for e in edges:
            if e in path_edges:
                edge_colors.append("green")
            else:
                edge_colors.append("black")

        plt.figure(figsize=(12, 4))
        nx.draw(
            self.G,
            self.pos,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            node_size=600,
            font_size=8,
            width=2,
        )
        plt.title(f"Topology with {self.num_nodes} nodes ({self.num_switches} switches)")
        plt.tight_layout()
        plt.show()


import networkx as nx


class RoutingEngine:
    def __init__(self, topology: NetworkTopology):
        """
        Routing engine operating on a NetworkTopology instance.
        """
        self.topology = topology

    def shortest_path_H_R(self, allowed_switches=None):
        """
        Compute shortest path from Host (H) to Receiver (R).

        Args:
            allowed_switches (list or set or None):
                - If None: use the full topology graph.
                - If list/set: build a subgraph that contains only H, R,
                  and these switches, and compute shortest path in it.

        Returns:
            list of nodes (path H -> ... -> R) or [] if no path exists.
        """
        if allowed_switches is None:
            G = self.topology.G
        else:
            allowed_nodes = set(allowed_switches) | {self.topology.host, self.topology.receiver}
            G = self.topology.G.subgraph(allowed_nodes).copy()

        source = self.topology.host
        target = self.topology.receiver

        try:
            path_nodes = nx.shortest_path(G, source=source, target=target, weight="weight")
        except nx.NetworkXNoPath:
            return []

        return path_nodes

    def path_from_switch_sequence(self, switch_sequence):
        """
        Build a path H -> switch_sequence -> R.

        Args:
            switch_sequence (list): ordered list of switch names (e.g., ["S3", "S4"]).

        Returns:
            list of nodes: [H, S_i, ..., R]
        """
        sequence = [self.topology.host] + list(switch_sequence) + [self.topology.receiver]
        return sequence

    @staticmethod
    def nodes_to_edges(path_nodes):
        """
        Convert a list of nodes [n0, n1, ..., nk] to edges [(n0,n1), ..., (n_{k-1},n_k)].
        """
        edges = []
        for i in range(len(path_nodes) - 1):
            edges.append((path_nodes[i], path_nodes[i + 1]))
        return edges



import networkx as nx


class RoutingEngine:
    def __init__(self, topology):
        """
        topology: NetworkTopology object
        """
        self.topology = topology

    def shortest_path_H_R(self, allowed_switches=None):
        """
        Compute shortest path from Host (H) to Receiver (R).

        Args:
            allowed_switches (list or set or None):
                - If None: use full graph.
                - If not None: restrict the path to these switches + H + R.

        Returns:
            path_nodes (list): list of node names from H to R.
                               [] if no path exists.
        """
        if allowed_switches is None:
            G = self.topology.G
        else:
            allowed_nodes = set(allowed_switches) | {self.topology.host, self.topology.receiver}
            G = self.topology.G.subgraph(allowed_nodes).copy()

        source = self.topology.host
        target = self.topology.receiver

        try:
            # Dijkstra / weighted shortest path (weight="weight")
            path_nodes = nx.shortest_path(G, source=source, target=target, weight="weight")
        except nx.NetworkXNoPath:
            return []

        return path_nodes

    def path_from_switch_sequence(self, switch_sequence):
        """
        Build path H -> switch_sequence -> R.

        Args:
            switch_sequence (list): ordered list of switch names (unique) selected in Stage 2.

        Returns:
            path_nodes (list): [H, S_i1, S_i2, ..., R]
        """
        return [self.topology.host] + list(switch_sequence) + [self.topology.receiver]

    @staticmethod
    def nodes_to_edges(path_nodes):
        """
        Convert a list of nodes [n0, n1, ..., nk] to edge list [(n0,n1), (n1,n2), ...].

        Args:
            path_nodes (list): node sequence.

        Returns:
            list of tuple edges.
        """
        edges = []
        for i in range(len(path_nodes) - 1):
            edges.append((path_nodes[i], path_nodes[i + 1]))
        return edges

import random


class PacketSimulator:
    def __init__(self, topology: NetworkTopology):
        """
        Packet-level simulator using a simple hop-based loss model.
        """
        self.topology = topology

    def simulate(self, packets, path_nodes):
        """
        Simulate sending packets along a given path.

        Args:
            packets (list): list of packets (any objects; length = total_sent).
            path_nodes (list): ordered nodes [H, S_i, ..., R] forming the path.

        Returns:
            qos (float): packet loss ratio = lost / total_sent.
            delivered (int): number of packets delivered.
            lost (int): number of packets lost.
        """
        # Convert nodes to edges to count hops
        path_edges = []
        for i in range(len(path_nodes) - 1):
            path_edges.append((path_nodes[i], path_nodes[i + 1]))

        total_sent = len(packets)
        delivered = 0
        lost = 0

        # === PACKET LOSS MODEL ===
        base_loss = 0.03                    # 3% base loss
        hops = len(path_edges)              # number of links
        path_based_loss = hops * 0.02       # +2% per hop

        loss_prob = base_loss + path_based_loss
        loss_prob = min(loss_prob, 0.30)    # cap at 30%

        # === APPLY LOSS TO EACH PACKET ===
        for _ in packets:
            if random.random() < loss_prob:
                lost += 1
            else:
                delivered += 1

        qos = lost / total_sent if total_sent > 0 else 0.0
        return qos, delivered, lost



import numpy as np
import random

from Optimization.PSO_optimizer import PSOOptimizer
from Optimization.GWO_optimizer import GWOOptimizer
from Optimization.WOA_optimizer import WOAOptimizer
from Optimization.MFO_optimizer import MFOOptimizer
from Optimization.HHO_optimizer import HHOOptimizer
from Optimization.FFO_optimizer import FFOOptimizer
from Optimization.Hybrid_optimizer import Hybrid_Optimizer


# ============================================================
# ADAPTERS: GIVE ALL OPTIMIZERS A COMMON INTERFACE
# ============================================================

class PSOAdapter:
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.core = PSOOptimizer(dim=dim, lower_bound=lower_bound, upper_bound=upper_bound)

    def run(self, objective_function, packets, epochs, pop_size):
        return self.core.run_pso(
            objective_function=objective_function,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )


class GWOAdapter:
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.core = GWOOptimizer(dim=dim, lower_bound=lower_bound, upper_bound=upper_bound)

    def run(self, objective_function, packets, epochs, pop_size):
        return self.core.run_gwo(
            objective_function=objective_function,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )


class WOAAdapter:
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.core = WOAOptimizer(dim=dim, lower_bound=lower_bound, upper_bound=upper_bound)

    def run(self, objective_function, packets, epochs, pop_size):
        return self.core.run_woa(
            objective_function=objective_function,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )


class MFOAdapter:
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.core = MFOOptimizer(dim=dim, lower_bound=lower_bound, upper_bound=upper_bound)

    def run(self, objective_function, packets, epochs, pop_size):
        return self.core.run_mfo(
            objective_function=objective_function,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )


class HHOAdapter:
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.core = HHOOptimizer(dim=dim, lower_bound=lower_bound, upper_bound=upper_bound)

    def run(self, objective_function, packets, epochs, pop_size):
        return self.core.run_hho(
            objective_function=objective_function,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )


class FFOAdapter:
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.core = FFOOptimizer(dim=dim, lower_bound=lower_bound, upper_bound=upper_bound)

    def run(self, objective_function, packets, epochs, pop_size):
        return self.core.run_ffo(
            objective_function=objective_function,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )


class HybridAdapter:
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.core = Hybrid_Optimizer(dim=dim, lower_bound=lower_bound, upper_bound=upper_bound)

    def run(self, objective_function, packets, epochs, pop_size):
        return self.core.run_hybrid(
            objective_function=objective_function,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )


# ============================================================
# TWO-STAGE OPTIMIZATION ENGINE
# ============================================================

class TwoStageOptimizationEngine:
    def __init__(self, topology, router, simulator):
        """
        topology : NetworkTopology (num_switches, switches list, etc.)
        router   : RoutingEngine
        simulator: PacketSimulator
        """
        self.topology = topology
        self.router = router
        self.sim = simulator

    # ---------------- Stage 1: ON/OFF switches ----------------

    def stage1_cost(self, binary_vec, packets):
        """
        binary_vec: numpy array (S,) with {0,1} for each switch.
        Use only ON switches to build subgraph, get H-R shortest path,
        simulate packets, and return QoS (packet loss ratio).
        """
        on_indices = np.where(binary_vec == 1)[0]
        if len(on_indices) == 0:
            return 1.0  # no switches ON → penalty

        on_switches = [self.topology.switches[i] for i in on_indices]

        path_nodes = self.router.shortest_path_H_R(allowed_switches=on_switches)
        if not path_nodes:
            return 1.0  # no path → penalty

        qos, delivered, lost = self.sim.simulate(packets, path_nodes)
        return qos

    def run_stage1_with(self, packets, OptimizerAdapter, epochs=100, pop_size=30):
        """
        Run Stage 1 with a given optimizer adapter class
        (PSOAdapter, GWOAdapter, etc.).

        Returns:
            best_bits: np.array of 0/1
            on_switches: list of switch names
            best_qos: float
        """
        S = self.topology.num_switches

        opt = OptimizerAdapter(dim=S, lower_bound=0, upper_bound=1)

        def objective_wrapper(particle, packets_inner):
            particle = np.array(particle)
            bits = (particle >= 0.5).astype(int)
            return self.stage1_cost(bits, packets_inner)

        best_solution, best_qos = opt.run(
            objective_function=objective_wrapper,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )

        best_bits = (np.array(best_solution) >= 0.5).astype(int)
        on_indices = np.where(best_bits == 1)[0]
        on_switches = [self.topology.switches[i] for i in on_indices]

        return best_bits, on_switches, best_qos

    # ---------------- Stage 2: Path over ON switches ----------------

    def decode_unique_path(self, particle, on_switches):
        """
        Decode a continuous particle to a unique sequence of switch names.
        particle: list/array of length L with values in [0, K-1]
        on_switches: list of ON switch names
        """
        K = len(on_switches)
        if K == 0:
            return []

        raw_indices = []
        for x in particle:
            idx = int(round(x))
            idx = max(0, min(K - 1, idx))
            raw_indices.append(idx)

        seen = set()
        unique_indices = []
        for idx in raw_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        if not unique_indices:
            unique_indices = [0]

        path_switches = [on_switches[i] for i in unique_indices]
        return path_switches

    def stage2_cost(self, particle, on_switches, packets):
        """
        particle: continuous vector, path_length dims
        on_switches: list of ON switch names
        """
        path_switches = self.decode_unique_path(particle, on_switches)
        if not path_switches:
            return 1.0

        path_nodes = self.router.path_from_switch_sequence(path_switches)
        qos, delivered, lost = self.sim.simulate(packets, path_nodes)
        return qos

    def run_stage2_with(self, packets, on_switches, OptimizerAdapter,
                        path_length=None, epochs=100, pop_size=30):
        """
        Run Stage 2 with a given optimizer adapter.

        Args:
            on_switches: list of ON switch names from Stage 1
            path_length: max number of switches in path (<= len(on_switches))

        Returns:
            best_path_switches: list of switch names in path
            best_qos: float
        """
        K = len(on_switches)
        if K == 0:
            return [], 1.0

        if path_length is None:
            path_length = min(K, 5)
        else:
            path_length = min(path_length, K)

        opt = OptimizerAdapter(dim=path_length, lower_bound=0, upper_bound=K - 1)

        def objective_wrapper(particle, packets_inner):
            return self.stage2_cost(particle, on_switches, packets_inner)

        best_solution, best_qos = opt.run(
            objective_function=objective_wrapper,
            packets=packets,
            epochs=epochs,
            pop_size=pop_size,
        )

        best_path_switches = self.decode_unique_path(best_solution, on_switches)
        return best_path_switches, best_qos


# ============================================================
# RUN ALL OPTIMIZERS IN ONE FLOW
# ============================================================

def run_all_optimizations(topo, router, sim, packets,
                          epochs_stage1=100, epochs_stage2=100,
                          pop_size=30, path_length=None):
    """
    Run two-stage optimization for all algorithms:
    PSO, FFO, MFO, WOA, GWO, HHO, Hybrid.

    Returns:
        results: dict with one entry per optimizer:
            {
              'PSO': {
                 'stage1_on_switches': [...],
                 'stage1_qos': float,
                 'stage2_path': [...],
                 'stage2_qos': float
              },
              ...
            }
    """
    engine = TwoStageOptimizationEngine(topo, router, sim)

    optimizers = {
        "PSO": PSOAdapter,
        "FFO": FFOAdapter,
        "MFO": MFOAdapter,
        "WOA": WOAAdapter,
        "GWO": GWOAdapter,
        "HHO": HHOAdapter,
        "Hybrid": HybridAdapter,
    }

    results = {}

    for name, Adapter in optimizers.items():
        print(f"\n===== {name} Optimization =====")

        # Stage 1: ON/OFF switches
        bits1, on_switches, qos1 = engine.run_stage1_with(
            packets=packets,
            OptimizerAdapter=Adapter,
            epochs=epochs_stage1,
            pop_size=pop_size,
        )
        print(f"{name} Stage1: {len(on_switches)} switches ON, QoS = {qos1:.4f}")

        # Stage 2: path over ON switches
        path_switches, qos2 = engine.run_stage2_with(
            packets=packets,
            on_switches=on_switches,
            OptimizerAdapter=Adapter,
            path_length=path_length,
            epochs=epochs_stage2,
            pop_size=pop_size,
        )
        print(f"{name} Stage2: path = {path_switches}, QoS = {qos2:.4f}")

        results[name] = {
            "stage1_on_switches": on_switches,
            "stage1_qos": qos1,
            "stage2_path": path_switches,
            "stage2_qos": qos2,
        }

    return results

