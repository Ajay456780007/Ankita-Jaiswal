# Assume you already created:
from Topology.simulation_run2 import NetworkTopology
from Topology.simulation_run2 import HostSender
from Topology.simulation_run2 import RoutingEngine
from Topology.simulation_run2 import HostSender
from Topology.simulation_run2 import PacketSimulator
from Topology.simulation_run2 import run_all_optimizations
topo = NetworkTopology(num_nodes)
router = RoutingEngine(topo)
sim = PacketSimulator(topo)
sender = HostSender()
packets, ids = sender.generate_packets()

results = run_all_optimizations(
    topo, router, sim, packets,
    epochs_stage1=500,
    epochs_stage2=500,
    pop_size=30,
    path_length=None,   # or a fixed max path length
)

# Example: draw topology for PSO result
pso_on = results["PSO"]["stage1_on_switches"]
pso_path = results["PSO"]["stage2_path"]
topo.draw_topology(path_switches=pso_path, on_switches=pso_on)
