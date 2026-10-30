# Example end-to-end run for network_builder v1.
#
# Runs the production code path on the tiny 4-node example network
# with 2 projects (1 small, 1 medium) for 2 epochs, starting
# Monday 2026-11-02. One deliberate example-mode deviation, flagged
# here and in the README: the example uses k = 2 groups (9 demand
# profiles) instead of production k = 4 (81 profiles), so the run
# finishes in tens of seconds instead of minutes. The node and
# link counts below do not depend on the profile count; only the
# penalty values do.
#
# Hand-computed expected result for horizon 2 (see README):
#   epoch 0: root (all not started).
#   epoch 1: do-nothing node, S-completed, S-in-progress, M-in-progress
#     = 4 nodes.
#   epoch 2: from the do-nothing node 4 nodes; from the S-completed
#     node an M-in-progress node plus a do-nothing node that MERGES
#     into the S-completed node also reached from the S-in-progress
#     node doing nothing (2 merges into that one node: one from
#     each of the two epoch-1 nodes, so that node is counted once,
#     not three times); from the M-in-progress node 1 node.
#     = 6 distinct nodes.
#   Total nodes 1 + 4 + 6 = 11, total links 4 + 8 = 12. The merged
#   S-completed node's reaching weight is 1.5 (0.5 + 0.5 + 0.5),
#   illustrating that reaching weights are merging weights, not
#   probabilities.
# Run from the extraction directory: python example_run.py
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

from network_builder.demand import load_demand
from network_builder.groups import compute_groups, load_coordinates
from network_builder.network import build_network, read_bare_network
from network_builder.projects import load_projects

network_dir = os.path.join(here, "example_network")
_nodes, link_set = read_bare_network(network_dir)
projects = load_projects(os.path.join(here, "example_project.csv"),
                         link_set)
demand = load_demand(os.path.join(here, "example_demand.csv"))
node_ids, _links = read_bare_network(network_dir)
coords = load_coordinates(os.path.join(here, "example_coordinates.csv"))
groups, group_coords = compute_groups(node_ids, coordinates=coords,
                                      k=2, seed=42)
print("groups (example mode, k=2, NOT production k=4):", groups)
started = time.time()
network = build_network(
    network_dir, projects, demand, groups, "2026-11-02", horizon=2,
    output_dir=os.path.join(here, "example_output"), verbose=True,
    coords=group_coords)
elapsed = time.time() - started
print("nodes: %d (expected 11)" % len(network.nodes))
print("links: %d (expected 12)" % len(network.links))
print("merges: %d (expected 2)" % network.totals["merges"])
print("transitions computed: %d, cache hits: %d, profile runs: %d"
      % (network.totals["transitions_computed"],
         network.totals["cache_hits"], network.totals["profile_runs"]))
print("elapsed: %.1fs" % elapsed)
assert len(network.nodes) == 11, len(network.nodes)
assert len(network.links) == 12, len(network.links)
assert network.totals["merges"] == 2, network.totals["merges"]
for node in network.terminal_nodes():
    print("terminal node %d epoch %d unfinished %d reaching %.3f"
          % (node.node_id, node.epoch, node.unfinished_count,
             node.reaching_weight))
print("example output in example_output/ (nodes.csv, links.csv, "
      "groups.csv, run_report.json)")
