# Example resume run for network_builder v2.
#
# Mirrors example_run.py, but in two stages: build horizon 2,
# then resume the checkpoint to horizon 4. The resumed run must
# give the uninterrupted horizon-4 counts (29 nodes, 34 links,
# 30 transitions computed, 6 merges -- the same numbers a
# single horizon-4 run produces), because resume restores the
# completed layers exactly and only recomputes layers 2 and 3.
# Those counts are profile-count-independent, so they are
# checked here with the full 9-profile k=2 enumeration;
# full node/link/penalty equivalence against an
# uninterrupted run is verified in tests/test_resume.py (to
# horizon 4 with the 3-profile override, and to horizon 2
# with the full 9-profile enumeration).
#
# Run from the extraction directory: python example_resume.py
# Outputs land in example_resume_output/ (nodes.csv, links.csv,
# groups.csv, run_report.json, checkpoint.pkl).
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
out_dir = os.path.join(here, "example_resume_output")
checkpoint = os.path.join(out_dir, "checkpoint.pkl")
print("groups (example mode, k=2, NOT production k=4):", groups)
started = time.time()
network = build_network(
    network_dir, projects, demand, groups, "2026-07-06", horizon=2,
    output_dir=out_dir, verbose=True, coords=group_coords)
print("stage 1 (horizon 2): nodes %d (expected 11), links %d "
      "(expected 12), checkpoint %s"
      % (len(network.nodes), len(network.links), checkpoint))
assert len(network.nodes) == 11, len(network.nodes)
assert len(network.links) == 12, len(network.links)
assert os.path.isfile(checkpoint), "checkpoint.pkl missing after stage 1"
network = build_network(
    network_dir, projects, demand, groups, "2026-07-06", horizon=4,
    output_dir=out_dir, verbose=True, coords=group_coords,
    resume_from=checkpoint)
elapsed = time.time() - started
print("stage 2 (resumed to horizon 4): nodes %d (expected 29), "
      "links %d (expected 34)" % (len(network.nodes), len(network.links)))
print("merges: %d (expected 6)" % network.totals["merges"])
print("transitions computed: %d (expected 30), cache hits: %d, "
      "profile runs: %d"
      % (network.totals["transitions_computed"],
         network.totals["cache_hits"], network.totals["profile_runs"]))
print("elapsed (both stages): %.1fs" % elapsed)
assert len(network.nodes) == 29, len(network.nodes)
assert len(network.links) == 34, len(network.links)
assert network.totals["merges"] == 6, network.totals["merges"]
assert network.totals["transitions_computed"] == 30, \
    network.totals["transitions_computed"]
for node in network.terminal_nodes():
    print("terminal node %d epoch %d unfinished %d reaching %.3f"
          % (node.node_id, node.epoch, node.unfinished_count,
             node.reaching_weight))
print("example resume output in example_resume_output/ (nodes.csv, "
      "links.csv, groups.csv, run_report.json, checkpoint.pkl)")
