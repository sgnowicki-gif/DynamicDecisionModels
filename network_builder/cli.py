"""Command-line entry point for the network builder.

Example (Sioux Falls, real geometry, synthetic projects)::

    python -m network_builder.cli \
        --network-dir sioux_falls_bare_inputs \
        --projects my_projects.csv \
        --demand demand.csv \
        --coordinates input_node_original.csv \
        --start-date 2026-11-02 \
        --horizon 240 \
        --output-dir network_out

Without --demand the network is built with zero demand (the
structure, actions, branches and merging are still exercised;
all penalties are 0). Without --coordinates and without
--groups-file, grouping fails loud: production mode needs
coordinates for k-means. --groups-file accepts a precomputed
groups file instead (tests and examples).
"""

import argparse
import json
import sys
import time

from .demand import load_demand
from .groups import compute_groups, load_coordinates
from .network import build_network, read_bare_network
from .projects import load_projects


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a backwards-induction construction network "
                    "(no solver).")
    parser.add_argument("--network-dir", required=True,
                        help="bare network dir with input_node.csv and "
                             "input_link.csv")
    parser.add_argument("--projects", required=True,
                        help="projects CSV (see README.md)")
    parser.add_argument("--demand", default=None,
                        help="demand CSV; omit for a zero-demand build")
    parser.add_argument("--coordinates", default=None,
                        help="CSV with node_id,x,y (original DTALite "
                             "input_node.csv works)")
    parser.add_argument("--groups-file", default=None,
                        help="precomputed groups CSV instead of k-means")
    parser.add_argument("--k", type=int, default=4,
                        help="k-means groups (production: 4)")
    parser.add_argument("--kmeans-seed", type=int, default=42)
    parser.add_argument("--start-date", required=True,
                        help="YYYY-MM-DD; epoch 0 starts at midnight")
    parser.add_argument("--horizon", type=int, default=240,
                        help="epochs (default 240 = 60 days)")
    parser.add_argument("--flow-model", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0,
                        help="0/1 = sequential, >1 = ProcessPool over "
                             "profiles")
    parser.add_argument("--prune-unfinishable-starts", action="store_true")
    parser.add_argument("--allow-out-of-range-durations",
                        action="store_true")
    parser.add_argument("--save-frontier", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    _nodes, link_set = read_bare_network(args.network_dir)
    projects = load_projects(
        args.projects, link_set,
        allow_out_of_range_durations=args.allow_out_of_range_durations)
    demand = load_demand(args.demand) if args.demand else {}
    node_ids, _links = read_bare_network(args.network_dir)
    coords = None
    if args.coordinates:
        coords = load_coordinates(args.coordinates)
    groups, group_coords = compute_groups(
        node_ids, coordinates=coords, groups_file=args.groups_file,
        k=args.k, seed=args.kmeans_seed)
    started = time.time()
    network = build_network(
        args.network_dir, projects, demand, groups, args.start_date,
        horizon=args.horizon, flow_model=args.flow_model,
        random_seed=args.random_seed, workers=args.workers,
        prune_unfinishable_starts=args.prune_unfinishable_starts,
        output_dir=args.output_dir, save_frontier=args.save_frontier,
        verbose=args.verbose, coords=group_coords)
    elapsed = time.time() - started
    print("nodes: %d, links: %d, epochs: %d, elapsed: %.1fs"
          % (len(network.nodes), len(network.links), args.horizon,
             elapsed))
    print("transitions computed: %d, cache hits: %d, merges: %d, "
          "profile runs: %d"
          % (network.totals["transitions_computed"],
             network.totals["cache_hits"], network.totals["merges"],
             network.totals["profile_runs"]))
    print("outputs in %s" % args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
