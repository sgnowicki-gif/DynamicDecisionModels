# Run one bare loading pass with construction closures: read the
# four bare input files, assign iteration-0 paths, capture the
# pre-closure original path, run network loading with barred-link
# reroutes, then write only the three construction output files.
# The input directory is used as given; it is never written to,
# and it does not need any particular name. There is no emissions
# post-pass and no agent/trip/link output in bare v2, but the
# internal link MOE arrays are still built and used, exactly as
# before, for travel_time_by_min and the prevailing-time shortest
# paths. Dropping the output writers does not touch that code.
import os
import sys

from .construction import ConstructionManager
from .construction_outputs import (
    write_construction_file, write_excess_files)
from .cpp import ProgramStop
from .loading import Simulator
from .network import World


def run_loading_construction(input_dir, output_dir,
                             construction_file=None, flow_model=2,
                             random_seed=100):
    world = World(input_dir, flow_model=flow_model,
                  random_seed=random_seed)
    if world.flow_model not in (1, 2, 3):
        raise ProgramStop(
            "pysim implements the queue-based flow models 1/2/3; "
            "traffic_flow_model=%s is out of scope" % (world.flow_model,))
    world.read_inputs()
    # Resolve the construction file before any simulation runs.
    # Default: input_dir/construction_projects.csv. An explicit
    # argument may be an absolute path or a name inside input_dir.
    # A missing file fails loud: this specialised version always
    # has a construction file and never silently runs without
    # closures.
    if construction_file is None:
        resolved = os.path.join(input_dir, "construction_projects.csv")
    elif os.path.isabs(construction_file):
        resolved = construction_file
    elif os.path.isfile(os.path.join(input_dir, construction_file)):
        resolved = os.path.join(input_dir, construction_file)
    else:
        resolved = construction_file
    manager = ConstructionManager(resolved, world)
    sim = Simulator(world, construction=manager)
    sim.assign_paths_iteration_zero()
    # Capture the pre-closure original path for every vehicle with
    # a path, before loading and before any closure reroute: for
    # bare pathless agents this is the free-flow shortest path
    # just assigned on the full network. FFTT values are already
    # set by Link.setup_moe at read time.
    for agent_id in world.vehicle_order:
        veh = world.vehicles[agent_id]
        if veh.node_size >= 2 and veh.link_ary is not None:
            veh.closed_original_links = [
                entry.link_no for entry in veh.link_ary[:-1]]
            total = 0.0
            for link_no in veh.closed_original_links:
                total += float(world.links[link_no].fftt)
            veh.closed_fftt = total
    sim.network_loading()
    os.makedirs(output_dir, exist_ok=True)
    summary = write_excess_files(
        world, os.path.join(output_dir, "output_excess_travel_time.csv"),
        os.path.join(output_dir, "output_excess_summary.csv"))
    write_construction_file(
        manager, os.path.join(output_dir, "output_construction_projects.csv"))
    return world, manager, summary


def _flag_value(argv, flag, default=None):
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default


def main(argv):
    if len(argv) < 3:
        print("usage: python -m pysim_construction.run <input_dir> "
              "<output_dir> [--construction <file>] "
              "[--flow-model 2] [--seed 100]")
        return 2
    input_dir = argv[1]
    output_dir = argv[2]
    rest = argv[3:]
    # Unknown flags fail loud instead of being silently ignored:
    # in particular --emissions must not look accepted when the
    # package has no emissions support at all.
    known = {"--construction", "--flow-model", "--seed"}
    i = 0
    while i < len(rest):
        token = rest[i]
        if token.startswith("--") and token not in known:
            print("error: unknown option %s (there is no --emissions "
                  "flag in bare v2)" % token)
            return 2
        if token in known:
            i += 2
        else:
            i += 1
    construction_file = _flag_value(rest, "--construction")
    flow_model = _flag_value(rest, "--flow-model", "2")
    seed = _flag_value(rest, "--seed", "100")
    try:
        flow_model = int(flow_model)
        seed = int(seed)
    except ValueError:
        print("error: --flow-model and --seed must be integers")
        return 2
    run_loading_construction(input_dir, output_dir,
                             construction_file=construction_file,
                             flow_model=flow_model, random_seed=seed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
