# Run one DTALite loading pass with construction closures, mirroring
# the v2 flow: settings -> scenario -> input files -> iteration-0
# path assignment -> capture of the pre-closure original path ->
# network loading with barred-link reroutes -> emission post-pass
# (if any) -> the four v2 output files unchanged, plus the three
# construction output files. The input directory is used as given;
# it is never written to, and it does not need any particular name.
import os
import sys

from .construction import ConstructionManager
from .construction_outputs import (
    write_construction_file, write_excess_files)
from .cpp import ProgramStop
from .loading import Simulator
from .network import World
from .outputs import write_agent_files, write_link_moe, write_link_tdmoe


def run_loading_construction(input_dir, output_dir, emissions=False,
                             construction_file=None):
    world = World(input_dir)
    world.emission_mode = "full" if emissions else "off"
    world.emission_output_flag = 2 if emissions else 0
    if world.flow_model not in (1, 2, 3):
        raise ProgramStop(
            "pysim implements the queue-based flow models 1/2/3; "
            "traffic_flow_model=%d (0=BPR, 4=car-following) is out of scope"
            % world.flow_model)
    if world.num_iterations != 0:
        raise ProgramStop(
            "pysim implements a single loading pass; the scenario requests "
            "%d assignment iterations" % (world.num_iterations + 1))
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
    # a path, before loading and before any closure reroute: the
    # preassigned input path, or for pathless agents the free-flow
    # shortest path just assigned on the full network. FFTT values
    # are already set by Link.setup_moe at read time.
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
    if emissions:
        try:
            from .emissions_full import calculate_emissions_full
        except ImportError as exc:
            if getattr(exc, "name", "") in (
                    "pysim_construction.emissions_full",
                    "emissions_full"):
                raise ProgramStop(
                    "emissions=True but emissions_full.py is not "
                    "present; refusing to fall back to off")
            raise
        calculate_emissions_full(world)
        if not getattr(world, "emissions_computed", False):
            raise ProgramStop(
                "emissions_full.py did not set "
                "world.emissions_computed=True (contract breach); "
                "refusing to write agent emissions from unset slots")
    os.makedirs(output_dir, exist_ok=True)
    write_agent_files(world,
                      os.path.join(output_dir, "output_agent.csv"),
                      os.path.join(output_dir, "output_trip.csv"))
    write_link_tdmoe(world, os.path.join(output_dir, "output_LinkTDMOE.csv"),
                     day_no=world.num_iterations + 1)
    write_link_moe(world, os.path.join(output_dir, "output_LinkMOE.csv"))
    summary = write_excess_files(
        world, os.path.join(output_dir, "output_excess_travel_time.csv"),
        os.path.join(output_dir, "output_excess_summary.csv"))
    write_construction_file(
        manager, os.path.join(output_dir, "output_construction_projects.csv"))
    return world, manager, summary


def main(argv):
    if len(argv) < 3:
        print("usage: python -m pysim_construction.run <input_dir> "
              "<output_dir> [--emissions] [--construction <file>]")
        return 2
    input_dir = argv[1]
    output_dir = argv[2]
    emissions = "--emissions" in argv[3:]
    construction_file = None
    if "--construction" in argv[3:]:
        idx = argv.index("--construction")
        if idx + 1 < len(argv):
            construction_file = argv[idx + 1]
    run_loading_construction(input_dir, output_dir, emissions=emissions,
                             construction_file=construction_file)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
