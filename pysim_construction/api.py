"""Public entry point for the pysim_construction folder (bare v2).

Import it with the folder's parent on sys.path::

    from pysim_construction import run_pysim_construction

    result = run_pysim_construction("my_inputs", "my_outputs")

This is version 2 of the construction package and it is NOT
drop-in compatible with v1: the inputs are four bare CSV files
only, there is no emissions argument (passing one raises
TypeError), and only three output files are written. See
README.md for the bare schema and the breaking changes.
"""
import os

from .run import run_loading_construction


def run_pysim_construction(input_dir, output_dir, verbose=False,
                           construction_file=None, flow_model=2,
                           random_seed=100):
    """Run one PySim loading pass with construction closures.

    input_dir holds exactly four bare input files (see README.md):
    input_node.csv, input_link.csv, input_agent.csv and, by
    default, construction_projects.csv. It is read, never written.
    Extra old files in the directory (DTASettings.txt, scenario
    or zone files, and so on) are ignored, not read and not
    rejected. construction_file defaults to
    input_dir/construction_projects.csv; pass an explicit path (or
    a name inside input_dir) to use a different file. A missing
    construction file fails loud instead of silently running
    without closures.

    flow_model must be 1, 2 or 3 (default 2, Newell); anything else
    fails loud. random_seed defaults to 100 but has no effect on
    results in bare v2: no RNG draw is made for information class
    or age, because every agent is hardcoded to
    information_type 1, vehicle_type 1 and age 5.

    Returns a dict with input_dir, output_dir, the three output
    paths under outputs (excess, excess_summary,
    construction_projects only), agents / completed, the excess
    totals under excess, and one row per project-link under
    construction_projects.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(input_dir):
        raise ValueError("input_dir does not exist: %s" % input_dir)
    if verbose:
        print("pysim_construction: %s -> %s (flow_model=%s)"
              % (input_dir, output_dir, flow_model))
    world, manager, summary = run_loading_construction(
        input_dir, output_dir, construction_file=construction_file,
        flow_model=flow_model, random_seed=random_seed)
    agents = len(world.vehicles)
    completed = sum(1 for veh in world.vehicles.values() if veh.complete)
    result = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "outputs": {
            "excess": os.path.join(
                output_dir, "output_excess_travel_time.csv"),
            "excess_summary": os.path.join(
                output_dir, "output_excess_summary.csv"),
            "construction_projects": os.path.join(
                output_dir, "output_construction_projects.csv"),
        },
        "agents": agents,
        "completed": completed,
        "excess": summary,
        "construction_projects": manager.rows_for_output(),
    }
    if verbose:
        print("pysim_construction: done, %d agents, %d completed"
              % (agents, completed))
    return result
