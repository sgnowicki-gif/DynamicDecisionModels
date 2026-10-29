"""Public entry point for the pysim_construction folder.

Import it with the folder's parent on sys.path::

    from pysim_construction import run_pysim_construction

    result = run_pysim_construction("my_inputs", "my_outputs")

run_pysim is an alias for run_pysim_construction inside this package
only, so code written against the v2 name keeps working here.
"""
import os

from .run import run_loading_construction


def run_pysim_construction(input_dir, output_dir, emissions=False,
                           verbose=False, construction_file=None):
    """Run one PySim loading pass with construction closures.

    input_dir is a directory holding DTALite-format input files plus
    construction_projects.csv (see README.md for the schema). It is
    read, never written. construction_file defaults to
    input_dir/construction_projects.csv; pass an explicit path (or
    a name inside input_dir) to use a different file. If no file
    exists at the resolved location this fails loud instead of
    silently running without closures.

    emissions is the v2 boolean toggle: False = off, True = full
    VSP (requires optional_vehicle_emission_rate.csv in input_dir).

    Returns a dict like v2 (input/output dirs, emissions, the four
    original output paths, agents/completed) plus the three
    construction output paths under outputs (excess,
    excess_summary, construction_projects), the excess totals
    under excess, and one row per project-link under
    construction_projects.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    if not os.path.isdir(input_dir):
        raise ValueError("input_dir does not exist: %s" % input_dir)
    if verbose:
        print("pysim_construction: %s -> %s (emissions=%s)"
              % (input_dir, output_dir, bool(emissions)))
    world, manager, summary = run_loading_construction(
        input_dir, output_dir, emissions=bool(emissions),
        construction_file=construction_file)
    agents = len(world.vehicles)
    completed = sum(1 for veh in world.vehicles.values() if veh.complete)
    result = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "emissions": bool(emissions),
        "outputs": {
            "agent": os.path.join(output_dir, "output_agent.csv"),
            "trip": os.path.join(output_dir, "output_trip.csv"),
            "link_tdmoe": os.path.join(output_dir, "output_LinkTDMOE.csv"),
            "link_moe": os.path.join(output_dir, "output_LinkMOE.csv"),
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


# Alias inside this package only.
run_pysim = run_pysim_construction
