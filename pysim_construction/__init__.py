"""pysim_construction: bare PySim with project-synced construction
closures (DDM only), version 2.

A specialised copy of the standalone PySim loading pass for the
DDM project. It replaces Sebastian's traffic_sim.py for DDM use
only: it is not a general DTALite replacement and it must not be
overlaid onto the standalone pysim tree. Version 2 is NOT
drop-in compatible with version 1 of this package (see README.md):
it reads four bare CSV files only, has no emissions support, and
writes only three output files. Extract it into a clean directory
rather than over an old v1 folder, or the old v1 files
(emissions_full.py, outputs.py) will be left behind as orphans.
"""
from .api import run_pysim_construction

__all__ = ["run_pysim_construction"]
