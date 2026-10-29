"""pysim_construction: PySim with project-synced construction closures.

A specialised copy of the standalone PySim (v2) loading pass for the
DDM project. It replaces Sebastian's traffic_sim.py for DDM use only:
it is not a general DTALite replacement and it must not be overlaid
onto the v2 tree. See README.md for the construction schema, the
closure semantics, and the limits.
"""
from .api import run_pysim, run_pysim_construction

__all__ = ["run_pysim_construction", "run_pysim"]
