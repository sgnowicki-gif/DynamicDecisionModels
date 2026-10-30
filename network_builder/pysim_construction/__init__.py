"""pysim_construction: stepwise, pause/resume PySim with
construction closures (DDM only), version 3.

A specialised copy of the standalone PySim loading pass for
Eli's Dynamic Decision Models project. It replaces bare v2 for
DDM use only: it is not a general DTALite replacement and it
must not be overlaid onto the standalone pysim tree. Version 3
is NOT drop-in compatible with version 2 (see README.md): the
full-run function is gone, and the run is driven segment by
segment through a session object.

    from pysim_construction import create_session, run_segment

    session = create_session("my_network")
    result = run_segment(session, agents, {(1, 2): 20.0}, 60.0)
    print(result["penalty"])  # segment penalty, vehicle-minutes
    print(result["state"] is session)  # True: state IS the session
"""
from .session import Session, create_session, run_segment

__all__ = ["create_session", "run_segment", "Session"]
