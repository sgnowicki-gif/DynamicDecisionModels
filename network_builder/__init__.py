"""network_builder: backwards-induction NETWORK construction (no solver).

The builder expands a layered network epoch by epoch. Each node is
(epoch, construction_status_vector); each link is one action from a
node (do nothing, or start one project) with one branch per duration
outcome of a start action. A transition runs every demand profile
through the v3 stepwise PySim session, averages the end states with
the weighted average_state, and shares that averaged state and
penalty across the action's branch nodes.

See README.md for the full specification, including the remaining
definition, the agent-id scheme, merging weights, caching and
checkpointing.
"""

from .average import average_state, medoid_session, weighted_average_state
from .network import Network, build_network

__all__ = [
    "average_state",
    "medoid_session",
    "weighted_average_state",
    "Network",
    "build_network",
]
