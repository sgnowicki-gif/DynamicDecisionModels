"""Layered backwards-induction network construction (no solver).

Node identity. A node is (epoch, construction_status_vector),
where the status vector has one entry per project, in project
file order: ("not_started", None), ("in_progress", remaining)
or ("completed", None). Two histories that reach the same
(epoch, status) are the same node, and their traffic states are
merged (see merge_traffic).

REMAINING DEFINITION -- read this before touching any of the
code below, because every closed set, decrement and completion
uses it, and the tests pin it down. remaining counts the epochs
still to run INCLUDING the next epoch. Concretely:
- A project started at epoch k with duration d epochs is closed
  during epochs k .. k+d-1 and completed at epoch k+d.
- The branch node created by the start action (at epoch k+1)
  therefore carries remaining = d - 1 if d > 1, and completed
  if d == 1. That is the status AFTER the first closed epoch.
- A node at epoch k with an in-progress project of remaining r
  has that link closed during epoch k (the transition out of
  the node), and the next node carries remaining r - 1, or
  completed if r - 1 == 0.
- In v3 terms the closed link's minutes_left is
  remaining * 360 for an in-progress project at the node, and
  duration * 360 for a newly started project. Minutes_left is
  always a multiple of 360, never fractional, because remaining
  and duration are integer epochs.

Actions. Exactly one new start per epoch, or do nothing. Only
not-started projects can start. Multiple projects can be in
progress at once, but only one TYPE at once: if any project is
in progress, a new start must be of that same type; if none is
in progress, any type may start. This invariant is maintained
by construction (a start is only allowed when it matches the
in-progress type, or when nothing is in progress), and
available_actions asserts it rather than assuming it.

Transition. For one node and one action, run all demand
profiles (3 per origin group, 81 for 4 groups) from deep copies
of the node's traffic state, each with that profile's agents
and the action's closed set, step_minutes = 360, through v3
run_segment. Weighted penalty = sum(profile_prob *
profile_penalty). Weighted end state = weighted_average_state
(profile end sessions, profile probs). The transition is
computed ONCE per action; a start action then creates one
branch node per duration outcome, all sharing that averaged
traffic state and penalty, differing only in construction
remaining and branch probability = P(duration outcome). A
do-nothing action creates one next node. Branch sharing is
mandatory: the branch nodes initially reference the same
session object, and no later step mutates a stored session in
place (transitions always deepcopy before running, merging
replaces the reference).

History merging. When a transition leads to an existing
(epoch, status) node, the traffic states merge:
new_traffic = weighted_average_state([existing, incoming],
[existing_reaching_weight, incoming_path_probability]),
normalised inside weighted_average_state, where
incoming_path_probability = source_reaching_weight *
branch_probability. Demand profile probabilities are NOT
included again: demand was already averaged out inside each
transition. reaching_weight starts at 1.0 for the root and
accumulates: a node's reaching_weight is the sum of the
incoming path probabilities merged into it. It is a merging
weight, NOT a probability -- it can exceed 1.0 when several
decision histories merge, because decisions are not random --
and it is NOT used in any Bellman equation (that is later
work). NOTE ON THE SPEC WORDING: the brief says the existing
weight "starts at 1.0 for root's children". Read literally,
that would give a root start child reaching 1.0 regardless of
its duration probability, which contradicts the product rule
in the same paragraph. This implementation uses the product
rule throughout (root = 1.0, child = parent * branch prob),
so a root do-nothing child has reaching 1.0 and a root start
child has reaching = its duration probability. The merge
function itself takes explicit weights, so the convention can
be changed in one place if Eli rules the other way.

Merging needs disjoint ids (see ids.py). Two histories' stored
sessions both carry compact sequential ids starting near 1,
for different trips, so averaging them directly would block
donor adds on every overlapping id. merge_traffic therefore
offsets the incoming session's ids above the existing
session's max, averages, then renumbers the merged session to
a compact 1..N range (base = 1) and prunes completed vehicles.
That renumbering is safe for the same reason the transition
renumbering is: the builder assigns every future id from
next_agent_id_base, and the retained ids after renumbering are
exactly the merged session's current vehicles, waiting
vehicles and pruned completed ids.

Caching. Cache key = (traffic_state_group_id, closed link
frozenset, epoch day of week, epoch starting hour). Minutes
are deliberately NOT in the key: every closed link's
minutes_left is a multiple of 360 and the step is fixed at
360, so any closed link is barred for the whole segment
regardless of its remaining count, and two transitions that
differ only in remaining counts (the usual case for branch
nodes of one start action doing nothing next) have identical
segment penalties and end states. If the step ever stopped
being 360, this cache would be wrong and must be revisited.
traffic_state_group_id is shared by branch nodes created from
one transition; a merged node gets a fresh group id. A cache
hit returns a deepcopy of the cached end session, never the
cached object itself, so callers cannot mutate the cache.

Terminal nodes. Epoch == T has no outgoing actions. Each
terminal node stores unfinished_count = number of projects
not completed at T. No terminal penalty number is invented;
the penalty is applied at analysis time and is documented as
large enough to force completion.

Checkpointing and resume (v2). Expansion is layered: all nodes at
epoch k are expanded before any node at k+1, per-layer stats
are logged, and nodes.csv/links.csv are rewritten after each
layer (cumulative). After every completed layer the whole
resumable state is also written to a single file,
output_dir/checkpoint.pkl (overwritten each layer; see
checkpoint.py for the format, the input fingerprint and what
is deliberately NOT stored -- the transition cache).
build_network(resume_from=...) loads that file, validates its
fingerprint against the current inputs, and continues the
layer loop from the checkpoint epoch + 1 without recomputing
completed layers. The v1 --save-frontier per-layer frontier
pickles (frontier_epoch_*.pkl) are gone: that flag is now a
no-op alias that prints a note saying checkpoint.pkl
supersedes it, and stale frontier_epoch_*.pkl files in the
output dir are deleted at the start of a run.
"""

import copy
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from .checkpoint import (build_state, fingerprint_from_objects,
                         load_checkpoint, prepare_output_dir,
                         projects_summary, save_checkpoint)

from .demand import (count_epoch_agents, enumerate_profiles,
                     generate_epoch_agents, parse_start_date)
from .groups import compute_groups, save_groups_csv
from .ids import (next_agent_id_base, offset_session, prune_session,
                  renumber_session)
from .projects import load_projects
from .average import weighted_average_state

try:
    from .pysim_construction import create_session, run_segment
except ImportError:
    from pysim_construction import create_session, run_segment

EPOCH_MINUTES = 360


def read_bare_network(network_dir):
    """Read node ids and the link set from a bare network dir."""
    node_path = os.path.join(network_dir, "input_node.csv")
    link_path = os.path.join(network_dir, "input_link.csv")
    if not os.path.isfile(node_path):
        raise ValueError("network dir %s has no input_node.csv"
                         % network_dir)
    if not os.path.isfile(link_path):
        raise ValueError("network dir %s has no input_link.csv"
                         % network_dir)
    nodes = []
    with open(node_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "node_id" not in [
                h.strip() for h in reader.fieldnames]:
            raise ValueError("input_node.csv must have a node_id column")
        for raw in reader:
            row = {k.strip(): v for k, v in raw.items() if k is not None}
            if all(v in (None, "") for v in row.values()):
                continue
            nodes.append(int(row["node_id"]))
    links = set()
    with open(link_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = [h.strip() for h in reader.fieldnames] \
            if reader.fieldnames else []
        for col in ("from_node_id", "to_node_id"):
            if col not in fields:
                raise ValueError("input_link.csv must have %s column" % col)
        for raw in reader:
            row = {k.strip(): v for k, v in raw.items() if k is not None}
            if all(v in (None, "") for v in row.values()):
                continue
            links.add((int(row["from_node_id"]), int(row["to_node_id"])))
    if not nodes:
        raise ValueError("input_node.csv has no nodes")
    if not links:
        raise ValueError("input_link.csv has no links")
    return nodes, links


def initial_status(n_projects):
    return tuple(("not_started", None) for _ in range(n_projects))


def status_to_json(status, projects):
    return [[projects[i].project_id, state, remaining]
            for i, (state, remaining) in enumerate(status)]


def in_progress_types(status, projects):
    types = set()
    for i, (state, _remaining) in enumerate(status):
        if state == "in_progress":
            types.add(projects[i].project_type)
    if len(types) > 1:
        raise ValueError("construction status has more than one "
                         "in-progress type: %s" % (sorted(types),))
    return types


def available_actions(status, projects, epoch, horizon,
                      prune_unfinishable_starts=False):
    """Actions from a node: ('do_nothing', None) plus starts.

    Returns a list of (action_name, project_or_None). project is
    a Project for a start action. The type constraint and the
    one-start-per-epoch rule are enforced here, not by callers.
    """
    actions = [("do_nothing", None)]
    active = in_progress_types(status, projects)
    for i, project in enumerate(projects):
        if status[i][0] != "not_started":
            continue
        if active and project.project_type not in active:
            continue
        if prune_unfinishable_starts and \
                project.min_duration > horizon - epoch:
            continue
        actions.append(("start", project))
    return actions


def closed_set_for(status, projects, action_name, action_project,
                   branch_duration):
    """Closed links dict for a transition, in v3 minutes_left form.

    In-progress projects at the node contribute remaining*360;
    a newly started project contributes duration*360. See the
    remaining definition in this module's docstring.
    """
    closed = {}
    for i, project in enumerate(projects):
        state, remaining = status[i]
        if state == "in_progress":
            if remaining is None or remaining < 1:
                raise ValueError("in-progress project %s has bad "
                                 "remaining %r" % (project.project_id,
                                                   remaining))
            closed[project.link] = remaining * EPOCH_MINUTES
    if action_name == "start":
        if action_project is None or branch_duration is None:
            raise ValueError("start action needs a project and a "
                             "branch duration")
        closed[action_project.link] = branch_duration * EPOCH_MINUTES
    return closed


def apply_status(status, projects, action_name, action_project,
                 branch_duration):
    """Status at the next epoch after one transition epoch.

    Existing in-progress remaining decrements by 1 (to completed
    at 0); a newly started project ends the epoch at
    duration - 1 (completed if duration == 1).
    """
    out = []
    for i, project in enumerate(projects):
        state, remaining = status[i]
        if action_name == "start" and project is action_project:
            if branch_duration is None or branch_duration < 1:
                raise ValueError("bad branch duration %r"
                                 % (branch_duration,))
            if branch_duration == 1:
                out.append(("completed", None))
            else:
                out.append(("in_progress", branch_duration - 1))
        elif state == "in_progress":
            if remaining is None or remaining < 1:
                raise ValueError("in-progress project %s has bad "
                                 "remaining %r" % (project.project_id,
                                                   remaining))
            if remaining == 1:
                out.append(("completed", None))
            else:
                out.append(("in_progress", remaining - 1))
        else:
            out.append((state, remaining))
    return tuple(out)


def _profile_worker(job):
    session, agents, closed_links, step_minutes = job
    result = run_segment(session, agents, closed_links, step_minutes)
    return result["penalty"], result["state"]


def compute_transition(source_session, closed_links, demand, origin_group,
                       sorted_groups, epoch, start_dow, workers=0,
                       profiles=None, stats=None):
    """Run one transition: all profiles, weighted average, cleanup.

    Returns (weighted_penalty, stored_session) where the stored
    session is renumbered (ids >= the source base rewritten to
    compact sequential ids) and pruned (completed vehicles to an
    id-only set). stats, when given, is a dict updated with
    profile_runs and seconds. profiles may override the default
    enumeration for tests (a list of (levels, probability)).
    """
    if profiles is None:
        profiles = enumerate_profiles(len(sorted_groups))
    total_prob = sum(prob for _levels, prob in profiles)
    if abs(total_prob - 1.0) > 1e-9:
        raise ValueError("profile probabilities sum to %r, not 1"
                         % total_prob)
    base = next_agent_id_base(source_session)
    counts = [count_epoch_agents(demand, origin_group, sorted_groups,
                                 levels, epoch, start_dow)
              for levels, _prob in profiles]
    stride = max(counts) + 1000 if counts else 1000
    started = time.time()
    end_sessions = []
    penalties = []
    if workers and workers > 1:
        jobs = []
        for index, ((levels, _prob), _count) in enumerate(
                zip(profiles, counts)):
            agents = generate_epoch_agents(
                demand, origin_group, sorted_groups, levels, epoch,
                start_dow, base, stride, index)
            jobs.append((copy.deepcopy(source_session), agents,
                         dict(closed_links), float(EPOCH_MINUTES)))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_profile_worker, jobs))
        for penalty, end_session in results:
            penalties.append(penalty)
            end_sessions.append(end_session)
    else:
        for index, ((levels, _prob), _count) in enumerate(
                zip(profiles, counts)):
            agents = generate_epoch_agents(
                demand, origin_group, sorted_groups, levels, epoch,
                start_dow, base, stride, index)
            session_copy = copy.deepcopy(source_session)
            result = run_segment(session_copy, agents, dict(closed_links),
                                 float(EPOCH_MINUTES))
            if sorted(result.keys()) != ["penalty", "state"]:
                raise ValueError("run_segment returned unexpected keys "
                                 "%s" % (sorted(result.keys()),))
            penalties.append(result["penalty"])
            end_sessions.append(result["state"])
    weighted_penalty = sum(prob * penalty
                           for (_levels, prob), penalty
                           in zip(profiles, penalties))
    averaged = weighted_average_state(end_sessions,
                                      [prob for _levels, prob in profiles])
    renumber_session(averaged, base)
    prune_session(averaged)
    if stats is not None:
        stats["profile_runs"] = stats.get("profile_runs", 0) + len(profiles)
        stats["transition_seconds"] = stats.get("transition_seconds", 0.0) \
            + (time.time() - started)
    return weighted_penalty, averaged


def merge_traffic(existing_session, incoming_session,
                  existing_weight, incoming_path_prob):
    """Merge an incoming traffic state into an existing node state.

    Offsets the incoming ids above the existing max (disjoint
    ranges, so average_state donor adds are not blocked by
    overlapping sequential ids from different histories),
    averages with the given merging weights, renumbers the
    result to a compact 1..N range and prunes completed
    vehicles. Returns the new session. Neither input session is
    mutated: the incoming session is deepcopied before
    offsetting, and weighted_average_state never mutates its
    inputs.
    """
    if existing_weight < 0 or incoming_path_prob < 0:
        raise ValueError("merging weights must be non-negative")
    if existing_weight + incoming_path_prob <= 0:
        raise ValueError("merging weights sum to zero")
    incoming_copy = copy.deepcopy(incoming_session)
    offset = next_agent_id_base(existing_session) + 1000
    offset_session(incoming_copy, offset)
    merged = weighted_average_state(
        [existing_session, incoming_copy],
        [existing_weight, incoming_path_prob])
    renumber_session(merged, 1)
    prune_session(merged)
    return merged


class _Node:
    def __init__(self, node_id, epoch, status, traffic, reaching_weight,
                 group_id, unfinished_count):
        self.node_id = node_id
        self.epoch = epoch
        self.status = status
        self.traffic = traffic
        self.reaching_weight = reaching_weight
        self.group_id = group_id
        self.unfinished_count = unfinished_count


class _Link:
    def __init__(self, link_id, from_node, to_node, action,
                 branch_probability, link_weight, closed_set,
                 profile_count):
        self.link_id = link_id
        self.from_node = from_node
        self.to_node = to_node
        self.action = action
        self.branch_probability = branch_probability
        self.link_weight = link_weight
        self.closed_set = closed_set
        self.profile_count = profile_count


class Network:
    """In-memory backwards-induction network (nodes, links, stats)."""

    def __init__(self, nodes, links, projects, horizon, start_date,
                 layer_stats, totals):
        self.nodes = nodes
        self.links = links
        self.projects = projects
        self.horizon = horizon
        self.start_date = start_date
        self.layer_stats = layer_stats
        self.totals = totals

    def nodes_at(self, epoch):
        return [node for node in self.nodes.values()
                if node.epoch == epoch]

    def outgoing(self, node_id):
        return [link for link in self.links if link.from_node == node_id]

    def terminal_nodes(self):
        return self.nodes_at(self.horizon)

    def write_outputs(self, output_dir, groups, coords=None):
        os.makedirs(output_dir, exist_ok=True)
        nodes_path = os.path.join(output_dir, "nodes.csv")
        with open(nodes_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["node_id", "epoch", "clock_min",
                             "construction_status", "reaching_weight",
                             "unfinished_count"])
            for node in sorted(self.nodes.values(), key=lambda n: n.node_id):
                writer.writerow([
                    node.node_id, node.epoch, node.epoch * EPOCH_MINUTES,
                    json.dumps(status_to_json(node.status, self.projects)),
                    node.reaching_weight,
                    node.unfinished_count
                    if node.unfinished_count is not None else ""])
        links_path = os.path.join(output_dir, "links.csv")
        with open(links_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["link_id", "from_node", "to_node", "action",
                             "branch_probability", "link_weight",
                             "closed_set", "profile_count"])
            for link in self.links:
                writer.writerow([
                    link.link_id, link.from_node, link.to_node,
                    link.action, link.branch_probability,
                    link.link_weight,
                    json.dumps([[frm, to, minutes]
                                for (frm, to), minutes
                                in sorted(link.closed_set.items())]),
                    link.profile_count])
        save_groups_csv(os.path.join(output_dir, "groups.csv"),
                        groups, coords)
        report_path = os.path.join(output_dir, "run_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump({"nodes": len(self.nodes), "links": len(self.links),
                       "horizon": self.horizon,
                       "start_date": self.start_date,
                       "layer_stats": self.layer_stats,
                       "totals": self.totals},
                      fh, indent=2)
        return {"nodes_csv": nodes_path, "links_csv": links_path,
                "groups_csv": os.path.join(output_dir, "groups.csv"),
                "run_report": report_path}


def build_network(network_dir, projects, demand, groups, start_date,
                  horizon=240, flow_model=2, random_seed=100, workers=0,
                  prune_unfinishable_starts=False, output_dir=None,
                  save_frontier=False, profiles=None, verbose=False,
                  coords=None, resume_from=None, fingerprint=None,
                  k=None, kmeans_seed=None):
    """Expand the network layer by layer; returns a Network.

    projects: list of Project (see projects.py). demand: dict
    from demand.load_demand (or an equivalent dict). groups:
    {node_id: group label} for the network nodes. start_date:
    YYYY-MM-DD string; epoch 0 starts at midnight on that date.
    profiles: optional override of the enumerated demand
    profiles (tests only; production always uses the full
    enumeration). When output_dir is given, nodes.csv, links.csv,
    groups.csv and run_report.json are rewritten after every
    layer, and checkpoint.pkl is overwritten after every
    completed layer (checkpointing is always on when output_dir
    is given; there is no separate flag). save_frontier is kept
    only as a backwards-compatible no-op: it prints a note that
    checkpoint.pkl supersedes the v1 frontier pickles and
    otherwise does nothing. resume_from: path to a checkpoint.pkl
    from an earlier run; the run continues from the checkpoint
    epoch + 1 after validating the checkpoint's fingerprint
    against the current inputs (ValueError on any mismatch, on
    a missing/corrupt checkpoint, or when the checkpoint epoch
    is already >= horizon). fingerprint: a caller-computed
    fingerprint string (the CLI passes one built from input
    file bytes); when None it is computed here from the loaded
    objects. k / kmeans_seed are recorded in the fingerprint
    when known (the CLI passes them).
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) \
            or horizon < 1:
        raise ValueError("horizon must be a positive int, got %r"
                         % (horizon,))
    _day, start_dow = parse_start_date(start_date)
    node_ids, link_set = read_bare_network(network_dir)
    for project in projects:
        if project.link not in link_set:
            raise ValueError("project %s link %s not in input_link.csv"
                             % (project.project_id, project.link))
    sorted_groups = sorted(set(groups.values()))
    coords_for_output = coords
    if output_dir is not None:
        # Run-start housekeeping (see checkpoint.py
        # prepare_output_dir): stale v1 frontier pickles and
        # orphan checkpoint tmp files are deleted, and on a
        # FRESH run a pre-existing checkpoint.pkl is renamed
        # to checkpoint.stale.pkl so a run killed in layer 0
        # leaves no stale checkpoint behind.
        for action in prepare_output_dir(output_dir, resume_from):
            if verbose:
                print(action)
    if save_frontier:
        print("--save-frontier is superseded in v2: checkpoint.pkl "
              "(one file, overwritten after every completed layer) "
              "replaces the v1 frontier_epoch_*.pkl files; the flag "
              "is accepted and ignored.")
    if fingerprint is None:
        fingerprint = fingerprint_from_objects(
            projects, demand, groups, coords, network_dir, start_date,
            horizon, flow_model, random_seed,
            prune_unfinishable_starts, k, kmeans_seed,
            profiles=profiles)
    nodes = {}
    nodes_by_key = {}
    links = []
    next_node_id = 0
    next_link_id = 0
    next_group_id = 1
    layer_stats = []
    totals = {"transitions_computed": 0, "cache_hits": 0,
              "profile_runs": 0, "transition_seconds": 0.0,
              "merges": 0, "transition_log": []}
    start_epoch = 0

    def unfinished(status):
        return sum(1 for state, _r in status if state != "completed")

    def add_node(epoch, status, traffic, reaching, group_id):
        nonlocal next_node_id
        node = _Node(next_node_id, epoch, status, traffic, reaching,
                     group_id,
                     unfinished(status) if epoch == horizon else None)
        nodes[next_node_id] = node
        nodes_by_key[(epoch, status)] = node
        next_node_id += 1
        return node

    if resume_from is not None:
        state = load_checkpoint(resume_from)
        if state["fingerprint"] != fingerprint:
            raise ValueError(
                "checkpoint fingerprint mismatch: the checkpoint "
                "was built from different inputs/settings than the "
                "current run (projects, demand, groups/coordinates, "
                "network, start_date, flow_model, random_seed, "
                "prune flag, k, kmeans seed, or profile override "
                "changed). Refusing to resume; rerun from scratch "
                "or resume with the original inputs.")
        if state["start_date"] != start_date:
            raise ValueError(
                "checkpoint start_date %s does not match current "
                "start_date %s" % (state["start_date"], start_date))
        if state["projects_summary"] != projects_summary(projects):
            raise ValueError(
                "checkpoint projects do not match the current "
                "projects file")
        if state["epoch"] >= horizon:
            raise ValueError(
                "checkpoint is already at epoch %d, which is >= "
                "horizon %d: nothing left to resume"
                % (state["epoch"], horizon))
        if state["epoch"] < 1:
            raise ValueError(
                "checkpoint epoch %r is not a completed layer "
                "(checkpoints are written after a layer completes)"
                % (state["epoch"],))
        node_ids_seen = set()
        for record in state["nodes"]:
            for key in ("node_id", "epoch", "status",
                        "reaching_weight", "group_id",
                        "unfinished_count", "traffic"):
                if key not in record:
                    raise ValueError(
                        "checkpoint node record is missing %r" % key)
            if record["epoch"] > state["epoch"]:
                raise ValueError(
                    "checkpoint node %r is at epoch %d, past the "
                    "checkpoint epoch %d"
                    % (record["node_id"], record["epoch"],
                       state["epoch"]))
            # unfinished_count is only meaningful on terminal
            # nodes. A restored node is terminal only if its
            # epoch is the resumed horizon, which cannot happen
            # here (every restored epoch is <= the checkpoint
            # epoch, and the checkpoint epoch is < horizon, as
            # checked above), so restored nodes carry None --
            # exactly as in an uninterrupted run to this horizon.
            # The stored value is kept in the checkpoint for
            # provenance only.
            node = _Node(record["node_id"], record["epoch"],
                         record["status"], record["traffic"],
                         record["reaching_weight"], record["group_id"],
                         record["unfinished_count"]
                         if record["epoch"] == horizon else None)
            if node.node_id in nodes:
                raise ValueError("checkpoint has duplicate node %r"
                                 % (node.node_id,))
            nodes[node.node_id] = node
            nodes_by_key[(node.epoch, node.status)] = node
            node_ids_seen.add(node.node_id)
        frontier = [n for n in nodes.values()
                    if n.epoch == state["epoch"]]
        if not frontier:
            raise ValueError(
                "checkpoint has no nodes at its epoch %d"
                % state["epoch"])
        for record in state["links"]:
            for key in ("link_id", "from_node", "to_node", "action",
                        "branch_probability", "link_weight",
                        "closed_set", "profile_count"):
                if key not in record:
                    raise ValueError(
                        "checkpoint link record is missing %r" % key)
            if record["from_node"] not in node_ids_seen \
                    or record["to_node"] not in node_ids_seen:
                raise ValueError(
                    "checkpoint link %r references an unknown node"
                    % (record["link_id"],))
            links.append(_Link(
                record["link_id"], record["from_node"],
                record["to_node"], record["action"],
                record["branch_probability"], record["link_weight"],
                dict(record["closed_set"]), record["profile_count"]))
        links.sort(key=lambda link: link.link_id)
        next_node_id = state["next_node_id"]
        next_link_id = state["next_link_id"]
        next_group_id = state["next_group_id"]
        if nodes and max(nodes) >= next_node_id:
            raise ValueError("checkpoint next_node_id %d does not "
                             "exceed the restored node ids"
                             % next_node_id)
        if links and max(l.link_id for l in links) >= next_link_id:
            raise ValueError("checkpoint next_link_id %d does not "
                             "exceed the restored link ids"
                             % next_link_id)
        if nodes and max(n.group_id for n in nodes.values()) \
                >= next_group_id:
            raise ValueError("checkpoint next_group_id %d does not "
                             "exceed the restored group ids"
                             % next_group_id)
        totals = state["totals"]
        layer_stats = state["layer_stats"]
        if [s["epoch"] for s in layer_stats] != list(
                range(state["epoch"])):
            raise ValueError(
                "checkpoint layer_stats do not cover epochs 0..%d"
                % (state["epoch"] - 1,))
        start_epoch = state["epoch"]
        if verbose:
            print("resumed from %s at epoch %d: %d nodes, %d links, "
                  "%d transitions computed so far"
                  % (resume_from, state["epoch"], len(nodes),
                     len(links), totals["transitions_computed"]))
    else:
        root_session = create_session(network_dir, flow_model=flow_model,
                                      random_seed=random_seed)
        root_status = initial_status(len(projects))
        add_node(0, root_status, root_session, 1.0, 0)
    cache = {}
    profile_list = profiles
    if profile_list is None:
        profile_list = enumerate_profiles(len(sorted_groups))
    for epoch in range(start_epoch, horizon):
        layer_started = time.time()
        layer_nodes = [node for node in sorted(nodes.values(),
                                               key=lambda n: n.node_id)
                       if node.epoch == epoch]
        layer_actions = 0
        layer_computed = 0
        layer_hits = 0
        layer_merges = 0
        dow = (start_dow + epoch // 4) % 7
        epoch_hour = (epoch * 6) % 24
        for node in layer_nodes:
            for action_name, project in available_actions(
                    node.status, projects, epoch, horizon,
                    prune_unfinishable_starts):
                layer_actions += 1
                branches = []
                if action_name == "do_nothing":
                    branches = [(1.0, None, apply_status(
                        node.status, projects, "do_nothing", None, None))]
                else:
                    for duration, prob in project.outcomes:
                        branches.append((prob, duration, apply_status(
                            node.status, projects, "start", project,
                            duration)))
                closed = closed_set_for(node.status, projects,
                                        action_name, project,
                                        branches[0][1] if action_name == "start"
                                        else None)
                # The closed set for a start action does not depend
                # on the branch duration in link membership (only in
                # minutes_left, which the cache deliberately ignores
                # -- see the module docstring), so one transition
                # serves all branches. For do-nothing the branch
                # duration is None and closed has no new link.
                cache_key = (node.group_id, frozenset(closed.keys()),
                             dow, epoch_hour)
                action_label = "do_nothing" if action_name == "do_nothing" \
                    else project.project_id
                if cache_key in cache:
                    penalty, end_session, group_id = cache[cache_key]
                    end_session = copy.deepcopy(end_session)
                    layer_hits += 1
                    totals["cache_hits"] += 1
                else:
                    stats = {}
                    penalty, end_session = compute_transition(
                        node.traffic, closed, demand, groups,
                        sorted_groups, epoch, start_dow, workers=workers,
                        profiles=profile_list, stats=stats)
                    group_id = next_group_id
                    next_group_id += 1
                    cache[cache_key] = (penalty, end_session, group_id)
                    layer_computed += 1
                    totals["transitions_computed"] += 1
                    totals["profile_runs"] += stats.get("profile_runs", 0)
                    totals["transition_seconds"] += stats.get(
                        "transition_seconds", 0.0)
                    totals["transition_log"].append({
                        "epoch": epoch, "from_node": node.node_id,
                        "action": action_label,
                        "penalty": penalty,
                        "seconds": stats.get("transition_seconds", 0.0),
                        "profile_count": len(profile_list)})
                for branch_prob, duration, new_status in branches:
                    target_key = (epoch + 1, new_status)
                    incoming_path = node.reaching_weight * branch_prob
                    action_label = "do_nothing" if action_name == "do_nothing" \
                        else project.project_id
                    # links.csv records this branch's own closed
                    # set: in-progress links as in the transition,
                    # plus the newly started link at THIS branch's
                    # duration (the transition itself is identical
                    # for every branch, because every duration is
                    # >= 360 minutes and the step is 360).
                    link_closed = dict(closed)
                    if action_name == "start":
                        link_closed[project.link] = \
                            duration * EPOCH_MINUTES
                    if target_key not in nodes_by_key:
                        target = add_node(epoch + 1, new_status,
                                          end_session, incoming_path,
                                          group_id)
                    else:
                        target = nodes_by_key[target_key]
                        merged = merge_traffic(
                            target.traffic, end_session,
                            target.reaching_weight, incoming_path)
                        target.traffic = merged
                        target.reaching_weight += incoming_path
                        target.group_id = next_group_id
                        next_group_id += 1
                        layer_merges += 1
                        totals["merges"] += 1
                    links.append(_Link(
                        next_link_id, node.node_id, target.node_id,
                        action_label, branch_prob, penalty,
                        link_closed, len(profile_list)))
                    next_link_id += 1
        layer_stat = {"epoch": epoch,
                      "nodes_at_epoch": len(layer_nodes),
                      "actions": layer_actions,
                      "transitions_computed": layer_computed,
                      "cache_hits": layer_hits,
                      "merges": layer_merges,
                      "nodes_total": len(nodes),
                      "links_total": len(links),
                      "seconds": time.time() - layer_started}
        layer_stats.append(layer_stat)
        if verbose:
            print("epoch %d: %d nodes, %d actions, %d computed, "
                  "%d cache hits, %d merges, %.1fs"
                  % (epoch, len(layer_nodes), layer_actions,
                     layer_computed, layer_hits, layer_merges,
                     layer_stat["seconds"]))
        if output_dir is not None:
            network = Network(nodes, links, projects, horizon,
                              start_date, layer_stats, totals)
            network.write_outputs(output_dir, groups, coords_for_output)
            layers_dir = os.path.join(output_dir, "layers")
            os.makedirs(layers_dir, exist_ok=True)
            with open(os.path.join(layers_dir, "layer_%d.json" % epoch),
                      "w", encoding="utf-8") as fh:
                json.dump(layer_stat, fh, indent=2)
            # Single latest checkpoint only: overwritten after
            # every completed layer. The transition cache is
            # deliberately NOT persisted (see checkpoint.py):
            # resume recomputes transitions in the resumed
            # layers, which cannot change the network (nodes,
            # links, penalties, reaching weights), though
            # recomputed group_ids may differ from an
            # uninterrupted run and the wall-clock reporting
            # counters (transition_seconds, transition_log
            # seconds) always differ.
            checkpoint_path = save_checkpoint(
                output_dir,
                build_state(epoch + 1, horizon, start_date,
                            fingerprint, projects, nodes, links,
                            next_node_id, next_link_id, next_group_id,
                            totals, layer_stats))
            if verbose:
                print("checkpoint %s: epoch %d, %.1f MB"
                      % (checkpoint_path, epoch + 1,
                         os.path.getsize(checkpoint_path) / 1e6))
    network = Network(nodes, links, projects, horizon, start_date,
                      layer_stats, totals)
    if output_dir is not None:
        network.write_outputs(output_dir, groups, coords_for_output)
    return network
