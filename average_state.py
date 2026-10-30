"""Build one resumable average session from v3 stepwise PySim sessions.

Placement and import requirement. This file must sit in the parent
directory of the ``pysim_construction/`` folder, that is, alongside
the v3 package, with that parent directory on ``sys.path``. In
practice: extract ``pysim_construction_v3.zip`` into the same
directory as this script, then ``from average_state import
average_state``. The script imports the v3 package (``session``,
``loading`` and ``cpp``); it does NOT contain a copy of the PySim
engine. If the package cannot be imported, importing this module
raises an ImportError that says exactly that, instead of failing
later with a confusing name error. Nothing in this module reads,
writes or averages construction state: construction stays external
to the session in v3, and the caller passes a fresh closed-links
dict to every resumed ``run_segment`` call, exactly as usual.

Use case (fixed by the caller, not re-litigated here). All input
sessions are at the SAME clock time. They are NOT from the same
demand samples: agent sets are unaligned, and an agent id that
appears in two sessions may represent two different trips. The
construction decisions behind them are the SAME, so construction
is never averaged. Resumed runs use the SAME future demand for
every case, so the average session's agent ids are NOT reassigned:
future demand ids are assumed distinct from every past id in the
input sessions, and reassigning would break that assumption.

API::

    average_state(sessions, verbose=False) -> session
    medoid_session(sessions) -> session

``sessions`` is a non-empty list of v3 ``Session`` objects. Both
functions return a NEW Session object, never an input object, and
neither mutates any input session. The returned session is usable
directly as ``run_segment(avg, future_agents, closed_dict,
step_minutes)``. An ``average_info`` plain dict is attached to the
returned session (``run_segment`` ignores the extra attribute);
it carries the diagnostics described below. A single session
returns a deepcopy of that session.

Algorithm.

A. Validation, fail loud. The list must be non-empty; every
element must be a v3 Session; every ``now`` must agree within
1e-9; every network must match (same node count, link count,
link map and flow model); the step size must be the v3 0.1
minutes. Anything else raises ValueError before anything is
copied, so a failed call leaves every input untouched.

B. Medoid first. For each session a feature vector phi is built
from per-link on-link counts (``cflow_arrival -
cflow_departure``), per-link loading, entrance and exit queue
lengths, per-link ``prevailing_tt``, and the totals in-system,
waiting, generated and completed, using the exact v3 field
names. The mean phi is computed, and the medoid is the session
minimising the Euclidean distance to it. Count features enter
unweighted; each ``prevailing_tt`` term is normalised by that
link's free-flow time (``prevailing_tt / fftt``) so that
travel-time scale does not dominate vehicle counts. The medoid
is deepcopied as the base for the average session. This
guarantees the base vehicle set has unique ids, feasible paths
and a consistent internal structure, which a componentwise
average of raw state could never provide.

C. Mean targets. For each link and each queue category
(loading, entrance, exit), the target count is the mean count
across sessions, rounded by largest-remainder rounding within
that category, so each category total equals the rounded mean
category total. The three category totals are then adjusted, if
rounding leaves them off by one or two, to the rounded mean
total in-system (the adjustment goes to the category with the
largest mean total). The on-link target per link is the sum of
its entrance and exit targets, not a separately rounded number,
so on-link and queue targets cannot disagree. Waiting and
completed targets are the rounded means. Mean ``prevailing_tt``
per link, mean MOE per link per minute up to ``now``, and mean
generated/completed are computed alongside.

D. Adjustment from the medoid base toward the targets. Counters
are never just overwritten; vehicles are actually moved, added
or removed, and only feasible placements are ever produced. A
move takes a vehicle on an over-target (link, queue) cell to an
under-target cell on the SAME vehicle's path, ahead of or
behind its current position: ``sim_link_seq`` is updated, the
remaining-path free-flow time (``remaining_fftt``) is recomputed
as the FFTT sum from the new position onward, and the arrival
stamps are rewritten as follows. A forward move keeps the
traversed stamps and synthesises stamps for the links it skips
as departure plus cumulative mean link times (mean
``prevailing_tt`` along the path); the new current link keeps
the unvisited sentinel (99999, the v3 default). A backward move
keeps the prefix before the new position and clears every stamp
from the new position onward to that sentinel. The queue item
stamp is synthesised the same way, and its next-node number
follows the generation rule (the to-node of the following path
link, or -1). Vehicles that stay put keep their original queue
item exactly. If the mean total is below the medoid total,
in-system vehicles are removed, preferring over-target cells
and then latest departures, and never removing a vehicle that
is the sole occupant of an at-or-below-target cell. If the mean
total is above the medoid total, vehicles are added from other
sessions whose agent id is NOT in the medoid base set and whose
current (link, queue) cell is under target, preferring earliest
departures, deepcopied with their donor stamps and path intact.
Waiting and completed sets are adjusted the same way (latest
departures out, earliest departures in). When the total
already matches but the per-link distribution does not --
the common case when scenarios differ in OD mix, so no
vehicle can move between the mismatched links -- an
over-target vehicle is swapped for an under-target donor
vehicle (latest departure out, earliest departure in,
same eligibility rules). Each swap keeps the total fixed
and reduces the total absolute residual by two. If a target cannot be
met because no feasible move or add exists, the medoid
placement is kept for those vehicles and the residual is
recorded; a vehicle on a link outside its path, a duplicate id,
or a vehicle in two queues at once is never produced as a
"residual" -- those are hard failures, and the adjustment code
raises rather than emit them.

E. Link state consistency after placement. Each link's three
queues are rebuilt from the placed vehicles in
(departure, agent id) order. ``cflow_arrival`` is set to the
rounded mean arrival count, then ``cflow_departure`` is set to
``cflow_arrival`` minus the placed on-link count, so the
difference exactly matches the placed vehicles while the
historical magnitude is preserved (if the rounded mean arrival
is below the placed on-link count, arrival is raised to the
on-link count and the departure is zero; the deviation is
recorded). Cumulative capacity counters (``cum_out_counter``,
``cum_in_counter``, ``cum_merge_counter``) and the ring buffers
(``ring_arrival``, ``ring_departure``) are mean-averaged;
``cum_out_prev`` and ``cum_in_prev`` are rounded means.
Per-step capacities (``link_in_capacity``,
``link_out_capacity``) are kept from the medoid; the loader
recomputes them on the next step. Demand and vehicle-type
counts are mean-rounded and then adjusted so the type-1 count
equals ``cflow_arrival`` (every bare-mode agent is type 1).
``prevailing_tt`` and ``departure_based_tt`` are set to their
means, ``departure_count`` to its rounded mean, and
``buffer_waiting_time`` to its mean; the prevailing-time
shortest-path network is then rebuilt from the mean
``prevailing_tt`` values. MOE history is the mean MOE up to
``now``, except the current-minute cumulative values, which are
set to the synthesised ``cflow`` counters at ``now``, and the
current-minute exit queue length, which is set to the placed
exit queue length. Minutes after ``now`` keep the medoid values
(they are unwritten). ``td_map`` is the medoid's, with removed
vehicles taken out and added vehicles inserted in their
departure buckets; this is correct because ``run_segment``
appends new agents' buckets to ``td_map`` rather than
rebuilding it, and past buckets are never revisited once the
clock has moved on (verified against ``session.py``).
``step_count`` is the common clock, ``segment_penalty`` is
reset to 0.0, and ``sim.construction`` is set to None because
``run_segment`` always installs a fresh closures object before
stepping. Generated and completed counters are set from the
actual placed sets (see below). The ``seen_ids`` set is the
union over all input sessions, so a future agent reusing any
past id still fails loud.

F. Waiting vehicles. The waiting dict is rebuilt from the
placed waiting vehicles, with unique ids, their waiting node
and their construction-wait bookkeeping fields
(``construction_wait_since``, ``construction_waited``) copied
as-is from the source vehicle.

G. Diagnostics. ``average_info`` on the returned session holds:
``n_input_sessions``, ``medoid_index``, ``mean_total_in_system``,
``actual_total_in_system``, per-link target vs actual counts,
``residual per link`` (target minus actual total per link),
``mean_prevailing_tt`` used, and ``unmet_targets``. A warning
is issued (``warnings.warn``) if any link residual exceeds 2
vehicles in absolute value, or the total residual exceeds 5%
of the mean total in-system (the sum of absolute link
residuals is also reported and also triggers the warning).

H. No construction averaging. This module never reads, writes
or averages any closed-links dict.

Accounting correction to the brief. The brief asks that
``len(vehicles) == generated`` after completed vehicles are
copied. That holds only when every registered vehicle was
generated. Real v3 sessions also register vehicles that were
never generated (pathless agents, which stay in ``vehicles``
with no path) and vehicles that are waiting (registered but
not yet loaded, hence not counted in ``generated``). The
invariant enforced here, and the one the loader actually
maintains, is ``generated == len(in-system) + len(completed)``,
with ``len(vehicles)`` equal to that plus waiting plus
pathless vehicles. Completed ids are never dropped silently:
a completed vehicle removed to meet a target stays in
``seen_ids``, so a later segment reusing its id still fails
loud, exactly as in a real session.

Limitations (read before using the average session for
decisions). (1) The average session is a synthetic state, not
a state any single run produced: its future penalty is NOT
expected to equal the mean penalty of the input sessions, and
the gap is the expected nonlinearity of queueing and rerouting,
not a defect, unless a feasibility check fails. (2) Synthetic
arrival stamps from moved vehicles anchor those vehicles'
future link travel times, so their future traversal penalties
carry that approximation. (3) MOE minutes before ``now`` are
means, not the history of any real run, and the current-minute
cumulative values are synthesised; MOE is diagnostic in bare
mode and does not drive the resumed dynamics (the prevailing
network carries free-flow times for information-type-1 agents).
(4) Link RNG states, generation RNG state and merge counters
are kept from the medoid; they are not averaged, because
averaging RNG state is meaningless. (5) Sessions with
different horizons keep the medoid's horizon; ``run_segment``
grows it as needed. (6) Only bare v3 sessions are supported:
no signals, emissions, zones or demand files, matching v3.
"""

import copy
import math
import os
import sys
import warnings

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

try:
    from pysim_construction.cpp import f32
    from pysim_construction.loading import SIM_INTERVAL
    from pysim_construction.session import Session
except ImportError as exc:
    raise ImportError(
        "average_state.py must sit alongside the pysim_construction/ "
        "folder: extract pysim_construction_v3.zip into the same "
        "directory as this script, then import again. "
        "(Original import error: %s)" % (exc,)) from exc

__all__ = ["average_state", "medoid_session"]

_EPS = 1e-9
_SENTINEL = f32(99999.0)
_CATEGORIES = ("loading", "entrance", "exit")


def _link_key(world, link):
    return "%d->%d" % (world.nodes[link.from_id].number,
                       world.nodes[link.to_id].number)


def _validate(sessions):
    if isinstance(sessions, (str, bytes)) or \
            not isinstance(sessions, (list, tuple)):
        raise ValueError(
            "sessions must be a non-empty list of v3 Session objects")
    if len(sessions) == 0:
        raise ValueError(
            "sessions must be a non-empty list of v3 Session objects")
    for item in sessions:
        if not isinstance(item, Session):
            raise ValueError(
                "all sessions must be v3 Session objects, got %r"
                % (item,))
    if abs(float(SIM_INTERVAL) - 0.1) > _EPS:
        raise ValueError(
            "expected the v3 step size 0.1, got %r" % (SIM_INTERVAL,))
    now0 = sessions[0].now
    for session in sessions[1:]:
        if abs(session.now - now0) > _EPS:
            raise ValueError(
                "sessions are at different clock times: %s vs %s"
                % (session.now, now0))
    world0 = sessions[0].world
    nodes0 = [node.number for node in world0.nodes]
    for session in sessions[1:]:
        if session.flow_model != sessions[0].flow_model:
            raise ValueError(
                "sessions have different flow models: %s vs %s"
                % (session.flow_model, sessions[0].flow_model))
        world = session.world
        if len(world.nodes) != len(world0.nodes) or \
                len(world.links) != len(world0.links):
            raise ValueError(
                "sessions are on different networks (node/link counts "
                "differ: %d nodes / %d links vs %d nodes / %d links)"
                % (len(world.nodes), len(world.links),
                   len(world0.nodes), len(world0.links)))
        if [node.number for node in world.nodes] != nodes0:
            raise ValueError(
                "sessions are on different networks (node sets differ)")
        if world.link_map != world0.link_map:
            raise ValueError(
                "sessions are on different networks (link maps differ)")
    return list(sessions)


def _session_counts(session):
    # Per-link queue counts and flow counters, plus session totals.
    # In-system here means queued (loading, entrance or exit); that
    # is exactly generated minus completed in a real session.
    per_link = []
    for link in session.world.links:
        per_link.append({
            "loading": len(link.loading_buffer),
            "entrance": len(link.entrance_queue),
            "exit": len(link.exit_queue),
            "on_link": link.cflow_arrival - link.cflow_departure,
            "cflow_arrival": link.cflow_arrival,
            "cflow_departure": link.cflow_departure,
            "prevailing_tt": float(link.prevailing_tt),
            "fftt": float(link.fftt),
        })
    totals = {
        "in_system": sum(d["loading"] + d["entrance"] + d["exit"]
                         for d in per_link),
        "waiting": len(session.sim.waiting),
        "generated": session.sim.generated,
        "completed": session.sim.completed,
    }
    return per_link, totals


def _phi(session):
    per_link, totals = _session_counts(session)
    vec = []
    for entry in per_link:
        vec.append(float(entry["on_link"]))
        vec.append(float(entry["loading"]))
        vec.append(float(entry["entrance"]))
        vec.append(float(entry["exit"]))
        fftt = entry["fftt"] if entry["fftt"] > 0 else 1.0
        vec.append(entry["prevailing_tt"] / fftt)
    vec.append(float(totals["in_system"]))
    vec.append(float(totals["waiting"]))
    vec.append(float(totals["generated"]))
    vec.append(float(totals["completed"]))
    return vec


def _medoid_index(sessions):
    vectors = [_phi(session) for session in sessions]
    mean = [sum(vec[i] for vec in vectors) / len(vectors)
            for i in range(len(vectors[0]))]
    best = None
    best_dist = None
    for i, vec in enumerate(vectors):
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, mean)))
        if best_dist is None or dist < best_dist - 1e-12:
            best = i
            best_dist = dist
    return best


def medoid_session(sessions):
    """Return a deepcopy of the session closest to the mean feature
    vector (see the module docstring, step B). Never an input
    object itself, and inputs are never mutated."""
    sessions = _validate(sessions)
    return copy.deepcopy(sessions[_medoid_index(sessions)])


def _largest_remainder(means, total):
    # Round means to integers summing to total, giving the extra
    # units to the largest fractional parts (ties: lowest index).
    floors = [int(math.floor(m)) for m in means]
    out = list(floors)
    left = total - sum(floors)
    order = sorted(range(len(means)),
                   key=lambda i: (-(means[i] - floors[i]), i))
    k = 0
    while left > 0 and order:
        out[order[k % len(order)]] += 1
        left -= 1
        k += 1
    while left < 0 and order:
        # Only reachable if total < sum(floors); trim from the
        # smallest fractional parts so the result stays closest.
        idx = order[-1 - (k % len(order))]
        if out[idx] > 0:
            out[idx] -= 1
            left += 1
        k += 1
        if k > 10 * len(order) + 10:
            break
    return out


def _classify(session):
    """Split a session's vehicles into in-system placements,
    waiting, completed and merely registered (pathless) sets.

    Returns (in_system, waiting, completed, registered) where
    in_system maps agent id -> (link_no, category, queue item
    copy). A vehicle found in two queues, or in a queue and in
    waiting/completed, means a corrupt session and raises
    ValueError rather than letting the average inherit it.
    """
    world = session.world
    sim = session.sim
    in_system = {}
    for link in world.links:
        queues = (("loading", link.loading_buffer),
                  ("entrance", link.entrance_queue),
                  ("exit", link.exit_queue))
        for category, queue in queues:
            for item in queue:
                aid = item[0]
                if aid in in_system:
                    raise ValueError(
                        "vehicle %d is in two queues at once" % aid)
                in_system[aid] = (link.link_no, category, list(item))
    waiting = set(sim.waiting)
    completed = set(aid for aid, veh in world.vehicles.items()
                    if veh.complete)
    for aid in waiting:
        if aid in in_system:
            raise ValueError(
                "waiting vehicle %d is also in a link queue" % aid)
    for aid in completed:
        if aid in in_system or aid in waiting:
            raise ValueError(
                "completed vehicle %d is also queued or waiting" % aid)
    registered = set()
    for aid in world.vehicles:
        if aid not in in_system and aid not in waiting \
                and aid not in completed:
            registered.add(aid)
    return in_system, waiting, completed, registered


def _path_links(veh):
    if veh.link_ary is None:
        return []
    return [entry.link_no for entry in veh.link_ary[:-1]]


def _recompute_remaining(veh, world):
    # Remaining-path FFTT from the vehicle's current position
    # onward, including its current link (v3 semantics: the
    # current link's FFTT leaves the total only when the link
    # is traversed).
    path = _path_links(veh)
    total = 0.0
    for link_no in path[veh.sim_link_seq:]:
        total += float(world.links[link_no].fftt)
    veh.remaining_fftt = total


def _synth_stamp(veh, upto, mean_prevailing):
    # Departure plus cumulative mean link times through path
    # index upto (inclusive). Used for synthesised queue-item
    # stamps and skipped-link arrival stamps after a move.
    path = _path_links(veh)
    total = float(veh.departure)
    for link_no in path[:upto + 1]:
        total += mean_prevailing[link_no]
    return f32(total)


def _apply_move(veh, new_seq, mean_prevailing, world):
    """Move a vehicle along its own path to index new_seq,
    rewriting stamps and remaining FFTT per the docstring."""
    path = _path_links(veh)
    old_seq = veh.sim_link_seq
    if new_seq < 0 or new_seq >= len(path):
        raise ValueError(
            "vehicle %d cannot move to path index %d (path length %d)"
            % (veh.agent_id, new_seq, len(path)))
    if new_seq > old_seq:
        for k in range(old_seq, new_seq):
            veh.link_ary[k].abs_arrival = _synth_stamp(veh, k,
                                                       mean_prevailing)
        veh.link_ary[new_seq].abs_arrival = _SENTINEL
    elif new_seq < old_seq:
        for k in range(new_seq, len(path)):
            veh.link_ary[k].abs_arrival = _SENTINEL
    veh.sim_link_seq = new_seq
    _recompute_remaining(veh, world)
    return _synth_stamp(veh, new_seq, mean_prevailing)


def _next_node_number(veh, world):
    path = _path_links(veh)
    nxt = veh.sim_link_seq + 1
    if nxt < len(path):
        return world.nodes[world.links[path[nxt]].to_id].number
    return -1


def _mean(values):
    return sum(values) / len(values)


def _drop(avg, aid):
    # Remove a vehicle from the session's vehicle registry and
    # waiting set. Its id stays in seen_ids (the caller keeps
    # the union), so a later segment reusing the id still
    # fails loud, exactly as in a real session.
    world = avg.world
    if aid in world.vehicles:
        del world.vehicles[aid]
    if aid in world.vehicle_order:
        world.vehicle_order.remove(aid)
    if aid in avg.sim.waiting:
        del avg.sim.waiting[aid]


def _bucket(departure):
    return int(f32(float(departure) * f32(10)))


def average_state(sessions, verbose=False):
    """Average a list of v3 sessions at a common clock into one
    new resumable Session. See the module docstring for the
    algorithm, the accounting correction and the limitations."""
    sessions = _validate(sessions)
    n_sessions = len(sessions)
    med_index = _medoid_index(sessions)
    medoid = sessions[med_index]
    world0 = medoid.world
    n_links = len(world0.links)

    counts_list = [_session_counts(s) for s in sessions]
    mean_loading = [_mean([c[0][i]["loading"] for c in counts_list])
                    for i in range(n_links)]
    mean_entrance = [_mean([c[0][i]["entrance"] for c in counts_list])
                     for i in range(n_links)]
    mean_exit = [_mean([c[0][i]["exit"] for c in counts_list])
                 for i in range(n_links)]
    mean_prevailing = [_mean([c[0][i]["prevailing_tt"]
                              for c in counts_list])
                       for i in range(n_links)]
    mean_cflow_arrival = [_mean([c[0][i]["cflow_arrival"]
                                 for c in counts_list])
                          for i in range(n_links)]
    mean_in_system = _mean([c[1]["in_system"] for c in counts_list])
    mean_waiting = _mean([c[1]["waiting"] for c in counts_list])
    mean_completed = _mean([c[1]["completed"] for c in counts_list])
    mean_generated = _mean([c[1]["generated"] for c in counts_list])

    # Targets: largest-remainder per category, then reconcile
    # the category totals with the rounded mean total in-system.
    total_target = int(round(mean_in_system))
    cat_means = {"loading": _mean(mean_loading),
                 "entrance": _mean(mean_entrance),
                 "exit": _mean(mean_exit)}
    cat_totals = {cat: int(round(cat_means[cat]))
                  for cat in _CATEGORIES}
    cat_targets = {
        "loading": _largest_remainder(mean_loading,
                                      cat_totals["loading"]),
        "entrance": _largest_remainder(mean_entrance,
                                       cat_totals["entrance"]),
        "exit": _largest_remainder(mean_exit, cat_totals["exit"]),
    }
    delta = total_target - sum(cat_totals.values())
    if delta != 0:
        # Give the difference to the category with the largest
        # mean total (ties: entrance, then exit, then loading).
        pick = sorted(_CATEGORIES,
                      key=lambda cat: (-cat_means[cat],
                                       ("entrance", "exit",
                                        "loading").index(cat)))[0]
        new_total = max(0, cat_totals[pick] + delta)
        cat_totals[pick] = new_total
        means = {"loading": mean_loading,
                 "entrance": mean_entrance,
                 "exit": mean_exit}[pick]
        cat_targets[pick] = _largest_remainder(means, new_total)
    waiting_target = int(round(mean_waiting))
    completed_target = int(round(mean_completed))
    targets = {}
    for i in range(n_links):
        targets[i] = {"loading": cat_targets["loading"][i],
                      "entrance": cat_targets["entrance"][i],
                      "exit": cat_targets["exit"][i]}

    # Base sets from the medoid (read-only), donor pools from
    # the other sessions. Donor adds are restricted to ids
    # outside the medoid base set, because ids are unaligned
    # across sessions and a reused id means a different trip.
    med_in_system, med_waiting, med_completed, _med_reg = \
        _classify(medoid)
    base_ids = set(medoid.world.vehicles)
    donor_in_system = []
    donor_waiting = []
    donor_completed = []
    for idx, session in enumerate(sessions):
        if idx == med_index:
            continue
        in_sys, waiting, completed, _reg = _classify(session)
        for aid, (link_no, category, item) in in_sys.items():
            donor_in_system.append(
                (float(session.world.vehicles[aid].departure), aid,
                 link_no, category, item,
                 session.world.vehicles[aid]))
        for aid in waiting:
            donor_waiting.append(
                (float(session.world.vehicles[aid].departure), aid,
                 session.world.vehicles[aid]))
        for aid in completed:
            veh = session.world.vehicles[aid]
            donor_completed.append(
                (float(veh.departure), float(veh.arrival), aid, veh))
    donor_in_system.sort(key=lambda row: (row[0], row[1]))
    donor_waiting.sort(key=lambda row: (row[0], row[1]))
    donor_completed.sort(key=lambda row: (row[0], row[2]))

    avg = copy.deepcopy(medoid)
    world = avg.world
    sim = avg.sim
    removed_aids = set()
    added_aids = []

    def drop(aid):
        _drop(avg, aid)
        removed_aids.add(aid)

    # Waiting set: latest departures out, earliest departures in.
    waiting = set(_classify(avg)[1])
    while len(waiting) > waiting_target:
        cand = max(waiting,
                   key=lambda aid: (float(world.vehicles[aid].departure),
                                    aid))
        drop(cand)
        waiting.discard(cand)
    for departure, aid, veh in donor_waiting:
        if len(waiting) >= waiting_target:
            break
        if aid in base_ids or aid in world.vehicles:
            continue
        veh_copy = copy.deepcopy(veh)
        world.vehicles[aid] = veh_copy
        sim.waiting[aid] = veh_copy
        waiting.add(aid)
        added_aids.append(aid)

    # Completed set: latest arrivals out, earliest departures in.
    completed = set(_classify(avg)[2])
    while len(completed) > completed_target:
        cand = max(completed,
                   key=lambda aid: (float(world.vehicles[aid].arrival),
                                    float(world.vehicles[aid].departure),
                                    aid))
        drop(cand)
        completed.discard(cand)
    for departure, _arrival, aid, veh in donor_completed:
        if len(completed) >= completed_target:
            break
        if aid in base_ids or aid in world.vehicles:
            continue
        veh_copy = copy.deepcopy(veh)
        world.vehicles[aid] = veh_copy
        completed.add(aid)
        added_aids.append(aid)

    # In-system placement: aid -> [link_no, category], plus the
    # queue item that will be rebuilt into that link's queue.
    placement = {}
    items = {}
    for aid, (link_no, category, item) in _classify(avg)[0].items():
        placement[aid] = [link_no, category]
        items[aid] = list(item)

    def cell_counts():
        counts = {}
        for i in range(n_links):
            for cat in _CATEGORIES:
                counts[(i, cat)] = 0
        for link_no, category in placement.values():
            counts[(link_no, category)] += 1
        return counts

    def move_pass():
        # One improving move at a time, from an over-target cell
        # to an under-target cell on the same vehicle's path,
        # until no such move exists. Moves never change the
        # total, only the per-link distribution.
        while True:
            actual = cell_counts()
            best = None
            for aid, (link_no, category) in placement.items():
                if actual[(link_no, category)] <= \
                        targets[link_no][category]:
                    continue
                veh = world.vehicles[aid]
                path = _path_links(veh)
                seq = veh.sim_link_seq
                if seq < 0 or seq >= len(path) or \
                        path[seq] != link_no:
                    raise ValueError(
                        "vehicle %d: queue link %d does not match "
                        "its path position" % (aid, link_no))
                for j, link_j in enumerate(path):
                    if j == seq:
                        continue
                    for cat2 in _CATEGORIES:
                        if actual[(link_j, cat2)] >= \
                                targets[link_j][cat2]:
                            continue
                        score = (0 if cat2 == category else 1,
                                 abs(j - seq))
                        cand = (score, float(veh.departure), aid,
                                link_j, cat2, j)
                        if best is None or cand < best:
                            best = cand
            if best is None:
                return
            _score, _dep, aid, link_j, cat2, j = best
            veh = world.vehicles[aid]
            stamp = _apply_move(veh, j, mean_prevailing_by_link,
                                world)
            placement[aid] = [link_j, cat2]
            items[aid] = [aid, _next_node_number(veh, world), stamp]

    mean_prevailing_by_link = {}
    for i, link in enumerate(world0.links):
        mean_prevailing_by_link[link.link_no] = mean_prevailing[i]

    move_pass()

    # Total adjustment: removals prefer over-target cells, then
    # latest departures, and never touch an at-or-below-target
    # cell; additions take under-target donor cells, earliest
    # departures first. A final move pass then rebalances the
    # per-link distribution at the new total.
    while len(placement) > total_target:
        actual = cell_counts()
        cands = [aid for aid, (link_no, category)
                 in placement.items()
                 if actual[(link_no, category)] >
                 targets[link_no][category]]
        if not cands:
            break
        aid = max(cands,
                  key=lambda a: (float(world.vehicles[a].departure),
                                 a))
        drop(aid)
        del placement[aid]
        del items[aid]
    while len(placement) < total_target:
        actual = cell_counts()
        picked = None
        for departure, aid, link_no, category, item, veh \
                in donor_in_system:
            if aid in base_ids or aid in world.vehicles:
                continue
            if actual[(link_no, category)] < \
                    targets[link_no][category]:
                picked = (aid, link_no, category, item, veh)
                break
        if picked is None:
            break
        aid, link_no, category, item, veh = picked
        veh_copy = copy.deepcopy(veh)
        world.vehicles[aid] = veh_copy
        placement[aid] = [link_no, category]
        items[aid] = list(item)
        added_aids.append(aid)
    # Swaps: when the total already matches but the per-link
    # distribution does not, trade an over-target vehicle
    # for an under-target donor vehicle. Each swap keeps
    # the total fixed and reduces the total absolute
    # residual by two, so the loop always terminates.
    while True:
        actual = cell_counts()
        over_aids = [aid for aid, (link_no, category)
                     in placement.items()
                     if actual[(link_no, category)] >
                     targets[link_no][category]]
        if not over_aids:
            break
        picked = None
        for departure, aid, link_no, category, item, veh \
                in donor_in_system:
            if aid in base_ids or aid in world.vehicles:
                continue
            if actual[(link_no, category)] < \
                    targets[link_no][category]:
                picked = (aid, link_no, category, item, veh)
                break
        if picked is None:
            break
        aid_out = max(over_aids,
                      key=lambda a: (float(world.vehicles[a].departure),
                                     a))
        drop(aid_out)
        del placement[aid_out]
        del items[aid_out]
        aid, link_no, category, item, veh = picked
        veh_copy = copy.deepcopy(veh)
        world.vehicles[aid] = veh_copy
        placement[aid] = [link_no, category]
        items[aid] = list(item)
        added_aids.append(aid)
    move_pass()

    # Rebuild every link queue from the final placement, in
    # (departure, agent id) order within each queue.
    for link in world.links:
        link.loading_buffer.clear()
        link.entrance_queue.clear()
        link.exit_queue.clear()
    buckets = {}
    for aid, (link_no, category) in placement.items():
        buckets.setdefault((link_no, category), []).append(aid)
    for (link_no, category), aids in buckets.items():
        link = world.links[link_no]
        queue = {"loading": link.loading_buffer,
                 "entrance": link.entrance_queue,
                 "exit": link.exit_queue}[category]
        for aid in sorted(aids,
                          key=lambda a: (float(world.vehicles[a].departure),
                                         a)):
            queue.append(list(items[aid]))
    # Self-check: every placed vehicle sits on the link its
    # path position names, and every loaded, incomplete
    # vehicle is placed exactly once.
    _classify(avg)
    for aid, (link_no, _category) in placement.items():
        veh = world.vehicles[aid]
        path = _path_links(veh)
        if veh.sim_link_seq >= len(path) or \
                path[veh.sim_link_seq] != link_no:
            raise ValueError(
                "vehicle %d: rebuilt queue link %d does not match "
                "its path position" % (aid, link_no))
    for aid, veh in world.vehicles.items():
        if veh.loaded and not veh.complete and aid not in placement:
            raise ValueError(
                "loaded vehicle %d is in no queue after averaging"
                % aid)

    # Link state: counters, MOE, travel times, capacities.
    now_minute = int(avg.now)
    actual_counts = cell_counts()
    link_targets = {}
    link_actuals = {}
    link_residuals = {}
    for i, link in enumerate(world.links):
        key = _link_key(world, link)
        actual = {cat: actual_counts[(i, cat)] for cat in _CATEGORIES}
        actual["on_link"] = actual["entrance"] + actual["exit"]
        link_targets[key] = {"loading": targets[i]["loading"],
                             "entrance": targets[i]["entrance"],
                             "exit": targets[i]["exit"],
                             "on_link": targets[i]["entrance"] +
                             targets[i]["exit"]}
        link_actuals[key] = dict(actual)
        link_residuals[key] = (link_targets[key]["loading"] +
                               link_targets[key]["entrance"] +
                               link_targets[key]["exit"]) - \
            (actual["loading"] + actual["entrance"] + actual["exit"])
        arr_target = int(round(mean_cflow_arrival[i]))
        arr_set = max(arr_target, actual["on_link"])
        link.cflow_arrival = arr_set
        link.cflow_departure = arr_set - actual["on_link"]
        link.prevailing_tt = f32(mean_prevailing[i])
        link.departure_based_tt = f32(_mean(
            [float(s.world.links[i].departure_based_tt)
             for s in sessions]))
        link.departure_count = int(round(_mean(
            [s.world.links[i].departure_count for s in sessions])))
        link.buffer_waiting_time = f32(_mean(
            [float(s.world.links[i].buffer_waiting_time)
             for s in sessions]))
        link.cum_out_counter = f32(_mean(
            [float(s.world.links[i].cum_out_counter)
             for s in sessions]))
        link.cum_in_counter = f32(_mean(
            [float(s.world.links[i].cum_in_counter)
             for s in sessions]))
        link.cum_merge_counter = f32(_mean(
            [float(s.world.links[i].cum_merge_counter)
             for s in sessions]))
        link.cum_out_prev = int(round(_mean(
            [s.world.links[i].cum_out_prev for s in sessions])))
        link.cum_in_prev = int(round(_mean(
            [s.world.links[i].cum_in_prev for s in sessions])))
        link.ring_arrival = [f32(_mean(
            [float(s.world.links[i].ring_arrival[k])
             for s in sessions])) for k in range(301)]
        link.ring_departure = [f32(_mean(
            [float(s.world.links[i].ring_departure[k])
             for s in sessions])) for k in range(301)]
        demand_means = [_mean([s.world.links[i].counts_demand[d]
                               for s in sessions]) for d in range(40)]
        vtype_means = [_mean([s.world.links[i].counts_vtype[d]
                              for s in sessions]) for d in range(15)]
        demand = [int(round(m)) for m in demand_means]
        vtype = [int(round(m)) for m in vtype_means]
        demand[1] += link.cflow_arrival - sum(demand)
        vtype[1] += link.cflow_arrival - sum(vtype)
        link.counts_demand = demand
        link.counts_vtype = vtype
        for t in range(len(link.moe)):
            if t > now_minute:
                continue  # unwritten minutes keep medoid values
            have = [s.world.links[i].moe[t] for s in sessions
                    if t < len(s.world.links[i].moe)]
            moe = link.moe[t]
            moe.total_travel_time = f32(_mean(
                [float(m.total_travel_time) for m in have]))
            moe.total_flow_count = int(round(_mean(
                [m.total_flow_count for m in have])))
            if t == now_minute:
                moe.cum_arrival = link.cflow_arrival
                moe.cum_departure = link.cflow_departure
                moe.exit_queue_length = actual["exit"]
            else:
                moe.cum_arrival = int(round(_mean(
                    [m.cum_arrival for m in have])))
                moe.cum_departure = int(round(_mean(
                    [m.cum_departure for m in have])))
                moe.exit_queue_length = int(round(_mean(
                    [m.exit_queue_length for m in have])))
    sim.prevailing_net.rebuild(True, avg.now)

    # Session-level state: counters from the actual placed sets,
    # vehicle order, departure buckets, and the union of ids.
    sim.generated = len(placement) + len(completed)
    sim.completed = len(completed)
    sim.waiting = {aid: world.vehicles[aid] for aid in waiting}
    sim.segment_penalty = 0.0
    sim.construction = None
    kept_order = [aid for aid in medoid.world.vehicle_order
                  if aid in world.vehicles]
    added_order = sorted(added_aids,
                         key=lambda a: (float(world.vehicles[a].departure),
                                        a))
    world.vehicle_order = kept_order + added_order
    for bucket_map in (sim.td_map, world.original_td_map):
        for bucket in list(bucket_map):
            kept = [aid for aid in bucket_map[bucket]
                    if aid not in removed_aids]
            if kept:
                bucket_map[bucket] = kept
            else:
                del bucket_map[bucket]
    for aid in added_aids:
        bucket = _bucket(world.vehicles[aid].departure)
        if aid not in sim.td_map.setdefault(bucket, []):
            sim.td_map[bucket].append(aid)
        if aid not in world.original_td_map.setdefault(bucket, []):
            world.original_td_map[bucket].append(aid)
    seen = set()
    for session in sessions:
        seen |= set(session.seen_ids)
    avg.seen_ids = seen

    # Diagnostics.
    actual_total = len(placement)
    total_residual = total_target - actual_total
    sum_abs_link_residual = sum(abs(v)
                                for v in link_residuals.values())
    unmet = []
    for key in link_targets:
        for cat in ("loading", "entrance", "exit", "on_link"):
            if link_targets[key][cat] != link_actuals[key][cat]:
                unmet.append("link %s %s: target %d, actual %d"
                             % (key, cat, link_targets[key][cat],
                                link_actuals[key][cat]))
    if len(waiting) != waiting_target:
        unmet.append("waiting: target %d, actual %d"
                     % (waiting_target, len(waiting)))
    if len(completed) != completed_target:
        unmet.append("completed: target %d, actual %d"
                     % (completed_target, len(completed)))
    if actual_total != total_target:
        unmet.append("total in-system: target %d, actual %d"
                     % (total_target, actual_total))
    mean_prevailing_tt = {}
    for i, link in enumerate(world.links):
        mean_prevailing_tt[_link_key(world, link)] = mean_prevailing[i]
    info = {
        "n_input_sessions": n_sessions,
        "medoid_index": med_index,
        "mean_total_in_system": mean_in_system,
        "target_total_in_system": total_target,
        "actual_total_in_system": actual_total,
        "total_residual": total_residual,
        "sum_abs_link_residual": sum_abs_link_residual,
        "mean_waiting": mean_waiting,
        "target_waiting": waiting_target,
        "actual_waiting": len(waiting),
        "mean_completed": mean_completed,
        "target_completed": completed_target,
        "actual_completed": len(completed),
        "mean_generated": mean_generated,
        "actual_generated": sim.generated,
        "link_targets": link_targets,
        "link_actuals": link_actuals,
        "link_residuals": link_residuals,
        "mean_prevailing_tt": mean_prevailing_tt,
        "unmet_targets": unmet,
    }
    avg.average_info = info
    worst_link = max((abs(v) for v in link_residuals.values()),
                     default=0)
    if worst_link > 2 or (mean_in_system > 0 and
                          (abs(total_residual) > 0.05 * mean_in_system
                           or sum_abs_link_residual >
                           0.05 * mean_in_system)):
        warnings.warn(
            "average_state: largest link residual %d, total residual "
            "%d on mean total %.1f; some mean targets could not be "
            "met with feasible placements (see average_info)"
            % (worst_link, total_residual, mean_in_system))
    if verbose:
        print("average_state: %d sessions, medoid %d, now %s, "
              "in-system target %d actual %d, waiting %d/%d, "
              "completed %d/%d"
              % (n_sessions, med_index, avg.now, total_target,
                 actual_total, len(waiting), waiting_target,
                 len(completed), completed_target))
        if unmet:
            print("average_state: unmet targets: %s"
                  % ("; ".join(unmet),))
    return avg


if __name__ == "__main__":
    print("average_state.py: import average_state and "
          "medoid_session from this file, with the "
          "pysim_construction/ folder alongside it.")
