# Stepwise, pause/resume PySim for the DDM project (v3).
#
# A session is created once from a bare network directory and
# then advanced one segment at a time with run_segment. The
# session object IS the resume state: it holds the world (nodes,
# links with their loading/entrance/exit queues, flow counters,
# cumulative counters, ring buffers, prevailing and
# departure-based travel times, and per-link MOE), the simulator
# (origin-waiting dict, generated/completed counts, departure
# buckets, prevailing-time shortest-path network), every vehicle
# (link sequence, arrival stamps, leaving-buffer time, original
# path and FFTT, remaining-path FFTT, loaded/complete flags),
# and the clock (step count; now = step_count * 0.1 minutes).
# Nothing about traffic is written to disk between segments,
# and run_segment writes no files at all. Construction state is
# deliberately NOT in the session: each segment's closed_links
# dict is the authority for that segment (see closures.py).
import csv
import math
import os

from .closures import SegmentClosures
from .cpp import f32
from .loading import SIM_INTERVAL, Simulator
from .network import Vehicle, VehicleLink, World
from .sp import find_best_path

_EPS = 1e-9


def create_session(network_dir, flow_model=2, random_seed=100):
    """Create a stepwise session from a bare network directory.

    network_dir holds input_node.csv and input_link.csv only,
    in the bare v2 schema (node_id only; link columns
    from_node_id, to_node_id, length, number_of_lanes,
    speed_limit, lane_capacity_in_vhc_per_hour, with optional
    jam_density default 180 and wave_speed default 12).
    Direction is ignored (one row is one directed link), link
    type is hardcoded to 1 / "f" (Freeway), and there are no
    signals, emissions, zones, demand files or settings files.

    The clock starts at 0.0 even if the first segment is empty.
    flow_model must be 1, 2 or 3 (default 2, Newell), anything
    else fails loud. random_seed is accepted but has no effect
    on results in bare mode: no RNG draw is made for
    information class or age (every agent is hardcoded to
    information_type 1, vehicle_type 1, age 5), exactly as in
    bare v2. Returns a Session.
    """
    if flow_model not in (1, 2, 3):
        raise ValueError(
            "flow_model must be 1, 2 or 3, got %s" % (flow_model,))
    network_dir = os.path.abspath(network_dir)
    if not os.path.isdir(network_dir):
        raise ValueError(
            "network_dir does not exist: %s" % network_dir)
    world = World(network_dir, flow_model=flow_model,
                  random_seed=random_seed, session_mode=True)
    world.read_inputs(load_agents=False)
    sim = Simulator(world, construction=None)
    return Session(network_dir, world, sim, flow_model, random_seed)


class Session:
    """Resume state for a stepwise run. Created by create_session;
    advanced only by run_segment. Do not mutate its fields by
    hand: the loader's queues, counters and MOE arrays only stay
    consistent when advanced through run_segment.
    """

    def __init__(self, network_dir, world, sim, flow_model,
                 random_seed):
        self.network_dir = network_dir
        self.world = world
        self.sim = sim
        self.flow_model = flow_model
        self.random_seed = random_seed
        # Global 0.1-minute step count. now is derived from it,
        # never accumulated, so segment boundaries never drift.
        self.step_count = 0
        # Every agent_id ever admitted, including intrazonal
        # agents (which are skipped, not simulated) and agents
        # that never found a path. A repeat in any later segment
        # fails loud.
        self.seen_ids = set()

    @property
    def now(self):
        return self.step_count * SIM_INTERVAL

    def _extend_horizon_to(self, target_time):
        # Grow the world's horizon (and the prevailing-time
        # shortest-path network's interval rows) so that MOE
        # minutes and path intervals exist up to target_time.
        # The prevailing network carries free-flow times only
        # in bare mode (information_type 1 never updates it), so
        # new intervals are filled with the same FFTT values.
        world = self.world
        if target_time <= world.horizon:
            return
        old_periods = world.num_periods
        new_horizon = int(math.ceil(target_time))
        world.extend_horizon(new_horizon)
        new_periods = world.num_periods
        if new_periods > old_periods:
            extra = new_periods - old_periods
            for link in world.links:
                bias = world.link_type_of(link.link_type).bias
                value = link.prevailing_tt * bias
                if value < f32(0.01):
                    value = f32(0.01)
                net = self.sim.prevailing_net
                net.td_time[link.link_no].extend([value] * extra)

    def completed_ids(self):
        return sorted(aid for aid, veh in self.world.vehicles.items()
                      if veh.complete)

    def waiting_ids(self):
        return sorted(self.sim.waiting)

    def vehicle_positions(self):
        # agent_id -> (from_node_id, to_node_id) of the link the
        # vehicle is currently on, or None if it is waiting at
        # its origin, not yet loaded, or already complete. A
        # vehicle in a link's loading buffer counts as on that
        # link (its sim_link_seq is already 0 and its first link
        # is fixed); a vehicle that never found a path has no
        # link at all. For inspection and tests only.
        positions = {}
        for aid, veh in self.world.vehicles.items():
            if veh.complete or aid in self.sim.waiting:
                positions[aid] = None
            elif veh.loaded and veh.link_ary is not None \
                    and veh.sim_link_seq < len(veh.link_ary) - 1:
                link = self.world.links[
                    veh.link_ary[veh.sim_link_seq].link_no]
                positions[aid] = (
                    self.world.nodes[link.from_id].number,
                    self.world.nodes[link.to_id].number)
            else:
                positions[aid] = None
        return positions


def _as_int(value, what):
    if isinstance(value, bool):
        raise ValueError("%s must be an integer, got %r" % (what, value))
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("%s must be an integer, got %r" % (what, value))


def _as_number(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number, got %r" % (what, value))
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("%s must be finite, got %r" % (what, value))
    return value


def _parse_agents(agents, world, now, window_end, seen_ids):
    # Validate a whole segment's agents BEFORE anything is
    # admitted, so a failing segment leaves the session
    # untouched. Returns a list of dicts with parsed fields,
    # plus a flag per agent for intrazonal (skipped) agents.
    if agents is None:
        agents = []
    if isinstance(agents, (str, bytes, os.PathLike)):
        agents = _read_agents_csv(agents)
    if not isinstance(agents, (list, tuple)):
        raise ValueError("agents must be a list of dicts or a CSV path")
    parsed = []
    pending = set()
    for item in agents:
        if not isinstance(item, dict):
            raise ValueError("each agent must be a dict, got %r" % (item,))
        if "agent_id" not in item:
            raise ValueError("agent missing agent_id: %r" % (item,))
        agent_id = _as_int(item["agent_id"], "agent_id")
        if agent_id in seen_ids or agent_id in pending:
            raise ValueError(
                "duplicate agent_id %d (agent ids are unique across "
                "all segments)" % agent_id)
        pending.add(agent_id)
        origin = item.get("origin_node_id")
        if origin is None:
            origin = item.get("from_origin_node_id")
        dest = item.get("destination_node_id")
        if dest is None:
            dest = item.get("to_destination_node_id")
        if origin is None or dest is None:
            raise ValueError(
                "agent %d: origin_node_id and destination_node_id "
                "are required (from_origin_node_id / "
                "to_destination_node_id accepted as aliases)"
                % agent_id)
        origin = _as_int(origin, "agent %d origin_node_id" % agent_id)
        dest = _as_int(dest, "agent %d destination_node_id" % agent_id)
        if origin not in world.node_index:
            raise ValueError(
                "agent %d origin node %d not in input_node.csv"
                % (agent_id, origin))
        if dest not in world.node_index:
            raise ValueError(
                "agent %d destination node %d not in input_node.csv"
                % (agent_id, dest))
        if "departure_time_in_min" not in item:
            raise ValueError(
                "agent %d: departure_time_in_min is required" % agent_id)
        departure = _as_number(item["departure_time_in_min"],
                               "agent %d departure_time_in_min" % agent_id)
        if departure < 0:
            raise ValueError(
                "agent %d departure %s is negative"
                % (agent_id, departure))
        # Half-open window [now, now+step_minutes): a departure
        # exactly at now+step_minutes belongs to the next
        # segment and fails here. The 1e-9 margins absorb float
        # representation error only (for example now computed
        # as 3 * 0.1 is 0.30000000000000004 while a caller
        # passes departure 0.3); they never admit a departure
        # a real step early or late.
        if departure < now - _EPS or departure >= window_end - _EPS:
            raise ValueError(
                "agent %d departure %s outside segment window "
                "[%s, %s)" % (agent_id, departure, now, window_end))
        path = item.get("path_node_sequence")
        if path is not None and path != "" and path != []:
            raise ValueError(
                "bare v3 agents must be pathless: agent %d has a "
                "path_node_sequence" % agent_id)
        parsed.append({
            "agent_id": agent_id, "origin": origin, "dest": dest,
            "departure": departure,
            "intrazonal": origin == dest,
        })
    return parsed


def _read_agents_csv(path):
    if not os.path.isfile(path):
        raise ValueError("agents CSV file not found: %s" % path)
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("agents CSV is empty: %s" % path)
        fields = [h.strip() for h in reader.fieldnames]
        rows = []
        for raw in reader:
            if raw is None:
                continue
            row = {}
            for key, value in raw.items():
                if key is None:
                    continue
                row[key.strip()] = value.strip() if isinstance(
                    value, str) else value
            if all(v in (None, "") for v in row.values()):
                continue
            agent = {}
            if "agent_id" in row and row["agent_id"] != "":
                try:
                    agent["agent_id"] = int(row["agent_id"])
                except ValueError:
                    raise ValueError(
                        "agents CSV: non-integer agent_id %r"
                        % row["agent_id"])
            origin = row.get("origin_node_id") or \
                row.get("from_origin_node_id")
            if origin:
                agent["origin_node_id"] = int(origin)
            dest = row.get("destination_node_id") or \
                row.get("to_destination_node_id")
            if dest:
                agent["destination_node_id"] = int(dest)
            if row.get("departure_time_in_min"):
                agent["departure_time_in_min"] = float(
                    row["departure_time_in_min"])
            if row.get("path_node_sequence"):
                agent["path_node_sequence"] = row["path_node_sequence"]
            rows.append(agent)
        return rows


def _parse_closed_links(closed_links, world):
    if closed_links is None:
        closed_links = {}
    if not isinstance(closed_links, dict):
        raise ValueError(
            "closed_links must be a dict {(from_node_id, to_node_id): "
            "minutes_left}, got %r" % (closed_links,))
    parsed = {}
    for key, value in closed_links.items():
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            raise ValueError(
                "closed_links key must be a (from_node_id, "
                "to_node_id) pair, got %r" % (key,))
        from_node = _as_int(key[0], "closed_links from_node_id")
        to_node = _as_int(key[1], "closed_links to_node_id")
        if (from_node, to_node) not in world.link_map:
            raise ValueError(
                "closed link %d->%d is not a link in the network"
                % (from_node, to_node))
        minutes_left = _as_number(
            value, "closed link %d->%d minutes_left"
            % (from_node, to_node))
        if minutes_left < 0:
            raise ValueError(
                "closed link %d->%d minutes_left %s is negative"
                % (from_node, to_node, minutes_left))
        parsed[(from_node, to_node)] = minutes_left
    return parsed


def run_segment(session, agents, closed_links, step_minutes):
    """Advance a session by one segment and return its penalty.

    session: a Session from create_session (mutated in place
    and returned as "state" for convenience).
    agents: a list of dicts with agent_id, origin_node_id,
    destination_node_id, departure_time_in_min (ABSOLUTE
    cumulative minutes, not relative to the segment start),
    or a CSV file path with those columns. Aliases
    from_origin_node_id / to_destination_node_id are accepted.
    Agents must be pathless. Departures must fall in the
    half-open window [now, now+step_minutes); a departure
    exactly at now+step_minutes belongs to the next segment
    and fails loud here. Intrazonal agents (origin == dest)
    are silently skipped, as in bare v2, but their ids are
    still recorded (a repeat id in any segment fails loud).
    closed_links: a dict {(from_node_id, to_node_id):
    minutes_left}. Absent or 0 means open. See closures.py
    for the countdown semantics; the dict is the authority
    for this segment only and is not carried forward.
    step_minutes: variable per call, must be > 0 and a
    multiple of 0.1 within 1e-9, otherwise fails loud.

    Returns exactly {"penalty": float, "state": session} and
    writes no files. The penalty is in vehicle-minutes on the
    excess_closed basis (see below). The session is mutated:
    a failed call (bad window, duplicate id, unknown node or
    link, bad step) raises before anything is admitted, so
    the session is still usable afterwards.

    Segment penalty, telescoping version. For each link
    traversal COMPLETED during the segment, add (actual link
    travel time - that link's FFTT), where the first link's
    travel time is link exit minus departure (so origin
    waiting, including waiting through earlier segments, is
    counted when the first link exits, not during the waiting
    segment) and later links are successive arrival-stamp
    differences. For each reroute event during the segment
    (the initial barred-avoiding path at generation, a
    waiting vehicle's release path, a loading-buffer reroute,
    and each en-route reroute at a node), add (new remaining
    path FFTT from the current node - old remaining path FFTT
    from the current node). The original path is the free-flow
    shortest path on the FULL network ignoring barred links,
    computed at admission; its FFTT is the baseline. This
    telescopes: over a vehicle's full trip, segment penalties
    sum to (arrival - departure) - original-path FFTT for a
    completed vehicle. It is NOT just the sum of (actual -
    FFTT) over actual links: that sum is excess_actual and
    misses the detour FFTT delta. A vehicle mid-link at
    segment end contributes nothing for that link yet, and an
    incomplete vehicle contributes only its traversed links
    plus its reroute deltas so far. A final drain segment
    (no new agents, no closures, long enough to drain) makes
    the segment penalties over all segments sum to the v2
    full-run excess_closed total for completed vehicles, up
    to the v2 travel_time-vs-trip_time buffer-wait difference
    (v2 excess uses arrival minus leaving-buffer time; the
    link-travel-time sum here uses arrival minus departure;
    the two agree exactly when vehicles enter their first
    link without storage-gate delay, as in uncongested runs).

    Generation timing precision is 0.1 minutes: departures
    are stored as floats and bucketed as in v2, by
    int(departure * 10) in float32 arithmetic. A departure of
    0.07 therefore generates at the 0.0 step and a departure
    of 0.12 at the 0.1 step; the free-flow stamp is still
    departure + FFTT, so uncongested travel times are exact
    and only the generation step is bucketed.
    """
    if not isinstance(session, Session):
        raise ValueError("session must come from create_session")
    if isinstance(step_minutes, bool) or not isinstance(
            step_minutes, (int, float)):
        raise ValueError("step_minutes must be a number, got %r"
                         % (step_minutes,))
    step_minutes = float(step_minutes)
    if step_minutes != step_minutes or \
            step_minutes in (float("inf"), float("-inf")):
        raise ValueError("step_minutes must be finite")
    if step_minutes <= 0:
        raise ValueError("step_minutes must be > 0, got %s"
                         % step_minutes)
    n_steps = int(round(step_minutes / SIM_INTERVAL))
    if n_steps < 1 or abs(step_minutes - n_steps * SIM_INTERVAL) > 1e-9:
        raise ValueError(
            "step_minutes must be a multiple of 0.1, got %s"
            % step_minutes)
    world = session.world
    sim = session.sim
    now = session.now
    window_end = (session.step_count + n_steps) * SIM_INTERVAL
    # Validate everything before mutating anything.
    parsed_agents = _parse_agents(agents, world, now, window_end,
                                  session.seen_ids)
    parsed_closed = _parse_closed_links(closed_links, world)
    # Horizon: keep 300 minutes of drain allowance past the
    # segment end, as v2 kept horizon = end + 300. Segments
    # may continue indefinitely; the MOE and path-interval
    # arrays grow instead of capping the run.
    session._extend_horizon_to(window_end + 300)
    for agent in parsed_agents:
        session.seen_ids.add(agent["agent_id"])
        if agent["intrazonal"]:
            continue
        veh = Vehicle(agent["agent_id"])
        veh.origin_zone = agent["origin"]
        veh.dest_zone = agent["dest"]
        veh.origin_node = world.node_index[agent["origin"]]
        veh.dest_node = world.node_index[agent["dest"]]
        veh.departure = agent["departure"]
        veh.demand_type = 1
        veh.vehicle_type = 1
        veh.pce = f32(1)
        veh.info_type = 1
        veh.vot = f32(10)
        veh.age = 5
        # Original path: free-flow shortest path on the FULL
        # network, ignoring barred links. The prevailing
        # network carries free-flow times in bare mode (it is
        # never updated, because information_type 1 never sets
        # the info-updating flag), so it is the right graph
        # for both the original path and the barred-avoiding
        # reroutes the loader computes from it.
        node_size, path_links, _ = find_best_path(
            world, sim.prevailing_net, veh.origin_zone,
            veh.origin_node, veh.departure, veh.dest_zone,
            veh.dest_node, veh.demand_type, veh.vot)
        if node_size >= 2 and path_links and \
                all(ln >= 0 for ln in path_links):
            veh.node_size = node_size
            veh.link_ary = [VehicleLink(link_no)
                            for link_no in path_links]
            veh.link_ary.append(VehicleLink(65530))
            distance = f32(0)
            for link_no in path_links:
                distance = f32(distance +
                               world.links[link_no].length)
            veh.distance = distance
            veh.closed_original_links = list(path_links)
            total = 0.0
            for link_no in path_links:
                total += float(world.links[link_no].fftt)
            veh.closed_fftt = total
            veh.remaining_fftt = total
        else:
            # No path even on the full network: the vehicle is
            # registered but never generated, exactly as a
            # pathless agent with node_size < 2 behaves in v2.
            veh.node_size = 0
            veh.link_ary = None
            veh.closed_original_links = []
            veh.closed_fftt = 0.0
            veh.remaining_fftt = 0.0
        veh.time_to_retrieve_info = veh.departure
        world.vehicles[veh.agent_id] = veh
        world.vehicle_order.append(veh.agent_id)
        bucket = int(f32(veh.departure * f32(10)))
        sim.td_map.setdefault(bucket, []).append(veh.agent_id)
    sim.construction = SegmentClosures(world, parsed_closed, now)
    sim.segment_penalty = 0.0
    for _ in range(n_steps):
        current_time = session.step_count * SIM_INTERVAL
        if int(current_time) + 2 > world.horizon:
            session._extend_horizon_to(current_time + 302)
        sim.step(current_time, session.step_count)
        session.step_count += 1
    return {"penalty": float(sim.segment_penalty), "state": session}


# Backwards-incompatible on purpose: v3 has no full-run entry
# point. Importing run_pysim_construction from v2 code against
# this package fails at import time, it is not silently a
# different function.
__all__ = ["create_session", "run_segment", "Session"]
