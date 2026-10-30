"""Agent-id handling, completed-vehicle pruning and renumbering.

Why this module exists. A transition runs many demand profiles from
the same starting session. Profile agents must not collide with
in-system ids and must not collide across profiles in a way that
blocks average_state donor adds, so profile runs use disjoint
offset id ranges. Those offsets cannot be stored: over 240 epochs
they would grow without bound and, more practically, the stored
session's seen_ids would accumulate every profile agent ever
admitted (81 profiles per transition), which is a memory leak
disguised as bookkeeping. The fix is renumbering, not larger
offsets.

v3 agent-id types (checked against the vendored session.py and
loading.py before choosing this scheme): agent ids are plain
Python ints used as dict keys and Vehicle.agent_id. There is no
int32 cast, no numpy int array and no fixed-width id field in
v3, so offsets are safe inside a single transition. They are
still not stored, because the builder controls all future ids
and sequential ids keep every stored session compact.

Scheme, in order, for one transition from a stored session S:
1. base = next_agent_id_base(S) = max id in S + 1 (1 if S is
   empty of ids). Profile p's agents are base + p * stride + j,
   where stride exceeds the largest profile's agent count and
   j enumerates that profile's agents in generation order.
2. The profile end states are averaged. The averaged session
   still carries offset ids.
3. renumber_session(avg, base) rewrites every id >= base to a
   fresh sequential id starting at base, in (departure, old id)
   order for vehicles and in old-id order for pruned completed
   ids. Every reference is rewritten: the vehicles dict keys,
   Vehicle.agent_id, vehicle_order, the waiting dict keys, every
   link queue item's first element, td_map and original_td_map
   values. Ids >= base that averaging dropped (agents admitted
   in a profile run but not placed in the average) are dropped
   from seen_ids as well: they were synthetic profile ids that
   never entered any stored node's history, and future builder
   ids are assigned sequentially from the new base, so they can
   never be reused by accident.
4. prune_session(avg) then drops completed Vehicle objects to an
   id-only set (see below). The stored node's next base is
   next_agent_id_base(stored session), which the next
   transition uses as its base.

Invariants checked by the tests: ids in a stored session are
unique; profile runs never collide with in-system ids; averaged
ids are unique; future demand starts at max + 1; a duplicate of
a retained id fails loud in the next segment.

Pruning. After averaging, before storing a node traffic state,
completed vehicles are dropped to an id-only set:
session.pruned_completed_ids holds their ids, the Vehicle objects
leave world.vehicles and world.vehicle_order, and sim.generated
and sim.completed stay as integer counters. Nothing else moves:
seen_ids keeps every id (so a later segment reusing a completed
id still fails loud in v3's _parse_agents, which checks
seen_ids, not the vehicles dict); td_map keeps its past buckets
(the loader only ever reads the bucket of the current step, and
a completed vehicle's bucket is in the past, so a stale entry is
harmless -- the loader also guards with world.vehicles.get and
skips a missing vehicle); in-system and waiting vehicles are
fully retained. Penalty is unaffected because the v3 segment
penalty is computed live from link traversals and reroute
deltas during run_segment, never from the completed list.
average_state accepts pruned sessions: completed counts come
from the sim.completed counter, completed ids are the union of
full completed vehicles and pruned_completed_ids, and donor
completed vehicles that exist only as pruned ids are carried as
ids, not fabricated back into Vehicle objects.
"""

import copy


def completed_ids_all(session):
    """All completed ids: full completed vehicles plus pruned ids."""
    ids = set(getattr(session, "pruned_completed_ids", set()) or set())
    for aid, veh in session.world.vehicles.items():
        if veh.complete:
            ids.add(aid)
    return ids


def pruned_ids(session):
    return set(getattr(session, "pruned_completed_ids", set()) or set())


def session_max_id(session):
    """Largest id anywhere in the session, or 0 if it has none."""
    best = 0
    for aid in session.seen_ids:
        if aid > best:
            best = aid
    for aid in session.world.vehicles:
        if aid > best:
            best = aid
    for aid in pruned_ids(session):
        if aid > best:
            best = aid
    for aid in session.sim.waiting:
        if aid > best:
            best = aid
    return best


def next_agent_id_base(session):
    """First free agent id for the next transition from session."""
    return session_max_id(session) + 1


def prune_session(session):
    """Drop completed Vehicle objects to an id-only set, in place.

    Returns the number of vehicles pruned. Raises ValueError if
    the session's completed counter disagrees with the completed
    vehicles found, because that means the session is already
    inconsistent and pruning would hide it.
    """
    completed = [aid for aid, veh in session.world.vehicles.items()
                 if veh.complete]
    pruned = pruned_ids(session)
    if len(completed) + len(pruned) != session.sim.completed:
        raise ValueError(
            "prune_session: completed counter %d does not match "
            "%d full + %d pruned completed vehicles"
            % (session.sim.completed, len(completed), len(pruned)))
    if set(completed) & pruned:
        raise ValueError("prune_session: vehicle is both full and pruned")
    for aid in completed:
        del session.world.vehicles[aid]
        if aid in session.world.vehicle_order:
            session.world.vehicle_order.remove(aid)
        pruned.add(aid)
    session.pruned_completed_ids = pruned
    if len(completed_ids_all(session)) != session.sim.completed:
        raise ValueError("prune_session: post-prune completed count "
                         "does not match the counter")
    return len(completed)


def renumber_session(session, base):
    """Renumber ids >= base in session to sequential ids from base.

    Mutates session in place (the builder only calls this on a
    freshly averaged session that is not yet shared or stored).
    Returns (mapping, new_base) where mapping is old -> new for
    every rewritten id and new_base is next_agent_id_base after
    renumbering. Raises ValueError on any inconsistency found
    (duplicate ids, a queue item naming a missing vehicle, a
    waiting key with no vehicle) instead of storing a corrupt
    state.
    """
    if not isinstance(base, int) or isinstance(base, bool) or base < 1:
        raise ValueError("renumber base must be a positive int, got %r"
                         % (base,))
    world = session.world
    sim = session.sim
    mapping = {}
    nxt = base
    new_vehicles = [aid for aid in world.vehicles if aid >= base]
    new_vehicles.sort(key=lambda aid: (float(world.vehicles[aid].departure),
                                       aid))
    for aid in new_vehicles:
        mapping[aid] = nxt
        nxt += 1
    old_pruned = sorted(aid for aid in pruned_ids(session) if aid >= base)
    for aid in old_pruned:
        mapping[aid] = nxt
        nxt += 1
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("renumber_session: mapping is not injective")
    if new_vehicles:
        rebuilt = {}
        for aid, veh in world.vehicles.items():
            if aid in mapping:
                veh.agent_id = mapping[aid]
                rebuilt[mapping[aid]] = veh
            else:
                rebuilt[aid] = veh
        world.vehicles = rebuilt
        world.vehicle_order = [mapping.get(aid, aid)
                               for aid in world.vehicle_order]
        if len(world.vehicle_order) != len(set(world.vehicle_order)):
            raise ValueError("renumber_session: duplicate in "
                             "vehicle_order after renumbering")
    if sim.waiting:
        rebuilt_waiting = {}
        for aid, veh in sim.waiting.items():
            rebuilt_waiting[mapping.get(aid, aid)] = veh
        sim.waiting = rebuilt_waiting
    for link in world.links:
        for queue in (link.loading_buffer, link.entrance_queue,
                      link.exit_queue):
            for item in queue:
                if item[0] in mapping:
                    item[0] = mapping[item[0]]
    for bucket_map in (sim.td_map, world.original_td_map):
        for bucket in list(bucket_map):
            kept = []
            for aid in bucket_map[bucket]:
                if aid >= base:
                    if aid in mapping:
                        kept.append(mapping[aid])
                    # Dropped profile ids: past buckets, dropped.
                else:
                    kept.append(aid)
            if kept:
                bucket_map[bucket] = kept
            else:
                del bucket_map[bucket]
    kept_pruned = set()
    for aid in pruned_ids(session):
        kept_pruned.add(mapping.get(aid, aid))
    session.pruned_completed_ids = kept_pruned
    new_seen = set()
    for aid in session.seen_ids:
        if aid < base:
            new_seen.add(aid)
    new_seen.update(mapping.values())
    session.seen_ids = new_seen
    _validate_renumbered(session)
    return mapping, next_agent_id_base(session)


def offset_session(session, offset):
    """Shift every id in session up by offset, in place.

    Rewrites vehicles dict keys, Vehicle.agent_id, vehicle_order,
    waiting keys, link queue items, td_map and original_td_map
    values, pruned ids and seen_ids. Used by history merging to
    give two histories disjoint id ranges before averaging.
    Raises ValueError on any inconsistency found afterwards.
    """
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise ValueError("offset must be an int, got %r" % (offset,))
    world = session.world
    sim = session.sim
    rebuilt = {}
    for aid, veh in world.vehicles.items():
        veh.agent_id = aid + offset
        rebuilt[aid + offset] = veh
    world.vehicles = rebuilt
    world.vehicle_order = [aid + offset for aid in world.vehicle_order]
    if sim.waiting:
        sim.waiting = {aid + offset: veh
                       for aid, veh in sim.waiting.items()}
    for link in world.links:
        for queue in (link.loading_buffer, link.entrance_queue,
                      link.exit_queue):
            for item in queue:
                item[0] = item[0] + offset
    for bucket_map in (sim.td_map, world.original_td_map):
        for bucket in list(bucket_map):
            bucket_map[bucket] = [aid + offset
                                  for aid in bucket_map[bucket]]
    session.pruned_completed_ids = set(
        aid + offset for aid in pruned_ids(session))
    session.seen_ids = set(aid + offset for aid in session.seen_ids)
    _validate_renumbered(session)
    return session


def _validate_renumbered(session):
    world = session.world
    vehicle_ids = set(world.vehicles)
    if len(vehicle_ids) != len(world.vehicles):
        raise ValueError("renumber_session: duplicate vehicle ids")
    for aid in world.vehicles:
        if world.vehicles[aid].agent_id != aid:
            raise ValueError("renumber_session: vehicle %d has "
                             "agent_id %d" % (aid, world.vehicles[aid].agent_id))
    if not vehicle_ids.issubset(session.seen_ids):
        raise ValueError("renumber_session: vehicle id missing from "
                         "seen_ids after renumbering")
    if vehicle_ids & pruned_ids(session):
        raise ValueError("renumber_session: id is both a vehicle "
                         "and pruned")
    for aid in session.sim.waiting:
        if aid not in vehicle_ids:
            raise ValueError("renumber_session: waiting id %d has "
                             "no vehicle" % aid)
    in_system = set()
    for link in world.links:
        for queue in (link.loading_buffer, link.entrance_queue,
                      link.exit_queue):
            for item in queue:
                if item[0] in in_system:
                    raise ValueError("renumber_session: vehicle %d "
                                     "in two queues" % item[0])
                if item[0] not in vehicle_ids:
                    raise ValueError("renumber_session: queued id %d "
                                     "has no vehicle" % item[0])
                in_system.add(item[0])
