# Time-dependent shortest path on the physical network:
# DTANetworkForSP::BuildPhysicalNetwork + FindBestPathWithVOTForSingleAgent
# (ShortestPath.cpp). TD link times are float32 per assignment interval.
from collections import deque

from .cpp import f32, floating_less

MAX_SPLABEL = f32(9999.0)


class SPNetwork:
    def __init__(self, world, use_prevailing=False, current_time=0.0):
        self.world = world
        n_intervals = world.num_periods
        self.td_time = []
        for link in world.links:
            row = [f32(0.01)] * n_intervals
            self.td_time.append(row)
        # v3 path cache (see find_best_path_cached below). The cache
        # lives on the network object, so it is per session: two
        # sessions never share cached paths, and a deepcopy of a
        # session carries its own copy. _path_cache_enabled is the
        # operator switch (--no-path-cache); _path_cache_ok is
        # None until the td_time uniformity check has run for the
        # current td_time, True when every link's row is uniform
        # (so the shortest path cannot depend on departure time),
        # False otherwise. rebuild() clears the cache and resets
        # the check; update_current_travel_time() disables it,
        # because prevailing times then vary by interval.
        self._path_cache = {}
        self._path_cache_ok = None
        self._path_cache_enabled = True
        self.rebuild(use_prevailing, current_time)

    def rebuild(self, use_prevailing, current_time):
        # BuildPhysicalNetwork TD loop: t walks the assignment intervals of
        # [start, horizon); each visited interval gets the link's current
        # travel time (simulated MOE average, or prevailing time when the
        # network carries current information), times the link-type bias.
        world = self.world
        for link in world.links:
            bias = world.link_type_of(link.link_type).bias
            t = world.start
            while t < world.horizon:
                idx = world.interval_index(t)
                if use_prevailing:
                    avg = link.prevailing_tt
                else:
                    avg = world.travel_time_by_min(
                        link, t, world.interval_length(t))
                avg = f32(avg * bias)
                if avg < f32(0.01):
                    avg = f32(0.01)
                self.td_time[link.link_no][idx] = avg
                t += world.interval_length(t)
        # The TD table changed: cached paths are stale until the
        # uniformity check re-runs for the new table.
        self._path_cache = {}
        self._path_cache_ok = None

    def update_current_travel_time(self, current_time):
        # DTANetworkForSP::UpdateCurrentTravelTime: refresh TD times from
        # CurrentTime forward using each link's prevailing travel time.
        world = self.world
        t = f32(current_time)
        while t < world.horizon:
            idx = world.interval_index(t)
            for link in world.links:
                bias = world.link_type_of(link.link_type).bias
                avg = f32(link.prevailing_tt * bias)
                if avg < f32(0.01):
                    avg = f32(0.01)
                self.td_time[link.link_no][idx] = avg
            t = f32(t + f32(world.interval_length(t)))
        # Intervals before current_time keep old values while later
        # intervals were refreshed, so the table is no longer
        # uniform across intervals: disable the path cache.
        self._path_cache = {}
        self._path_cache_ok = False

    def _verify_path_cache(self):
        # True iff every link's TD row holds a single value, so a
        # shortest path depends only on (origin, destination,
        # barred set) and not on departure time. Runs once per
        # TD table; the result is cached in _path_cache_ok.
        if self._path_cache_ok is not None:
            return self._path_cache_ok
        ok = True
        for row in self.td_time:
            if not row:
                ok = False
                break
            first = row[0]
            for value in row[1:]:
                if value != first:
                    ok = False
                    break
            if not ok:
                break
        self._path_cache_ok = ok
        return ok


def find_best_path(world, net, origin_zone, origin, departure_time,
                   destination_zone, destination, demand_type, vot,
                   barred=None):
    # FindBestPathWithVOTForSingleAgent (ShortestPath.cpp:797), generalized
    # cost flag off. Returns (node_size, path_link_list, total_cost);
    # node_size 0 means no feasible path.
    if vot <= 0.01:
        vot = f32(1)
    if origin == destination:
        return 0, [], f32(0)
    if demand_type == 0:
        demand_type = 1
    departure_time = int(departure_time)
    if departure_time < world.start:
        departure_time = world.start
    if departure_time > world.end:
        departure_time = world.end
    nodes = world.nodes
    if len(nodes[origin].outgoing) == 0:
        return 0, [], f32(0)
    n = len(nodes)
    pred_node = [-1] * n
    pred_link = [-1] * n
    status = [0] * n
    label_time = [MAX_SPLABEL] * n
    label_cost = [MAX_SPLABEL] * n
    label_time[origin] = f32(departure_time)
    label_cost[origin] = f32(departure_time)
    se = deque([origin])
    cost_upper_bound = MAX_SPLABEL
    while se:
        from_id = se.popleft()
        status[from_id] = 2
        for link_no in nodes[from_id].outgoing:
            # Construction version: a barred link is removed from
            # the graph for this path computation, so no reroute
            # can send a vehicle back onto a closed link.
            if barred is not None and link_no in barred:
                continue
            link = world.links[link_no]
            to_id = link.to_id
            if "c" in link.type_code:
                origin_taz = nodes[link.from_id].zone_id
                dest_taz = nodes[link.to_id].zone_id
                if origin_taz >= 1 and dest_taz <= 0 and \
                        origin_taz != origin_zone:
                    continue
                if dest_taz >= 1 and origin_taz <= 0 and \
                        dest_taz != destination_zone:
                    continue
                if origin_taz >= 1 and origin_taz != origin_zone and \
                        dest_taz >= 1 and dest_taz != destination_zone:
                    continue
                if to_id == origin:
                    continue
            interval = world.interval_index(label_time[from_id])
            td = net.td_time[link_no][interval]
            new_time = f32(label_time[from_id] + td)
            new_cost = f32(label_cost[from_id] + td)
            if floating_less(new_cost, label_cost[to_id]) and \
                    new_cost < cost_upper_bound:
                if new_time > world.horizon - 1:
                    new_time = f32(world.horizon - 1)
                label_time[to_id] = new_time
                label_cost[to_id] = new_cost
                pred_node[to_id] = from_id
                pred_link[to_id] = link_no
                if to_id == destination:
                    cost_upper_bound = label_cost[to_id]
                if status[to_id] == 2:
                    se.appendleft(to_id)
                    status[to_id] = 1
                elif status[to_id] == 0:
                    se.append(to_id)
                    status[to_id] = 1
    total_cost = f32(label_cost[destination] - f32(departure_time))
    if total_cost > f32(MAX_SPLABEL - f32(10)):
        return 0, [], total_cost
    # Construction fix: an unreachable destination leaves
    # pred_link[destination] at -1, and the extraction below would
    # otherwise return node_size 2 with a bogus [-1] path (the base
    # threshold is a strict > and 9999 - departure can land exactly
    # on it). Unreachable is no path, full stop.
    if pred_link[destination] == -1:
        return 0, [], total_cost
    # extract the link list by walking predecessors back from the destination
    reversed_links = [pred_link[destination]]
    pred = pred_node[destination]
    while pred != origin and pred != -1 and len(reversed_links) < 3000:
        reversed_links.append(pred_link[pred])
        pred = pred_node[pred]
    path_links = list(reversed(reversed_links))
    return len(path_links) + 1, path_links, total_cost


def find_best_path_cached(world, net, origin_zone, origin, departure_time,
                          destination_zone, destination, demand_type, vot,
                          barred=None):
    """Cached find_best_path, safe by construction.

    In bare mode the TD table is uniform per link (prevailing_tt
    == FFTT for every link and the info-updating flag is False,
    so prevailing times never change mid-run), which makes a
    Dijkstra result depend only on (origin, destination, barred
    set) -- not on departure time. This wrapper caches the path
    (not the cost: the cost is recomputed from the uniform TD
    row on a hit) under that key, but ONLY after verifying the
    table really is uniform: _verify_path_cache scans the rows
    once per TD table, and any rebuild or prevailing-time update
    clears the cache and forces a re-check. If the table is not
    uniform, or the cache is disabled (--no-path-cache), this
    falls back to an uncached find_best_path call, so a future
    info-type change can never silently reuse a stale path.
    The cache lives on the SPNetwork object, so it is per
    session; the key also carries the origin/destination zones
    (connector-link filtering), demand_type and vot, and the
    barred-link frozenset.
    """
    if not getattr(net, "_path_cache_enabled", True):
        return find_best_path(world, net, origin_zone, origin,
                              departure_time, destination_zone,
                              destination, demand_type, vot, barred=barred)
    if not net._verify_path_cache():
        return find_best_path(world, net, origin_zone, origin,
                              departure_time, destination_zone,
                              destination, demand_type, vot, barred=barred)
    if vot <= 0.01:
        vot_key = 1
    else:
        vot_key = float(vot)
    key = (origin, destination, origin_zone, destination_zone,
           demand_type, vot_key,
           frozenset(barred) if barred else frozenset())
    hit = net._path_cache.get(key)
    if hit is not None:
        node_size, path_links = hit
        if node_size < 2 or not path_links:
            return 0, [], f32(0)
        interval = world.interval_index(departure_time)
        total = f32(0)
        for link_no in path_links:
            total = f32(total + net.td_time[link_no][interval])
        return node_size, list(path_links), total
    node_size, path_links, total_cost = find_best_path(
        world, net, origin_zone, origin, departure_time,
        destination_zone, destination, demand_type, vot, barred=barred)
    net._path_cache[key] = (node_size, list(path_links))
    return node_size, path_links, total_cost
