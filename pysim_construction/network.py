# Network/demand data model and the input readers, ported from
# g_ReadInputFiles / g_ReadTripCSVFile in the 2017 DTALite source.
# All C++ float fields are numpy float32; C++ double fields are Python float.
import math
import os
from collections import deque

import numpy as np

from .cpp import (CsvParser, LinkRNG, ProgramStop, VehicleGenerationRng,
                  WellRng, f32, parse_line)

MAX_LINK_NO = 65530  # DTALite.h, non-Windows branch


class LinkType:
    __slots__ = ("link_type", "name", "type_code", "bias", "cap_adj")

    def __init__(self):
        self.link_type = 0
        self.name = ""
        self.type_code = ""
        self.bias = f32(1.0)
        self.cap_adj = f32(1.0)

    def is_freeway(self):
        return "f" in self.type_code

    def is_ramp(self):
        return "r" in self.type_code

    def is_highway(self):
        return "h" in self.type_code

    def is_connector(self):
        return "c" in self.type_code


class Node:
    __slots__ = ("node_id", "number", "x", "y", "zone_id", "outgoing",
                 "incoming")

    def __init__(self):
        self.zone_id = 0
        self.outgoing = []
        self.incoming = []


class Zone:
    __slots__ = ("zone_id", "seq_no", "origin_activity", "dest_activity")

    def __init__(self):
        self.seq_no = 0
        self.origin_activity = []
        self.dest_activity = []


class MinuteMOE:
    __slots__ = ("total_travel_time", "total_flow_count", "exit_queue_length",
                 "cum_arrival", "cum_departure")

    def __init__(self):
        self.total_travel_time = f32(0)
        self.total_flow_count = 0
        self.exit_queue_length = 0
        self.cum_arrival = 0
        self.cum_departure = 0


class Link:
    __slots__ = (
        "link_no", "from_id", "to_id", "length", "speed_limit", "lane_capacity",
        "k_jam", "inflow_lanes", "outflow_lanes", "link_type", "type_code",
        "fftt", "wave_speed", "backward_wave_steps", "space_capacity",
        "merge_flag", "on_ramp", "merge_incoming",
        "merge_remaining", "rng", "toll_size",
        # simulation state
        "loading_buffer", "entrance_queue", "exit_queue",
        "cflow_arrival", "cflow_departure",
        "link_in_capacity", "link_out_capacity",
        "cum_out_counter", "cum_in_counter", "cum_merge_counter",
        "cum_out_prev", "cum_in_prev",
        "counts_demand", "counts_vtype",
        "ring_arrival", "ring_departure", "moe",
        "departure_based_tt", "departure_count", "prevailing_tt",
        "buffer_waiting_time", "jam_time_stamp", "total_energy",
        "org_geometry")

    def __init__(self):
        self.loading_buffer = deque()
        self.entrance_queue = deque()
        self.exit_queue = deque()
        self.cflow_arrival = 0
        self.cflow_departure = 0
        self.link_in_capacity = 0
        self.link_out_capacity = 0
        self.cum_out_counter = f32(0)
        self.cum_in_counter = f32(0)
        self.cum_merge_counter = f32(0)
        self.cum_out_prev = 0
        self.cum_in_prev = 0
        self.counts_demand = [0] * 40
        self.counts_vtype = [0] * 15
        self.departure_based_tt = f32(0)
        self.departure_count = 0
        self.prevailing_tt = f32(0)
        self.buffer_waiting_time = f32(0)
        self.jam_time_stamp = f32(0)
        self.total_energy = f32(0)
        self.toll_size = 0
        self.org_geometry = ""

    def setup_moe(self, horizon):
        # DTALink::SetupMOE (DTALite.h:2203): FFTT is (re)computed here, at
        # network read and again at the start of every loading. The
        # departure-based travel time total restarts at the FFTT.
        self.fftt = f32(self.length / self.speed_limit * f32(60.0))
        self.prevailing_tt = self.fftt
        self.departure_based_tt = self.fftt
        self.jam_time_stamp = f32(horizon)
        # int(m_Length / m_BackwardWaveSpeed * 60 / g_DTASimulationInterval)
        # with float arithmetic until the final division by the double 0.1
        self.backward_wave_steps = int(
            float(f32(f32(self.length / self.wave_speed) * f32(60))) / 0.1)
        # Drop the previous list before building the new one: the old MOE
        # values are dead at rebuild, and holding both lists at once
        # doubles peak memory (one list per link per minute).
        self.moe = None
        self.moe = [MinuteMOE() for _ in range(horizon + 1)]
        self.ring_arrival = [f32(0)] * 301
        self.ring_departure = [f32(0)] * 301
        self.loading_buffer.clear()
        self.entrance_queue.clear()
        self.exit_queue.clear()
        self.cflow_arrival = 0
        self.cflow_departure = 0
        self.cum_out_counter = f32(0)
        self.cum_in_counter = f32(0)
        self.cum_merge_counter = f32(0)
        self.cum_out_prev = 0
        self.cum_in_prev = 0
        self.counts_demand = [0] * 40
        self.counts_vtype = [0] * 15
        self.departure_count = 0
        self.buffer_waiting_time = f32(0)


class VehicleLink:
    __slots__ = ("link_no", "abs_arrival")

    def __init__(self, link_no):
        self.link_no = link_no
        self.abs_arrival = f32(99999)


class Vehicle:
    __slots__ = (
        "agent_id", "tour_id", "dependency_agent_id", "following_agent_id",
        "duration",
        "origin_zone", "dest_zone", "origin_node", "dest_node",
        "departure", "arrival", "trip_time", "travel_time",
        "demand_type", "vehicle_type", "pce", "info_type", "vot", "age",
        "toll", "distance", "trip_fftt", "delay", "node_size", "link_ary",
        "original_link_ary", "original_node_size", "loaded", "complete",
        "sim_link_seq", "leaving_buffer_time", "time_to_retrieve_info", "co",
        "em_energy", "em_co2", "em_nox", "em_co", "em_hc", "em_pm",
        "em_pm25",
        # Construction version only: the pre-closure original path
        # (link numbers, sentinel excluded) and its free-flow total,
        # captured after iteration-0 assignment and before loading,
        # plus closure-wait bookkeeping for origin waiting.
        "closed_original_links", "closed_fftt",
        "construction_wait_since", "construction_waited",
        # Stepwise v3 only: free-flow time of the links not yet
        # traversed from the vehicle's current node onward. The
        # session penalty bookkeeping updates it on every reroute
        # and every completed link traversal, so reroute deltas
        # telescope against the original-path FFTT (see session.py).
        "remaining_fftt")

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.tour_id = -1
        self.dependency_agent_id = -1
        self.following_agent_id = -1
        self.duration = f32(-1)
        self.origin_zone = 0
        self.dest_zone = 0
        self.origin_node = 0
        self.dest_node = 0
        self.departure = f32(0)
        self.arrival = f32(0)
        self.trip_time = f32(0)
        self.travel_time = f32(0)
        self.demand_type = 1
        self.vehicle_type = 1
        self.pce = f32(1)
        self.info_type = 1
        self.vot = f32(10)
        self.age = 5
        self.toll = f32(0)
        self.distance = f32(0)
        self.trip_fftt = f32(0)
        self.delay = f32(0)
        self.node_size = 0
        self.link_ary = None
        self.original_link_ary = None
        self.original_node_size = 0
        self.co = f32(0)
        self.em_energy = 0.0
        self.em_co2 = 0.0
        self.em_nox = 0.0
        self.em_co = 0.0
        self.em_hc = 0.0
        self.em_pm = 0.0
        self.em_pm25 = 0.0
        self.loaded = False
        self.complete = False
        self.closed_original_links = []
        self.closed_fftt = f32(0)
        self.remaining_fftt = 0.0
        self.construction_wait_since = None
        self.construction_waited = False
        self.sim_link_seq = 0
        self.leaving_buffer_time = f32(0)
        self.time_to_retrieve_info = f32(0)


class DemandType:
    __slots__ = ("avg_vot", "cum_info_pct")

    def __init__(self):
        self.avg_vot = f32(10)
        # cumulative percentages for info classes 0..3 (DTALite.cpp:2030-2070)
        self.cum_info_pct = [0.0, 0.0, 0.0, 0.0]


class World:
    def __init__(self, input_dir, flow_model=2, random_seed=100,
                 session_mode=False):
        if flow_model not in (1, 2, 3):
            raise ProgramStop(
                "pysim implements the queue-based flow models 1/2/3; "
                "traffic_flow_model=%s is out of scope" % (flow_model,))
        self.dir = input_dir
        # Bare v2 settings, hardcoded: there is no DTASettings.txt
        # and no input_scenario_settings.csv. signal_model is 0
        # (no signals); the point-queue flags are dead because no
        # link is a ramp (every link is hardcoded type "f"), and
        # the info-updating interval is dead because every agent
        # has information_type 1, so info_updating_flag stays False.
        self.flow_model = flow_model
        self.random_seed = random_seed
        self.signal_model = 0
        self.num_iterations = 0  # a single loading pass, always
        self.aggregation_interval = 15
        self.agent_based_flag = 1
        self.demand_scaling = f32(1.0)
        self.use_pq_on_ramps = 1
        self.use_pq_off_ramps = 1
        self.info_updating_interval = 1
        self.emission_mode = "off"
        self.emissions_computed = False
        self.full_link_moe = None
        self.full_link_tdmoe = None
        # Simulation window. In the v2 full run it comes from the
        # agent file itself: the earliest departure to the latest
        # departure, horizon 300 minutes later (the v1 construction
        # horizon rule). In stepwise v3 mode there is no agent file
        # and no known end: the clock starts at 0.0 and the horizon
        # starts at 300 minutes, then grows dynamically as segments
        # advance (see extend_horizon). end tracks the horizon in
        # session mode, so the loader's post-end capacity relaxation
        # and the shortest-path departure clamp never trigger early.
        self.start = 0
        self.end = 1440
        self.horizon = 1740
        self.session_mode = session_mode
        if session_mode:
            self.start = 0
            self.end = 300
            self.horizon = 300
        else:
            self._determine_demand_period()
            self.horizon = self.end + 300
        # assignment period grid (g_ReadAssignmentPeriodSettings)
        self._build_assignment_periods()
        # network containers
        self.nodes = []
        self.node_index = {}       # node number -> internal id
        self.links = []
        self.link_map = {}         # (from number, to number) -> link index
        self.link_types = {}       # link_type -> LinkType (bare: only 1)
        self.zones = {}            # zone_id -> Zone (bare: zone = node)
        self.od_zone_size = 0      # max zone id
        self.demand_types = [DemandType()]  # bare: only type 1, unused
        # vehicles: g_VehicleMap (sorted by agent_id) + creation order list
        self.vehicles = {}
        self.vehicle_order = []
        # departure buckets: g_OriginalVehicleTDListMap[int(departure*10)]
        self.original_td_map = {}
        self.max_density_ratio = f32(0.8)   # hard-coded in g_NetworkLoading
        self.min_inflow_ratio = f32(0)      # g_MinimumInFlowRatio default
        self.well_rng = None       # created at read_inputs (exe order)
        self.gen_rng = VehicleGenerationRng()
        self.info_updating_flag = False

    def _path(self, name):
        return os.path.join(self.dir, name)

    def _determine_demand_period(self):
        # Bare v2: the window comes from input_agent.csv, not from
        # a demand file list. start is the floor of the earliest
        # departure, end the ceiling of the latest; a single
        # departure still gets a one-minute window so the agent
        # itself is inside it. The 1,000,000-minute cap and the
        # negative-start rule are carried over from v1.
        path = self._path("input_agent.csv")
        if not os.path.exists(path):
            raise ProgramStop(
                "input_agent.csv cannot be opened in %s" % self.dir)
        parser = CsvParser(path)
        earliest = None
        latest = None
        while parser.read_record():
            ok, departure = parser.get("departure_time_in_min", "double")
            if not ok:
                raise ProgramStop(
                    "departure_time_in_min missing in input_agent.csv")
            if departure < 0:
                raise ProgramStop(
                    "agent departure %s is negative" % (departure,))
            earliest = departure if earliest is None else min(earliest, departure)
            latest = departure if latest is None else max(latest, departure)
        if earliest is None:
            raise ProgramStop("no agents in input_agent.csv")
        start = int(math.floor(float(earliest)))
        end = int(math.ceil(float(latest)))
        if end > 1000000:
            raise ProgramStop("demand loading period out of range")
        if end <= start:
            end = start + 1
        self.start, self.end = start, end

    def _build_assignment_periods(self):
        # g_ReadAssignmentPeriodSettings (DTALite.cpp:5900): the
        # uniform grid is built first, then unconditionally
        # overwritten by the absolute 15-minute grid; the last
        # interval closes at the loading end time. The grid is
        # sized by the horizon instead of a fixed 300 slots, so
        # departures beyond 4485 min no longer index past the end
        # of the grid (v1 fix, kept here).
        max_slot = self.horizon // 15 + 1
        size = max_slot + 1
        index = [0] * size
        starts = [0] * size
        ends = [0] * size
        if self.start // 15 < size:
            index[self.start // 15] = 0
        starts[0] = self.start
        counter = 0
        for i in range(size):
            if i * 15 >= self.start and i * 15 < self.horizon:
                if i * 15 != self.start:
                    counter += 1
                    ends[counter - 1] = i * 15
                    starts[counter] = i * 15
                index[i] = counter
        ends[counter] = self.end
        self.assign_index = index
        self.assign_start = starts
        self.assign_end = ends
        self.num_periods = counter + 1

    def interval_index(self, time_in_min):
        # g_FindAssignmentIntervalIndexFromTime (ShortestPath.cpp:37)
        if time_in_min < self.start:
            return 0
        if time_in_min < self.horizon:
            slot = max(0, int(f32(time_in_min) + 0.1) // 15)
            value = self.assign_index[slot]
        else:
            value = max(0, self.num_periods - 1)
        return min(value, self.num_periods - 1)

    def interval_length(self, time_in_min):
        # g_FindAssignmentIntervalLengthInMinFromTime: max(15, end - start)
        idx = self.interval_index(time_in_min)
        return max(15, int(self.assign_end[idx] - self.assign_start[idx]))

    def travel_time_by_min(self, link, start_time, interval):
        # DTALink::GetTravelTimeByMin (DTALite.h:2473), non-BPR branch:
        # windowed MOE average floored at the free-flow travel time.
        start_time = max(0, int(start_time))
        time_end = min(start_time + interval, self.horizon)
        total_tt = f32(0)
        total_flow = 0
        for t in range(start_time, time_end):
            total_tt = f32(total_tt + link.moe[t].total_travel_time)
            total_flow += link.moe[t].total_flow_count
        if total_flow >= 1:
            travel_time = f32(total_tt / f32(total_flow))
            if travel_time < link.fftt:
                travel_time = link.fftt
            return travel_time
        return link.fftt

    def link_speed(self, link, time, aggregation=1):
        # DTALink::GetSpeed: m_Length / max(0.1f, travel time) * 60
        tt = self.travel_time_by_min(link, time, aggregation)
        return f32(f32(link.length / max(f32(0.1), tt)) * f32(60.0))

    def arrival_flow(self, link, time, aggregation=1):
        # DTALink::GetArrivalFlow (DTALite.h:2396)
        time = max(0, time)
        if time + aggregation < self.horizon - 1:
            return max(0, link.moe[time + aggregation].cum_arrival -
                       link.moe[time].cum_arrival)
        return 0

    def departure_flow(self, link, time, aggregation=1):
        time = max(0, time)
        if time + aggregation < self.horizon - 1:
            return max(0, link.moe[time + aggregation].cum_departure -
                       link.moe[time].cum_departure)
        return 0

    # ----- input readers (bare v2) -----

    def read_inputs(self, load_agents=True):
        self.well_rng = WellRng(self.random_seed)
        self._read_nodes()
        self._read_links()
        self._setup_merges()
        self._generate_zones()
        if load_agents:
            self._read_agents()

    def extend_horizon(self, new_horizon):
        # Grow the horizon in place, preserving everything already
        # simulated. Only appends: each link's MOE list gains one
        # fresh MinuteMOE per added minute, and the assignment
        # grid is rebuilt from the same start, which preserves
        # every interval index for times below the old horizon
        # (the grid is a deterministic 15-minute walk from start,
        # so extending it only appends new intervals after the
        # old ones; the old tail entries, placeholder zeros for
        # slots past the old horizon, become real interval
        # numbers, which is the point of the extension). Never call
        # Link.setup_moe here: it would wipe the queues, counters
        # and MOE minutes already written. The caller (session.py)
        # must also extend any SPNetwork td_time rows, which are
        # sized by num_periods; this method updates num_periods
        # first so the caller can see the new size.
        if new_horizon <= self.horizon:
            return
        old_len = len(self.links[0].moe) if self.links else 0
        self.horizon = new_horizon
        self.end = new_horizon
        self._build_assignment_periods()
        extra = new_horizon + 1 - old_len
        for link in self.links:
            link.moe.extend(MinuteMOE() for _ in range(extra))
            link.jam_time_stamp = f32(new_horizon)

    def _setup_merges(self):
        # merge identification over the incoming links at each link's
        # from-node (DTALite.cpp:4495-4625). The in-capacity ratio lives on
        # the (merge link, incoming link) entry, from OUTFLOW lane counts.
        for link in self.links:
            node = self.nodes[link.from_id]
            freeway_ramp_count = 0
            no_arterial = True
            for in_no in node.incoming:
                in_link = self.links[in_no]
                if in_link.from_id == link.to_id:
                    continue  # U-turn link does not count
                if "f" in in_link.type_code or "r" in in_link.type_code:
                    freeway_ramp_count += 1
                else:
                    no_arterial = False
            if freeway_ramp_count >= 2 and no_arterial and \
                    len(node.outgoing) == 1:
                link.merge_flag = 1
        for link in self.links:
            if link.merge_flag != 1:
                continue
            incoming = self.nodes[link.from_id].incoming
            if len(incoming) == 2:
                ramp_seen = False
                freeway_seen = False
                for in_no in incoming:
                    in_link = self.links[in_no]
                    if "r" in in_link.type_code:
                        ramp_seen = True
                        in_link.on_ramp = True
                    if "f" in in_link.type_code or "h" in in_link.type_code:
                        freeway_seen = True
                    if ramp_seen and freeway_seen:
                        link.merge_flag = 2
                        break
            total_lanes = sum(self.links[n].outflow_lanes for n in incoming)
            link.merge_incoming = [
                [in_no, f32(f32(self.links[in_no].outflow_lanes) /
                            f32(total_lanes)), f32(0)]
                for in_no in incoming]
        # point-queue overwrite of ramp storage capacities (end of link read)
        for link in self.links:
            if "r" not in link.type_code:
                continue
            if link.on_ramp:
                if self.use_pq_on_ramps == 1:
                    link.space_capacity = f32(99999)
            else:
                if self.use_pq_off_ramps == 1:
                    link.space_capacity = f32(99999)

    def _generate_zones(self):
        # Bare v2: zone = node. No zone file is read; the zone
        # dict is built from the nodes in file order, and a node
        # is an origin activity location when it has an outgoing
        # link and a destination activity location when it has an
        # incoming link (the exe's fallback when no activity
        # location file exists).
        for i, node in enumerate(self.nodes):
            zone = Zone()
            zone.zone_id = node.number
            zone.seq_no = i
            node.zone_id = node.number
            if len(node.outgoing) >= 1:
                zone.origin_activity.append(i)
            if len(node.incoming) >= 1:
                zone.dest_activity.append(i)
            self.zones[node.number] = zone
            self.od_zone_size = max(self.od_zone_size, node.number)

    def _read_nodes(self):
        if not os.path.exists(self._path("input_node.csv")):
            raise ProgramStop(
                "input_node.csv cannot be opened in %s" % self.dir)
        # Only node_id is required. x/y are not read at all: no
        # output in this package uses coordinates, so a bare file
        # without them and an old file with them behave the same.
        parser = CsvParser(self._path("input_node.csv"))
        i = 0
        while parser.read_record():
            ok, node_id = parser.get("node_id", "int", required=True)
            if not ok:
                break
            if node_id in self.node_index:
                raise ProgramStop(
                    "duplicate node_id %d in input_node.csv" % node_id)
            node = Node()
            node.node_id = i
            node.number = node_id
            self.nodes.append(node)
            self.node_index[node_id] = i
            i += 1
        if not self.nodes:
            raise ProgramStop("no nodes in input_node.csv")

    def link_type_of(self, link_type):
        # Bare v2: only type 1 exists (set in _read_links); anything
        # else default-constructs, as g_LinkTypeMap[type] did.
        element = self.link_types.get(link_type)
        if element is None:
            element = LinkType()
            element.link_type = link_type
            self.link_types[link_type] = element
        return element

    def _read_links(self):
        # Bare v2 link schema. direction is ignored (one row is
        # always exactly one directed link, from -> to as written;
        # no reverse link is created and no link is reversed).
        # link_type is ignored and hardcoded to 1 / "f" (Freeway,
        # bias 1.0, capacity adjustment 1.0): Sioux Falls is 76/76
        # Freeway, so this preserves its merge behaviour, and the
        # merge test in _setup_merges never triggers there. mode,
        # geometry, link_type_name and any other extra column are
        # ignored, never rejected. Every row is a vehicle link.
        freeway = LinkType()
        freeway.link_type = 1
        freeway.name = "Freeway"
        freeway.type_code = "f"
        freeway.bias = f32(1.0)
        freeway.cap_adj = f32(1.0)
        self.link_types[1] = freeway
        if not os.path.exists(self._path("input_link.csv")):
            raise ProgramStop(
                "input_link.csv cannot be opened in %s" % self.dir)
        parser = CsvParser(self._path("input_link.csv"))
        while parser.read_record():
            ok, from_name = parser.get("from_node_id", "int", required=True)
            if not ok:
                raise ProgramStop("from_node_id missing in input_link.csv")
            ok, to_name = parser.get("to_node_id", "int", required=True)
            if not ok:
                raise ProgramStop("to_node_id missing in input_link.csv")
            if from_name not in self.node_index:
                raise ProgramStop(
                    "input_link.csv from_node_id %d not in input_node.csv"
                    % from_name)
            if to_name not in self.node_index:
                raise ProgramStop(
                    "input_link.csv to_node_id %d not in input_node.csv"
                    % to_name)
            if (from_name, to_name) in self.link_map:
                raise ProgramStop(
                    "duplicate link %d->%d in input_link.csv"
                    % (from_name, to_name))
            ok, length = parser.get("length", "double")
            if not ok:
                ok, length = parser.get("length_", "double")
            if not ok:
                length = 1.0
            ok, lanes = parser.get("number_of_lanes", "int")
            if not ok:
                lanes = 1
            ok, speed = parser.get("speed_limit", "int")
            if not ok:
                speed = 0
            ok, capacity = parser.get("lane_capacity_in_vhc_per_hour",
                                      "double")
            if not ok:
                ok, capacity = parser.get("lane_cap", "double")
            if not ok:
                capacity = 0.0
            ok, k_jam = parser.get("jam_density", "double")
            if not ok:
                ok, k_jam = parser.get("jam_density_in_vhc_pmpl", "double")
            if not ok:
                k_jam = 180.0
            ok, wave = parser.get("wave_speed", "double")
            if not ok:
                ok, wave = parser.get("wave_speed_in_mph", "double")
            if not ok:
                wave = 12.0
            if lanes == 0 or capacity < 1 or speed < 1:
                raise ProgramStop(
                    "bad link %d->%d in input_link.csv: lanes=%s, "
                    "capacity=%s, speed=%s"
                    % (from_name, to_name, lanes, capacity, speed))
            self._add_link(from_name, to_name, length, lanes, speed,
                           capacity, k_jam, wave, 1, "")

    def _add_link(self, from_name, to_name, length, lanes, speed, capacity,
                  k_jam, wave, link_type, geometry):
        link = Link()
        link.link_no = len(self.links)
        link.from_id = self.node_index[from_name]
        link.to_id = self.node_index[to_name]
        speed_f = f32(speed)
        length_f = f32(max(0.00001, length))
        if self.flow_model != 0:  # not BPR: minimum length from speed
            length_f = f32(max(length_f, f32(f32(speed_f * f32(0.1)) /
                                             f32(60.0))))
        link.length = length_f
        link.speed_limit = speed_f
        element = self.link_type_of(link_type)
        link.lane_capacity = f32(f32(capacity) * element.cap_adj)
        link.k_jam = f32(k_jam)
        link.wave_speed = f32(wave)
        link.inflow_lanes = lanes
        link.outflow_lanes = lanes
        link.link_type = link_type
        link.type_code = element.type_code
        link.merge_flag = 0
        link.on_ramp = False
        link.merge_incoming = []
        link.merge_remaining = f32(0)
        link.rng = LinkRNG(link.link_no)
        link.org_geometry = geometry
        # m_VehicleSpaceCapacity = max(1, m_Length * outflow lanes * K_jam)
        # evaluated in double (K_jam is the double local), stored as float
        # (DTALite.cpp:1455)
        link.space_capacity = f32(max(1.0, float(link.length) * lanes *
                                      k_jam))
        # DTALink::SetupMOE at read time; repeated at loading start
        link.setup_moe(self.horizon)
        self.links.append(link)
        self.nodes[link.from_id].outgoing.append(link.link_no)
        self.nodes[link.to_id].incoming.append(link.link_no)
        self.link_map[(from_name, to_name)] = link.link_no

    def _read_agents(self):
        path = self._path("input_agent.csv")
        if not os.path.exists(path):
            raise ProgramStop(
                "input_agent.csv cannot be opened in %s" % self.dir)
        parser = CsvParser(path)
        line_no = 0
        while parser.read_record():
            if len(parser.fields) == 0:
                break
            line_no += 1
            self._read_agent_record(parser, line_no)

    def _read_agent_record(self, parser, line_no):
        # Bare v2 agent schema: exactly input_agent.csv, pathless
        # agents only, zone = node. from_zone_id / to_zone_id are
        # accepted and ignored when present.
        ok, agent_id = parser.get("agent_id", "int", required=True)
        if not ok:
            raise ProgramStop("agent_id missing in input_agent.csv")
        if agent_id in self.vehicles:
            raise ProgramStop(
                "duplicate agent_id %d in input_agent.csv" % agent_id)
        ok, from_node = parser.get("origin_node_id", "int")
        if not ok:
            ok, from_node = parser.get("from_origin_node_id", "int")
        if not ok:
            raise ProgramStop(
                "agent %d: origin node missing in input_agent.csv "
                "(origin_node_id or from_origin_node_id)" % agent_id)
        ok, to_node = parser.get("destination_node_id", "int")
        if not ok:
            ok, to_node = parser.get("to_destination_node_id", "int")
        if not ok:
            raise ProgramStop(
                "agent %d: destination node missing in input_agent.csv "
                "(destination_node_id or to_destination_node_id)"
                % agent_id)
        # Zone columns, if present, are ignored: zone = node.
        parser.get("from_zone_id", "int")
        parser.get("to_zone_id", "int")
        if from_node not in self.node_index:
            raise ProgramStop(
                "agent %d origin node %d not in input_node.csv"
                % (agent_id, from_node))
        if to_node not in self.node_index:
            raise ProgramStop(
                "agent %d destination node %d not in input_node.csv"
                % (agent_id, to_node))
        ok, departure = parser.get("departure_time_in_min", "double")
        if not ok:
            raise ProgramStop(
                "agent %d departure_time_in_min missing" % agent_id)
        if departure < 0:
            raise ProgramStop(
                "agent %d departure %s is negative"
                % (agent_id, float(departure)))
        if departure < self.start or departure > self.end:
            raise ProgramStop(
                "agent %d departure %s outside loading window"
                % (agent_id, float(departure)))
        ok, path_text = parser.get("path_node_sequence", "str")
        if ok:
            raise ProgramStop(
                "bare v2 agents must be pathless: agent %d has a "
                "path_node_sequence" % agent_id)
        if from_node == to_node:
            return  # intrazonal: silently skipped, as in v1
        veh = Vehicle(agent_id)
        veh.origin_zone = from_node
        veh.dest_zone = to_node
        veh.origin_node = self.node_index[from_node]
        veh.dest_node = self.node_index[to_node]
        veh.departure = departure
        # Hardcoded per agent (v1 defaults): no RNG draw is made
        # for information class or age in bare v2, so random_seed
        # has no effect on results. info_type 1 means
        # info_updating_flag stays False and every reroute is a
        # free-flow shortest path excluding barred links.
        veh.demand_type = 1
        veh.vehicle_type = 1
        veh.pce = f32(1)
        veh.info_type = 1
        veh.vot = f32(10)
        veh.age = 5
        self._register(veh, True)

    def _register(self, veh, is_new):
        # create-branch registration: g_VehicleMap insert plus the
        # g_OriginalVehicleTDListMap bucket (int)(departure * 10)
        if is_new:
            veh.time_to_retrieve_info = veh.departure
            self.vehicles[veh.agent_id] = veh
            self.vehicle_order.append(veh.agent_id)
            key = int(f32(veh.departure * f32(10)))
            self.original_td_map.setdefault(key, []).append(veh.agent_id)

