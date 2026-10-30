# pysim_construction — stepwise, pause/resume PySim with construction closures (DDM only), version 3

A specialised copy of the standalone PySim loading pass for
Eli's Dynamic Decision Models project. It replaces bare v2 for
DDM use only. It is NOT a general DTALite replacement, and it
must not be overlaid onto the standalone `pysim` tree.

## Breaking changes vs v2 — read this first

Version 3 is NOT drop-in compatible with version 2, even
though the folder name (`pysim_construction`) is the same.
The v2 zip on disk is untouched; this is its successor, not a
patch over it. Extract it into a clean directory, never over
an old v2 folder, or the deleted v2 files (`api.py`, `run.py`,
`construction.py`, `construction_outputs.py`) will be left
behind as orphans.

- There is no full-run function. `run_pysim_construction`
  does not exist in v3. The run is driven segment by segment:
  `create_session` once, then `run_segment` per segment.
- There is no `construction_projects.csv`. Closures are a
  per-segment dict passed to `run_segment` (below). There is
  no project, no scheduled start, and no clear-first rule:
  the countdown starts at the segment start regardless of
  cars on the link (Eli's approved simplification, replacing
  v2's barred-then-clear-then-construct state machine).
- There are no output files at all. `run_segment` returns
  exactly `{"penalty": float, "state": session}` and writes
  nothing. No completed-vehicle list, no in-system summary in
  the return. Inspect the session object (see below) if you
  need positions or counts.
- There is no `input_agent.csv` and no simulation window from
  an agent file. Agents are passed per segment as a list of
  dicts (or a CSV path), with absolute departure times.
- The segment penalty replaces the v2 excess outputs. It is
  computed live during the run (below), not from a post-pass
  over completed vehicles, so incomplete vehicles contribute
  their traversed links and reroute deltas so far.

## Quick start

```python
import sys
sys.path.insert(0, "/path/to/this/folder/..")  # parent of pysim_construction/

from pysim_construction import create_session, run_segment

session = create_session("example_network")
r1 = run_segment(session, [
    {"agent_id": 1, "origin_node_id": 1,
     "destination_node_id": 2, "departure_time_in_min": 5.0},
], {(1, 2): 20.0}, 60.0)
print(r1["penalty"])  # segment penalty, vehicle-minutes
print(sorted(r1.keys()))  # ["penalty", "state"] — nothing else
```

A runnable loop is included in the zip as
`example_step_loop.py` (top level, next to
`pysim_construction/`), using the tiny `example_network/`
folder also at the top level. Run it from the extraction
directory with `python example_step_loop.py`.

## Signature

```python
def create_session(network_dir, flow_model=2, random_seed=100):
def run_segment(session, agents, closed_links, step_minutes):
```

- `network_dir`: exactly two bare input files (below). Read,
  never written.
- `flow_model`: 1, 2 or 3 (default 2, Newell). Anything else
  fails loud.
- `random_seed`: default 100, but it has no effect on results
  in bare mode: no RNG draw is made for information class or
  age, because every agent is hardcoded to information_type 1,
  vehicle_type 1 and age 5. Stated honestly, as in v2.
- `agents`: a list of dicts, or a CSV file path with the same
  columns. Each agent: `agent_id`, `origin_node_id`,
  `destination_node_id`, `departure_time_in_min` (ABSOLUTE
  cumulative minutes, NOT relative to the segment start).
  Aliases `from_origin_node_id` / `to_destination_node_id`
  are accepted. Agents must be pathless: a non-empty
  `path_node_sequence` fails loud.
- `closed_links`: a dict `{(from_node_id, to_node_id):
  minutes_left}`. Absent or 0 means open. `None` means no
  closures (a drain segment). See "Closure semantics" below.
- `step_minutes`: variable per call, must be > 0 and a
  multiple of 0.1 within 1e-9, otherwise fails loud.
  Simulation step is 0.1 minutes (6 seconds), as in v2.
- Returns exactly `{"penalty": float, "state": session}`.
  `state` is the same session object passed in, mutated.

## Bare network schema — exactly two files

### 1. input_node.csv

```
node_id
1
2
```

Only `node_id` is required. `x` / `y`, if present, are
ignored: no output in this package uses coordinates. A
missing `node_id` column fails loud; a duplicate `node_id`
fails loud.

### 2. input_link.csv

```
from_node_id,to_node_id,length,number_of_lanes,speed_limit,lane_capacity_in_vhc_per_hour,jam_density,wave_speed
1,2,5,2,60,1800,180,12
```

Required: `from_node_id`, `to_node_id`, `length`,
`number_of_lanes`, `speed_limit`,
`lane_capacity_in_vhc_per_hour` (`lane_cap` is accepted as an
alias). Optional, with defaults: `jam_density` (180),
`wave_speed` (12); the aliases `jam_density_in_vhc_pmpl` and
`wave_speed_in_mph` are also accepted. A missing `length`
defaults to 1.0, as in v2.

Dropped columns, all ignored if present, never rejected:

- `direction`: one row is always exactly one directed link,
  from -> to as written. No reverse link is created and no
  link is reversed.
- `link_type`: hardcoded to 1, type code `"f"` (Freeway),
  bias 1.0, capacity adjustment 1.0.
- `mode_code`, `geometry`, `link_type_name` and every other
  extra column: ignored. Every row is a vehicle link.

Fail loud on: missing from/to node, a from/to node that is
not in the node file, a duplicate directed link,
`number_of_lanes` 0, capacity below 1, or speed below 1.

There is no DTASettings.txt, no scenario/zone/link-type/
demand file, no signal model (hardcoded 0), no emissions, no
ramps (every link is type `"f"`), no vehicle or demand types,
and no zones file: zone = node, generated internally.

## The session is the resume state

`create_session` reads the network once. The session object
carries everything a new segment needs to pick up exactly
where the old one left off, in memory:

- all vehicles (link sequence, arrival stamps,
  leaving-buffer time, original path and FFTT, remaining-path
  FFTT, loaded/complete flags, waiting status);
- per-link loading, entrance and exit queues in order, flow
  arrival/departure counters, cumulative counters, ring
  buffers, prevailing and departure-based travel times, and
  per-link MOE;
- the simulator's origin-waiting dict, generated/completed
  counts, departure buckets, and prevailing-time shortest
  path network;
- the clock, as a step count (`now = step_count * 0.1`
  minutes, never accumulated, so boundaries never drift).

Construction state is NOT carried in the session. Each
segment's `closed_links` dict is the authority for that
segment; the caller (the outside tracker) subtracts
`step_minutes` from each running project's remaining time
and passes the fresh dict next segment. PySim does not
report when a project finishes: finish time = segment start
+ minutes_left, computed by the caller.

Inspection helpers (not part of the return value):
`session.now`, `session.completed_ids()`,
`session.waiting_ids()`, `session.vehicle_positions()`
(agent_id -> (from_node_id, to_node_id) of the link the
vehicle is currently on, or None if waiting, not yet loaded,
or complete). The session is an in-memory object; it is not
a serialised snapshot format, and pickling is untested — do
not rely on it.

## Segment windows and fail-loud rules

- The clock starts at 0.0, even if the first segment is
  empty (no agents, no closures).
- A segment covers the half-open departure window
  `[now, now+step_minutes)`. A departure exactly at
  `now+step_minutes` belongs to the next segment and fails
  loud in this one. A departure before `now` fails loud.
  The 1e-9 margins in the check absorb float representation
  error only (for example `now` computed as `3 * 0.1` is
  `0.30000000000000004` while a caller passes departure
  `0.3`); they never admit a departure a real step early
  or late.
- A duplicate `agent_id` in ANY segment fails loud. Agent
  ids are unique for the life of the session.
- An unknown origin/destination node, an unknown closed
  link, a negative `minutes_left`, a non-numeric
  `minutes_left`, `step_minutes <= 0`, or `step_minutes`
  not a multiple of 0.1 fails loud.
- A non-empty `path_node_sequence` fails loud: bare v3
  agents are pathless, as in v2.
- A departure below 0 fails loud.
- Intrazonal agents (origin == destination) are silently
  skipped, as in v2: not added to the simulated state, not
  counted. Their ids are still recorded, so a repeat id
  fails loud (this is stricter than v2's file reader, where
  a skipped intrazonal id was never registered; the session
  rule is the documented one).
- An agent with no path even on the full network (origin
  disconnected from destination) is registered but never
  generated, exactly as a `node_size < 2` agent behaves in
  v2. It is not dropped with an error, because v2 does not
  error for it either; it simply never completes.
- Validation of a whole segment (step, closed links, all
  agents) happens BEFORE anything is admitted, so a failed
  call leaves the session untouched and the next valid call
  still works.
- Out-of-window agents are never silently moved to another
  segment. If the caller wants them in the next segment, the
  caller passes them there.

## Closure semantics (per-segment countdown)

- `closed_links` maps each closed directed link to the
  minutes left in its construction at the segment start.
- A link with `minutes_left > 0` is barred for ENTRY from
  the segment start for `min(minutes_left, step_minutes)`
  minutes. It reopens at `segment_start + minutes_left`
  if that falls inside the segment: mid-segment reopening
  takes effect for entry and rerouting from the next
  0.1-minute step onward. Otherwise it remains barred at
  segment end.
- A link absent from the dict, or with `minutes_left == 0`,
  is open for the whole segment.
- Multiple closed links are independent (single-link
  projects; no sync, no shared start or duration).
- Cars already ON a barred link finish traversing it:
  never ejected, never frozen. "On" means already placed
  in that link's entrance queue or beyond (arrival counter
  incremented). A vehicle in a barred link's loading buffer
  has not entered yet and is rerouted from its origin, or
  moved to origin waiting if no barred-free path exists.
- NO clear-first rule. The countdown starts at the segment
  start regardless of cars on the link. This is Eli's
  approved simplification and the one deliberate traffic
  difference from v2: v2 waited for each link to clear
  before starting the construction clock, so a v2 project
  with cars on its link at the scheduled start reopens
  later than the same v3 dict. Comparisons against v2 must
  use closures that start with the link empty, or expect
  that difference.
- Reroutes, all with barred links removed from the
  shortest-path graph (so no reroute can send a vehicle
  back onto a closed link):
  - not-yet-departed vehicles whose current path includes a
    barred link are rerouted at generation, from their
    origin;
  - in-system vehicles whose next link is barred are
    rerouted at the node, before entering the barred link,
    never mid-link;
  - vehicles queued to enter a barred link are rerouted
    from their current node instead of entering it.
- If no barred-free alternative path exists, the vehicle
  WAITS (never dropped): at its origin if not yet departed
  (retried every step, persisting across segments in the
  session's waiting dict), or in the previous link's exit
  queue if in-system. Waiting time counts toward the penalty
  through the first-link travel time (below), not during
  the waiting segment itself.

## Segment penalty — the telescoping version

Penalty is in vehicle-minutes, on the excess_closed basis:
against the vehicle's ORIGINAL free-flow path on the FULL
network, ignoring closures. The original path is computed
at admission (free-flow shortest path, barred links NOT
removed); its FFTT is the baseline.

For each link traversal COMPLETED during the segment, add
(actual link travel time - that link's FFTT), where the
first link's travel time is link exit minus departure
(including origin wait) and later links are successive
arrival-stamp differences, exactly as in v2's
`_link_travel_time`. PLUS, for each reroute event during
the segment, add (new remaining path FFTT from the current
node - old remaining path FFTT from the current node).
Reroute events are: the initial barred-avoiding path
assigned at generation (delta = actual path FFTT -
original path FFTT, counted in the generation segment),
a waiting vehicle's release path, a loading-buffer reroute,
and each en-route reroute at a node.

This telescopes. Over a vehicle's full trip, segment
penalties sum to (arrival - departure) - original-path
FFTT for a completed vehicle. The penalty is NOT just the
sum of (actual - FFTT) over actual links: that sum is
excess_actual and misses the detour FFTT delta. Example:
direct FFTT 5.0, detour FFTT 10.0, uncongested, closed
link: the segment penalty is 5.0 (detour delta 5.0 + link
travel-time terms 0.0), matching v2's excess_closed 5.0
for the same case.

Consequences, stated explicitly so the caller is not
surprised:

- Waiting time is counted when the vehicle's first link
  exits (its first-link travel time includes origin wait),
  not during the waiting segment itself. A segment in
  which a vehicle only waits has penalty 0 from that
  vehicle.
- A vehicle mid-link at segment end contributes nothing
  for that link yet; the traversal is booked when it
  leaves the link, in the segment where that happens.
- Incomplete vehicles at the end contribute only their
  traversed links plus their reroute deltas so far.
- A final drain segment (no new agents, no closures, long
  enough to drain) makes the segment penalties over all
  segments sum to the v2 full-run excess_closed total for
  completed vehicles, up to one documented difference: v2
  excess uses travel time = arrival minus leaving-buffer
  time, while the link-travel-time sum here uses arrival
  minus departure. The two agree exactly when vehicles
  enter their first link without storage-gate delay, as in
  uncongested runs; under storage-gate queueing they differ
  by that buffer wait. The delivery note reports the
  measured difference on the comparison tests instead of
  claiming exact equality.

## Horizon and MOE growth

The total horizon is unknown at session creation (segments
may continue indefinitely), so the MOE is NOT pre-sized to
a fixed horizon. The initial horizon is 300 minutes. Each
segment keeps 300 minutes of drain allowance past its end;
as the clock advances, each link's MOE list, the
assignment-period grid, and the shortest-path interval rows
are extended by appending, preserving already-written
minutes. The run never dies because the clock passed the
initial MOE size. The v2 `_post_loading` finalisation (MOE
tail propagation and completed-trip tally) is NOT run per
segment: per-segment state stays live, and only MOE minutes
already passed are final. Memory grows with simulated time
(one MOE object per link per minute): for Sioux Falls scale
(76 links) a 30-day run is about 3.3M objects, feasible; for
Baltimore scale (297,739 directional links) a long MOE array
is infeasible in this design, exactly as in v2. Do not claim
Baltimore multi-month runs work in this package.

## Generation timing precision

Departures are stored as floats and bucketed as in v2, by
`int(departure * 10)` in float32 arithmetic, into 0.1-minute
steps. A departure of 0.03 therefore generates at the 0.0
step and a departure of 0.12 at the 0.1 step; the free-flow
stamp is still departure + FFTT, so uncongested travel times
are exact and only the generation step is bucketed. The
simulation step itself is exactly 0.1 minutes (Eli confirmed
0.1, not 0.01 and not 1.0); 0.1 is the finest departure
precision the engine represents.

## Scope and limits (honest)

- One loading pass per session, flow models 1/2/3 only.
  There is no assignment iteration and no scenario file.
- All links are assumed Freeway type `"f"`. A network with
  real arterial, ramp or connector links is out of scope:
  its links would be silently treated as freeways.
- No signals, no ramps, no vehicle/demand types, agents must
  be pathless (see above).
- Paths are free-flow shortest paths excluding barred links
  (information_type 1). There is no prevailing-time or
  en-route information updating in bare mode: the info
  updating block never fires, so reroute paths do not react
  to congestion, only to closures. This matches v2.
- No Sioux Falls 54k run and no Baltimore run was performed
  for v3; the tests are synthetic small networks plus a
  small Sioux Falls smoke run (see the delivery note). Do
  not cite v3 numbers for either network at scale.
- The engine is deterministic given the same segment calls
  in the same order: same agents, same closed dicts, same
  step sizes. Changing the segmentation (for example one
  120-minute segment vs two 60-minute segments, with agents
  and closed dicts split accordingly) reproduces the same
  penalties and end state, because steps, barred sets per
  step, and generation buckets are all functions of the
  global step count, not of segment boundaries. The delivery
  note reports the measured test of exactly this.

## Packages needed

Python 3 plus `numpy` only, as in v2.

## Layout

- `session.py` — `create_session`, `run_segment`, `Session`,
  and all segment validation. The only entry points.
- `closures.py` — per-segment barred-link schedule
  (SegmentClosures). Replaces v2's `construction.py`.
- `loading.py` — the loader plus barred-link reroutes,
  origin waiting, and the segment-penalty bookkeeping
  (reroute deltas and completed-traversal terms). Adapted
  from v2; the v2 `network_loading` full-run loop and
  `_post_loading` finalisation are still present in the
  file but are NOT used by sessions and must not be run
  per segment.
- `network.py` — the data model with the bare readers
  (nodes id only, links hardcoded Freeway, zones generated
  from nodes) plus session mode (no agent file, dynamic
  horizon growth via `extend_horizon`).
- `sp.py` — shortest paths, with barred-link exclusion and
  the unreachable-destination fix. Unchanged from v2.
- `cpp.py` — CsvParser, float32 helpers, RNG helpers,
  `ProgramStop`. Unchanged from v2.
- Deleted vs v2: `api.py`, `run.py`, `construction.py`,
  `construction_outputs.py`.
