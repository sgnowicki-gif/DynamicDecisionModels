# pysim_construction — bare PySim with construction closures (DDM only), version 2

A specialised copy of the standalone PySim loading pass for
Eli's Dynamic Decision Models project. It replaces Sebastian's
`traffic_sim.py` for DDM use only. It is NOT a general DTALite
replacement, and it must not be overlaid onto the standalone
`pysim` tree.

## Breaking changes vs v1 — read this first

Version 2 is NOT drop-in compatible with version 1 of this
package, even though the folder name (`pysim_construction`) and
the function name (`run_pysim_construction`) are the same. The
v1 zip on disk is untouched; this is its successor, not a patch
over it.

- Inputs are exactly four bare CSV files (below). The v1
  DTALite-format inputs (demand file list, zone file, link-type
  file, scenario settings, DTASettings.txt, demand/vehicle-type
  files, emission-rate file) are no longer read. If old files
  are still sitting in the input directory they are ignored,
  not read and not rejected — but do not rely on them: nothing
  in them affects a v2 run.
- Emissions are removed. `emissions_full.py` is deleted, there
  is no `emissions` argument and no `--emissions` CLI flag. An
  old v1 call with `emissions=True` fails with a clear
  `TypeError` (unexpected keyword argument), it is not silently
  ignored.
- Outputs are only three files (below). `outputs.py` is deleted:
  no `output_agent.csv`, `output_trip.csv`,
  `output_LinkTDMOE.csv` or `output_LinkMOE.csv` is written.
  The internal link MOE arrays are still built and used for
  travel-time lookups and shortest paths; only the writers are
  gone, so dropping them cannot change traffic results.
- The result dict no longer contains `emissions` or the four
  old output paths. `outputs` has exactly three keys: `excess`,
  `excess_summary`, `construction_projects`.
- Clean extraction required. If you unzip v2 over an existing v1
  package folder, the deleted v1 files (`emissions_full.py`,
  `outputs.py`, plus any `__pycache__` entries) are left behind
  as orphans. Extract v2 into a new, empty directory instead.

## Quick start

```python
import sys
sys.path.insert(0, "/path/to/this/folder/..")  # parent of pysim_construction/

from pysim_construction import run_pysim_construction

# construction_projects.csv must be in the input directory (or
# passed explicitly); a missing file fails loud, it never runs
# silently without closures.
result = run_pysim_construction("my_inputs", "my_outputs")
print(result["outputs"]["excess"])  # .../my_outputs/output_excess_travel_time.csv
print(result["excess"])  # totals over completed vehicles
```

A small runnable example is included in the zip under
`example_inputs/` (top level, next to `pysim_construction/`):
a four-node network with one construction project. Run it with
`run_pysim_construction("example_inputs", "example_outputs")`
and compare against the numbers in the delivery note.

## Signature

```python
def run_pysim_construction(input_dir, output_dir, verbose=False,
                           construction_file=None, flow_model=2,
                           random_seed=100):
```

- `input_dir`: the four bare input files below. Read, never
  written.
- `output_dir`: created if needed; receives only the three
  construction files below.
- `construction_file`: default `input_dir/construction_projects.csv`.
  An explicit value may be an absolute path or a file name inside
  `input_dir`. Missing file: `ProgramStop`, no silent run.
- `flow_model`: 1, 2 or 3 (default 2, Newell). Any other value
  fails loud.
- `random_seed`: default 100, but it has no effect on results in
  bare v2 (see below).
- `verbose`: prints the input/output directories and the final
  agent/completed counts.

Returns a dict with `input_dir`, `output_dir`, `outputs`
(`excess`, `excess_summary`, `construction_projects` only),
`agents` / `completed`, the excess totals under `excess`, and
one row per project-link under `construction_projects`.

A command-line form is also available:

```
python -m pysim_construction.run <input_dir> <output_dir>
    [--construction <file>] [--flow-model 2] [--seed 100]
```

There is no `--emissions` flag.

## Bare input schema — exactly four files

### 1. input_node.csv

```
node_id
1
2
```

Only `node_id` is required. `x` / `y`, if present, are ignored:
no output in this package uses coordinates. A missing `node_id`
column fails loud; a duplicate `node_id` fails loud.

### 2. input_link.csv

```
from_node_id,to_node_id,length,number_of_lanes,speed_limit,lane_capacity_in_vhc_per_hour,jam_density,wave_speed
1,2,2.5,1,60,6000,180,12
```

Required: `from_node_id`, `to_node_id`, `length`,
`number_of_lanes`, `speed_limit`,
`lane_capacity_in_vhc_per_hour` (`lane_cap` is accepted as an
alias; the long name is primary and wins if both are present).
Optional, with defaults: `jam_density` (180),
`wave_speed` (12); the aliases `jam_density_in_vhc_pmpl` and
`wave_speed_in_mph` are also accepted. A missing `length`
defaults to 1.0, as in v1.

Dropped columns, all ignored if present, never rejected:

- `direction`: one row is always exactly one directed link,
  from -> to as written. No reverse link is created and no link
  is reversed. A file with only 1 -> 2 does NOT create 2 -> 1.
- `link_type`: hardcoded to 1, type code `"f"` (Freeway),
  bias 1.0, capacity adjustment 1.0. Verified: Sioux Falls is
  76/76 link_type 1 Freeway with bias 1.0, so hardcoding `"f"`
  preserves its merge logic, and that logic never triggers on
  Sioux Falls (no node has exactly one outgoing link).
- `mode_code`, `geometry`, `link_type_name` and every other
  extra column: ignored. Every row is a vehicle link (bare
  files have no walk/transit/bike rows; Sioux Falls `mode` is
  empty).

Fail loud on: missing from/to node, a from/to node that is not
in the node file, a duplicate directed link, `number_of_lanes`
0, capacity below 1, or speed below 1.

### 3. input_agent.csv — fixed filename, no demand file list

```
agent_id,origin_node_id,destination_node_id,departure_time_in_min
1,1,4,12
```

Bare columns: `agent_id`, `origin_node_id`,
`destination_node_id`, `departure_time_in_min`. Aliases are
accepted for the OD columns: `from_origin_node_id` for origin,
`to_destination_node_id` for destination. `from_zone_id` /
`to_zone_id`, if present, are accepted and ignored: zone = node
(the zone dict is generated internally from the nodes, in node
file order).

Agents MUST be pathless. If a `path_node_sequence` column is
present and non-empty for any agent, the run fails loud with
"bare v2 agents must be pathless" — a path is never silently
ignored. A duplicate `agent_id` fails loud. A departure below 0
fails loud.

Hardcoded per agent (the v1 defaults): `demand_type` 1,
`vehicle_type` 1, PCE 1, `information_type` 1, value of time 10,
age 5. With `information_type` 1, `info_updating_flag` stays
False, so every reroute is a free-flow shortest path excluding
barred links — deterministic, as a DDM reward needs. No RNG
draw is made for information class or age in bare v2, so
`random_seed` has no effect on results; that is stated honestly
here rather than left as a silent no-op.

Simulation window: start = minimum departure, end = maximum
departure, horizon = end + 300 (the v1 construction horizon
rule, including the multi-day cap lifted to 1,000,000 minutes
and the horizon-sized assignment grid). Warm-up is not special
code: warm-up is just early departures in the agent file, a few
hours before construction start.

### 4. construction_projects.csv — unchanged from v1

```
project_id,project_name,from_node_id,to_node_id,start_time_in_min,duration_in_min
```

One row per directed link per project; the same `project_id`
for a multi-link or both-direction project, with the same start
and duration on every row of the project (differing values fail
loud). See the closure semantics below; the fail-loud rules are
the same as v1 (missing file/column, unknown link, duplicate
project-link row, duration <= 0, non-numeric times).

## Dropped files and code

Never read, never required, ignored if present in the input
directory (an old input directory still runs, with identical
results): `DTASettings.txt`, `input_scenario_settings.csv`,
`input_link_type.csv`, `input_zone.csv`,
`input_activity_location.csv`, `input_demand_type.csv`,
`optional_vehicle_type.csv`, `optional_vehicle_emission_rate.csv`,
`input_demand_file_list.csv`. There is no signal feature:
`signal_model` is hardcoded to 0. PySim does not read
node-control or movement files, and Sioux Falls has no signals
defined anyway, so nothing is lost by not adding one. There are
no ramps (every link is type `"f"`), no vehicle or demand types,
and no zones file: zone = node, generated internally.

## Closure semantics (project-level sync, unchanged from v1)

- `scheduled_start` is the project's common start time. At that
  time ALL links in the project become BARRED: no vehicle may
  ENTER a barred link after that time.
- Reroutes, all with barred links removed from the shortest-path
  graph (so no reroute can send a vehicle back onto a closed
  link):
  - not-yet-departed vehicles whose current path includes a
    barred link are rerouted at generation, from their origin;
  - in-system vehicles whose remaining path includes a barred
    link are rerouted at the next node, before entering the
    barred link, never mid-link;
  - vehicles queued to enter a barred link are rerouted from
    their current node instead of entering it. In the loader's
    terms, "queued to enter" means in the previous link's exit
    queue awaiting discharge, or still in the first link's
    loading buffer (which has not yet counted as entered: the
    arrival counter is only incremented on entrance). A vehicle
    in a barred link's loading buffer is moved to origin waiting
    (see below) rather than left there, because leaving it there
    would block the link from ever clearing.
- Vehicles already ON a barred link at `scheduled_start` finish
  traversing it normally: not ejected, not frozen. "On" means
  already placed in that link's entrance queue or beyond
  (arrival counter incremented).
- A link is clear when its arrival count equals its departure
  count AND its loading, entrance, and exit queues are all empty.
  A project is clear when ALL its links are clear.
- `construction_start` (per project) is the first simulation
  time at or after `scheduled_start` at which the project is
  clear; if already clear at `scheduled_start`, it equals
  `scheduled_start`. `construction_end = construction_start +
  duration_in_min` (common duration). At `construction_end`
  ALL links in the project reopen together and rerouting stops
  for that project. Times are reported at 0.1-minute step
  granularity: a link that truly clears at 5.0 may be recorded
  at 5.1 if its last discharge lands on a step boundary; treat
  construction times as accurate to 0.1 min.
- If `construction_end` is beyond the simulation horizon, the
  status is `still_closed_at_horizon` and there is no reopen.
  If `scheduled_start` is beyond the horizon, or after an early
  exit with no vehicles left, the status is `never_started`
  (scheduled start recorded, no traffic impact).
- Overlapping projects on the same directed link do not crash
  and do not reopen early. A link is barred while ANY active
  project bars it (union); it reopens only when no active
  project still bars it. Project completion is still recorded
  per project, at its own end time.
- If no barred-free alternative path exists, the vehicle WAITS
  (never dropped): at its origin if not yet departed, or in
  the previous link's exit queue if in-system. Waiting time
  counts toward excess (see below). Origin waiting is held
  outside the loader's buffers and retried every step; when it
  loads, its free-flow stamp is load time + FFTT (not departure
  + FFTT), so the link is not traversed instantly on release.
- Early exit does not fire while waiting vehicles remain
  or while any project is barred or in construction, even if
  the network is otherwise drained, so a project's end is
  reached and recorded as completed rather than
  still-closed. It MAY fire after all vehicles complete while a
  project is still only pending; that project is then
  `never_started`.

## Excess travel time (both values, unclamped)

At generation, BEFORE any closure reroute, each vehicle's
original path is stored: for bare pathless agents, the
free-flow shortest path assigned at iteration 0 on the full
network (no barred links).

- `trip_fftt_closed` = sum of FFTT over that original path.
- `trip_fftt_actual` = sum of FFTT over the links actually
  traversed (PySim's post-loading `trip_fftt`, kept as actual,
  never overwritten).
- `travel_time` = PySim `veh.travel_time` (arrival minus
  leaving-buffer time), NOT `trip_time`. For origin-waited
  vehicles the leaving-buffer time is preserved from the first
  wait, so closure waiting is inside `travel_time`; for
  in-system waiting it already is, because the buffer time was
  set at origin entry.
- `excess_closed_in_min = travel_time - trip_fftt_closed`;
  `excess_actual_in_min = travel_time - trip_fftt_actual`.
  Neither is clamped to zero: a reroute to a shorter path can
  legitimately be negative. Incomplete vehicles carry empty
  excess fields, are counted separately, and are excluded from
  the totals. Totals are over completed vehicles only.

## Outputs written to output_dir — only three

- `output_excess_travel_time.csv`, one row per vehicle:
  `agent_id,origin,destination,departure,arrival,travel_time_in_min,`
  `trip_fftt_actual_in_min,trip_fftt_closed_in_min,`
  `excess_actual_in_min,excess_closed_in_min,completed`
  (`origin`/`destination` are node numbers; `completed` is 1/0;
  empty fields for incomplete vehicles as described above;
  times in minutes, 4 decimals.)
- `output_excess_summary.csv`, one totals row:
  `completed,incomplete,total_excess_closed_in_min,`
  `mean_excess_closed_in_min,total_excess_closed_in_hours,`
  `mean_excess_closed_in_hours,total_excess_actual_in_min,`
  `mean_excess_actual_in_min,total_excess_actual_in_hours,`
  `mean_excess_actual_in_hours`
  (means are over completed vehicles only; 0.0 when none
  completed.)
- `output_construction_projects.csv`, one row per
  project-link:
  `project_id,project_name,from_node_id,to_node_id,`
  `scheduled_start_in_min,barred_at_in_min,link_cleared_at_in_min,`
  `construction_start_in_min,construction_end_in_min,`
  `duration_in_min,status`
  (`status` is `completed`, `still_closed_at_horizon`, or
  `never_started`; for `never_started` the barred/cleared/
  start/end fields are empty; for a still-closed project whose
  construction never started, start/end are empty but any
  individually cleared links keep their cleared times.)

No other file is written to `output_dir`.

## Suggested reward (not implemented)

No reward function is implemented in the code. A suggested
form for later work, only: reward =
`-(total excess_closed) - lambda * max(0, finish - deadline)`,
where `finish` is the last project end and `deadline` is a
DDM-chosen deadline. This is a suggestion, not code, and
`lambda` and `deadline` are not defined here.

## Scope and limits (honest)

- One loading pass, flow models 1/2/3 only, exactly one
  assignment iteration (there is no scenario file to ask for
  more).
- All links are assumed Freeway type `"f"`. A network with real
  arterial, ramp or connector links is out of scope for bare v2:
  its links would be silently treated as freeways.
- No signals, no ramps, no vehicle/demand types, agents must be
  pathless (see above).
- Multi-day runs: the demand-period cap is 1,000,000 minutes
  and the assignment-period grid is sized by the horizon (both
  carried over from v1). The 300-step ring buffers in the loader
  are a deliberate 30-minute ring, not a horizon cap.
- Runtime is driven by the last arrival or reopening, not by
  the horizon alone, because steps are
  (last completion - start) / 0.1 with the early exit, and
  because the early exit is vetoed while a project is barred
  or in construction. A run that must reach a day-4 reopening
  therefore steps to day 4 even if the network drains earlier.
- Memory: one MinuteMOE object per link per minute of horizon.
  For Sioux Falls scale (76 links, 30 days = 43,200 min) that
  is about 3.3M objects, feasible. For Baltimore scale
  (297,739 directional links) a 30-day MOE array is infeasible
  in this design (297k x 43k objects). Do NOT claim Baltimore
  multi-month works in this package.
- "On a link" and "queued to enter" are defined above in
  loader terms (entrance queue onward vs exit queue / loading
  buffer). A vehicle in a barred link's entrance queue at the
  scheduled start is treated as on the link and finishes it;
  it is not rerouted.
- No Sioux Falls 54k run and no Baltimore run was performed for
  v2; the tests are synthetic small networks (see the delivery
  note). Do not cite v2 numbers for either network.

## Packages needed

Python 3 plus `numpy` only, as in v1.

## Layout

- `api.py` — `run_pysim_construction()` and the result dict.
- `run.py` — loading orchestration, original-path capture, and
  the three output writers. CLI: `python -m pysim_construction.run`.
- `construction.py` — construction file parsing/validation and
  the project state machine (barred union, clear, start, end).
  Unchanged from v1.
- `construction_outputs.py` — the three output writers and the
  excess totals. Unchanged from v1.
- `loading.py` — the loader plus barred-link reroutes, origin
  waiting, and the early-exit construction veto. Unchanged
  from v1 (it never touched emissions or the old outputs).
- `network.py` — the data model with the bare readers: nodes
  (id only), links (hardcoded Freeway, direction ignored),
  zones generated from nodes, pathless agents only, window
  from the agent file, flow model / seed as arguments.
- `sp.py` — shortest paths, with barred-link exclusion and the
  unreachable-destination fix. Unchanged from v1.
- `cpp.py` — CsvParser, float32 helpers, RNG helpers,
  `ProgramStop`. The DTASettings INI reader was removed: no
  settings file is read in bare v2.
- Deleted vs v1: `emissions_full.py`, `outputs.py`.
