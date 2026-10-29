# pysim_construction — PySim with construction closures (DDM only)

A specialised copy of the standalone PySim (v2) loading pass for
Eli's Dynamic Decision Models project. It replaces Sebastian's
`traffic_sim.py` for DDM use only. It is NOT a general DTALite
replacement, and it must not be overlaid onto the v2 tree: the
folder name is `pysim_construction` (not `pysim`) precisely to
avoid an import collision with the v2 package.

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

`run_pysim` is an alias for `run_pysim_construction` inside this
package only, so code written against the v2 name keeps working
here.

## Signature

```python
def run_pysim_construction(input_dir, output_dir, emissions=False,
                           verbose=False, construction_file=None):
```

- `input_dir`: DTALite-format input files (as in v2) plus
  `construction_projects.csv`. Read, never written.
- `output_dir`: created if needed; receives the four v2 files
  unchanged plus the three construction files below.
- `emissions`: the v2 boolean toggle. False = off, True = full
  VSP (requires `optional_vehicle_emission_rate.csv`).
- `construction_file`: default `input_dir/construction_projects.csv`.
  An explicit value may be an absolute path or a file name inside
  `input_dir`. Missing file: `ProgramStop`, no silent run.

Returns a dict like v2 (`input_dir`, `output_dir`, `emissions`,
the four original output paths, `agents` / `completed`) plus:
`outputs.excess`, `outputs.excess_summary`,
`outputs.construction_projects`, the excess totals under `excess`,
and one row per project-link under `construction_projects`.

A command-line form is also available:

```
python -m pysim_construction.run <input_dir> <output_dir>
    [--emissions] [--construction <file>]
```

## Construction file schema

`construction_projects.csv`, header exactly:

```
project_id,project_name,from_node_id,to_node_id,start_time_in_min,duration_in_min
```

- One row per DIRECTED link per project. A project closing both
  directions of one link is 2 rows with the same `project_id`.
  A project closing several links is several rows, same id.
- `project_name` is optional (may be empty, carried to output).
  Every other column is required.
- All rows of one `project_id` must share one
  `start_time_in_min` and one `duration_in_min`; differing values
  fail loud, because a project starts together and finishes
  together.

Fail loud (no silent degradation) on: missing file, wrong
header, missing column, unknown `(from_node_id, to_node_id)`
pair (no link in PySim's `link_map`), duplicate
`(project_id, from, to)` row, `duration_in_min <= 0`,
non-numeric times, or differing start/duration within a project.

## Closure semantics (project-level sync)

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
- Early exit (v2) does not fire while waiting vehicles remain
  or while any project is barred or in construction, even if
  the network is otherwise drained, so a project's end is
  reached and recorded as completed rather than
  still-closed. It MAY fire after all vehicles complete while a
  project is still only pending; that project is then
  `never_started`, as the brief allows.

## Excess travel time (both values, unclamped)

At generation, BEFORE any closure reroute, each vehicle's
original path is stored: the preassigned input path, or for
pathless agents the free-flow shortest path assigned at
iteration 0 on the full network (no barred links).

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

## Outputs written to output_dir

The four v2 files, unchanged in format:

- `output_agent.csv`, `output_trip.csv`,
  `output_LinkTDMOE.csv`, `output_LinkMOE.csv`

Plus three new files:

- `output_excess_travel_time.csv`, one row per vehicle:
  `agent_id,origin,destination,departure,arrival,travel_time_in_min,`
  `trip_fftt_actual_in_min,trip_fftt_closed_in_min,`
  `excess_actual_in_min,excess_closed_in_min,completed`
  (`origin`/`destination` are node numbers, as in the agent
  file; `completed` is 1/0; empty fields for incomplete vehicles
  as described above; times in minutes, 4 decimals.)
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

## Suggested reward (not implemented)

No reward function is implemented in the code. A suggested
form for later work, only: reward =
`-(total excess_closed) - lambda * max(0, finish - deadline)`,
where `finish` is the last project end and `deadline` is a
DDM-chosen deadline. This is a suggestion, not code, and
`lambda` and `deadline` are not defined here.

## Scope and limits (honest)

- Everything in the v2 scope section applies here: one loading
  pass, flow models 1/2/3 only, exactly one assignment iteration.
- Multi-day/month runs: all horizon caps in this copy were
  audited and fixed (see BUILD_NOTE.md). The demand-period cap
  is lifted from 2880 to 1,000,000 min (fail loud only for a
  negative start, an end beyond that cap, or end at/before
  start), and the assignment-period grid is sized by the
  horizon instead of a fixed 300 slots (the old grid indexed
  past its end for departures beyond about 4485 min). The
  300-step ring buffers in the loader are a deliberate 30-minute
  ring, not a horizon cap, and were left alone.
- Runtime is driven by the last arrival or reopening, not by
  the horizon alone, because steps are
  (last completion - start) / 0.1 with the v2 early exit, and
  because the early exit is vetoed while a project is barred
  or in construction. A run that must reach a day-4 reopening
  therefore steps to day 4 even if the network drains earlier.
- Memory: one MinuteMOE object per link per minute of horizon.
  For Sioux Falls scale (76 links, 30 days = 43,200 min) that
  is about 3.3M objects, feasible. For Baltimore scale
  (297,739 directional links) a 30-day MOE array is infeasible
  in this design (297k x 43k objects). Do NOT claim Baltimore
  multi-month works in this package.
- Output format limits carried over from v2/DTALite:
  LinkMOE hourly columns (`vol_h1..24`, `speed_h1..24`) cover
  only the first 24 hours; LinkTDMOE has one row per link per
  minute to the horizon, so a 30-day run writes a very large
  file even for a small network.
- "On a link" and "queued to enter" are defined above in
  loader terms (entrance queue onward vs exit queue / loading
  buffer). A vehicle in a barred link's entrance queue at the
  scheduled start is treated as on the link and finishes it;
  it is not rerouted.

## Packages needed

Python 3 plus `numpy` only, as in v2.

## Layout

- `api.py` — `run_pysim_construction()` (plus the `run_pysim`
  alias) and the result dict.
- `run.py` — loading orchestration, original-path capture, and
  the three new output writers.
- `construction.py` — construction file parsing/validation and
  the project state machine (barred union, clear, start, end).
- `construction_outputs.py` — the three new output writers and
  the excess totals.
- `loading.py` — the v2 loader plus barred-link reroutes,
  origin waiting, and the early-exit construction veto.
- `network.py` — v2 data model, with the horizon fixes and the
  per-vehicle closed-path fields.
- `sp.py` — v2 shortest paths, with barred-link exclusion and
  the unreachable-destination fix (see BUILD_NOTE.md).
- `outputs.py`, `cpp.py`, `emissions_full.py` — byte-identical
  to v2 (the four original output files are unchanged).
