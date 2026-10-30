# network_builder v1 — backwards-induction NETWORK construction (no solver)

Builds the layered network for Eli's Dynamic Decision Models
backwards-induction analysis. This package constructs the
network only: nodes, actions, branch probabilities and link
weights (weighted penalties). It does NOT solve the Bellman
equation, choose an optimal policy, or apply a terminal
penalty. Those are later work, on top of the CSVs this package
writes.

Built on two prior deliverables, vendored here so the package
is self-contained (no imports from any other working tree at
runtime):

- pysim_construction v3 (stepwise PySim): `network_builder/
  pysim_construction/`, a verbatim copy of the v3 package from
  pysim_construction_v3.zip, with its own README.md inside that
  folder. API: `create_session(network_dir, flow_model=2,
  random_seed=100)` + `run_segment(session, agents,
  closed_links, step_minutes)` returning exactly
  `{"penalty", "state"}`. Step 0.1 minutes, clock starts at
  0.0, departures absolute, half-open window
  [now, now+step), fail loud on out-of-window / duplicate /
  unknown agents. Construction is external per segment,
  countdown from segment start, no clear-first rule. Penalty
  is the link-traversal version (traversed-link (tt - fftt)
  plus reroute deltas), which telescopes to the v2
  excess_closed total except for loading-buffer/storage-gate
  delay (v3 0.130 vs v2 0.300 in the one gated case found
  during v3 testing, by design, not a defect).
- average_state: `network_builder/average.py`, adapted from
  average_state.py. It adds `weighted_average_state(sessions,
  weights)` and keeps `average_state(sessions)` as the
  unweighted wrapper (equal weights; the only difference from
  the original is one extra diagnostics key, "weights"). It
  also accepts pruned sessions (see "Pruning" below).

Python 3, numpy only, plus the standard library. scipy and
sklearn are never required: scipy is used for the Poisson ppf
only when installed (a verified fallback is built in), and
k-means is implemented in numpy (sklearn is never consulted).

## Quick start (synthetic example)

Extract this zip into a clean directory, then from that
directory:

```
python example_run.py
```

Expected output: 11 nodes, 12 links, 2 merges, in tens of
seconds (see "Example mode" below for why it is not 81
profiles). Outputs land in `example_output/` (nodes.csv,
links.csv, groups.csv, run_report.json). The full test suite
is `python tests/test_network_builder.py` (a few minutes; the
full-81-profile transition test alone takes ~25 seconds on
the toy network).

## Running it with real inputs

Python API:

```python
from network_builder.demand import load_demand
from network_builder.groups import compute_groups, load_coordinates
from network_builder.network import build_network, read_bare_network
from network_builder.projects import load_projects

node_ids, link_set = read_bare_network("sioux_falls_bare_inputs")
projects = load_projects("my_projects.csv", link_set)
demand = load_demand("demand.csv")
coords = load_coordinates("original_input_node.csv")  # x/y file
groups, group_coords = compute_groups(node_ids, coordinates=coords,
                                      k=4, seed=42)
network = build_network(
    "sioux_falls_bare_inputs", projects, demand, groups,
    "2026-11-02", horizon=240, output_dir="network_out",
    coords=group_coords)
```

Command line (same inputs, same defaults):

```
python -m network_builder.cli \
    --network-dir sioux_falls_bare_inputs \
    --projects my_projects.csv \
    --demand demand.csv \
    --coordinates original_input_node.csv \
    --start-date 2026-11-02 \
    --horizon 240 \
    --output-dir network_out
```

Required files: a bare network directory with input_node.csv
(node_id only) and input_link.csv (the Sioux Falls bare
inputs, or any bare v3 network); a projects CSV (below); a
start_date. Demand is a parameter: pass a demand CSV, or omit
it on the CLI for a zero-demand build (structure and merging
still exercised, all penalties 0). Coordinates are required
for production grouping; `--groups-file` accepts a
precomputed groups file instead (tests/examples only).
`--workers N` (N > 1) parallelises profiles with a
ProcessPoolExecutor; the default (0/1) is sequential.
`--prune-unfinishable-starts` enables the optional prune
flag (default off, see below). `--save-frontier` pickles the
frontier traffic states per layer (see "Checkpointing").

## Locked specification

Epochs. One epoch = 360 minutes. Horizon T epochs, default
240 (60 days), configurable. Node k has clock k * 360 minutes.
Epoch 0 starts at midnight on start_date (YYYY-MM-DD,
user-supplied); the day of week is derived from the date and
cycles daily; epochs are aligned to midnight, so no epoch
crosses a day. There are 4 epochs per day.

Projects CSV. Header exactly `project_id,project_type,
from_node_id,to_node_id,duration_epochs,probability`, one row
per duration outcome, project_id repeated. Types small /
medium / large only; no project count is hardcoded (Eli's
real file is expected to hold about 8 projects: 4 small,
3 medium, 1 large, but the builder validates types, not
counts). Duration ranges, enforced unless
`allow_out_of_range_durations=True`: small 1-2 epochs, medium
4-12, large 28-56. Probabilities per project sum to 1.0
within 1e-6. Each project has exactly one directed link; the
link must exist in input_link.csv; a link may belong to at
most one project; a (project_id, from, to, duration) row may
not repeat; a project_id may not carry two links or types;
self-loops fail. Everything else fails loud at load time.

Actions. Exactly one new start per epoch, or do nothing.
Only not-started projects can start. Multiple projects can
be in progress at once, but only one TYPE at once: if any
project is in progress, a new start must be of that same
type; if none is in progress, any type may start.

REMAINING DEFINITION (used everywhere, pinned by tests).
remaining counts the epochs still to run INCLUDING the next
epoch. A project started at epoch k with duration d epochs
is closed during epochs k .. k+d-1 and completed at epoch
k+d. The branch node at epoch k+1 therefore carries
remaining = d - 1 (or completed if d == 1). A node at epoch
k with in-progress remaining r has that link closed during
epoch k, and the next node carries remaining r - 1 (or
completed if r - 1 == 0). In v3 terms, closed_links maps an
in-progress link to remaining * 360 minutes and a newly
started link to duration * 360 minutes. Minutes_left is
always a multiple of 360, never fractional.

Demand groups. 4 node groups via k-means on the original
Sioux Falls x/y, unscaled, k = 4, deterministic seed 42,
implemented in numpy (Lloyd, k-means++ initialisation,
empty-cluster re-seeding to the farthest point). Assignments
are saved as node_id,group,x,y in groups.csv. For a synthetic
or test network without x/y, a precomputed groups file
(node_id,group, optional x,y) is accepted instead.

Demand levels. Per origin group only, independent across
groups, held for a whole epoch. Medium probability 0.5, high
0.25, low 0.25. With 4 groups there are 3^4 = 81 profiles;
a profile's probability is the product over groups, and the
81 probabilities sum to 1. For each hour h of the epoch,
base is the average demand for that OD pair, that hour, that
day of week (the day of the epoch). Counts: medium =
round_half_up(base); high = round_half_up(poisson_ppf(0.75,
base)); low = round_half_up(poisson_ppf(0.25, base)).
round_half_up is floor(x + 0.5) for x >= 0 (NOT Python's
banker's round). The Poisson ppf uses scipy when installed,
otherwise a mode-anchored cumulative-pmf fallback (checked
against scipy to base 50000). Base may be float.

Demand file. CSV with header origin_node_id,
destination_node_id, day_of_week, hour, average_demand_vph
(extra columns ignored; missing columns fail loud). The
canonical day_of_week forms are: integer 0 = Monday ..
6 = Sunday, or a three-letter abbreviation Mon..Sun (any
capitalisation). Anything else (full names, other
abbreviations, out-of-range numbers) fails loud at load
time. start_date's weekday uses datetime.weekday(), which is
also Monday = 0, so the two agree by construction. An OD pair
with no row for a needed (day, hour) is zero demand for that
hour; the file is sparse by design. Demand is a parameter:
no path that only exists on one machine is hardcoded.

Departures. D cars in an hour depart equidistantly:
hour_start + i * 60 / D for i = 0 .. D-1. First car exactly
at the hour start, last strictly before the next hour,
D = 0 means no cars. Departures are NOT rounded by the
builder; v3 buckets to int(departure * 10) (0.1 minutes), so
departures less than 0.1 minutes apart share a bucket in the
simulator -- disclosed v3 behaviour, not a builder choice.
All departures are absolute minutes from simulation start
(epoch k starts at k * 360), generation splits exactly by
epoch, and every generated departure lies in
[epoch_start, epoch_start + 360). A departure outside that
window fails loud in v3.

Transition (per node, per action). All profiles are run,
each from a DEEP COPY of the node's traffic state (the
stored session is never mutated by a transition), with that
profile's agents and the action's closed set, step_minutes =
360, through v3 run_segment. Weighted penalty =
sum(profile_prob * profile_penalty). Weighted end state =
weighted_average_state(profile end sessions, profile
probs). The transition is computed ONCE per action; a start
action then creates one branch node per duration outcome,
all sharing that averaged traffic state and penalty,
differing only in construction remaining and branch
probability = P(duration outcome). A do-nothing action
creates one next node. Branch sharing is mandatory and is
tested (the branch nodes initially reference the same
session object).

Node identity and merging. A node is (epoch,
construction_status_vector). When a transition leads to an
existing node, the traffic states merge:
new_traffic = weighted_average_state([existing, incoming],
[existing_reaching_weight, incoming_path_probability]),
normalised inside weighted_average_state, where
incoming_path_probability = source_reaching_weight *
branch_probability. Demand profile probabilities are NOT
included again (demand was already averaged out inside each
transition). reaching_weight starts at 1.0 for the root and
accumulates: it is the sum of incoming path probabilities
merged into a node. It is a merging weight, NOT a
probability -- it can exceed 1.0 when several decision
histories merge (the example's merged node has reaching
1.5), because decisions are not random -- and it is NOT
used in any Bellman equation (later work). NOTE: the brief's
phrase "existing_reaching_weight starts at 1.0 for root's
children" is read here as the product rule throughout (root
= 1.0, child = parent * branch probability), so a root
do-nothing child has reaching 1.0 and a root start child has
reaching = its duration probability; the literal alternative
(first arrival always 1.0) would contradict the product rule
in the same paragraph. The merge function takes explicit
weights, so the convention can be changed in one place if
Eli rules the other way.

Terminal nodes. Epoch == T has no outgoing actions. Each
terminal node stores unfinished_count = number of projects
not completed at T. NO terminal penalty number is invented
here; it is applied at analysis time, documented as large
enough to force completion.

## Weighted average_state

weighted_average_state(sessions, weights): weights are
normalised to sum 1 inside; every mean in the original
algorithm becomes a weighted mean (medoid feature mean,
per-category targets, link counters, MOE, travel times).
The medoid is the session closest to the weighted mean
feature vector (counts unweighted, prevailing_tt normalised
by FFTT, as in the original). Rounded targets use
largest-remainder on the weighted mean counts. Adjustment
from the medoid base is as in the original (moves only on a
vehicle's remaining path, donor adds blocked on duplicate
id, swap pass, etc.). Diagnostics in average_info include
the weights used, weighted mean vs actual, unmet_targets,
and the original warning thresholds (link residual > 2 or
total > 5% of mean total). Tests: weights [1, 0] reproduce
the first session (up to deepcopy); [0.75, 0.25] is closer
to the first session than the unweighted average; a single
session returns a deepcopy; empty / mismatched / negative /
sum-zero weights fail loud.

## Agent ids

Scheme (see network_builder/ids.py for the full rationale,
including the v3 int32 check): v3 agent ids are plain Python
ints (dict keys, Vehicle.agent_id) -- there is no int32 cast
or fixed-width id field in the vendored code, so offsets are
safe inside one transition. They are still not stored.
Per transition from a stored session S: base =
next_agent_id_base(S) = max id in S + 1 (1 if S has no ids);
profile p's agents are base + p * stride + j, where stride
exceeds the largest profile's agent count (stride =
max_count + 1000) and j enumerates that profile's agents in
generation order. Profile runs therefore never collide with
in-system ids and never collide across profiles in a way
that blocks donor adds. After averaging,
renumber_session(avg, base) rewrites every id >= base to a
fresh sequential id starting at base, in (departure, old id)
order for vehicles and old-id order for pruned completed
ids, rewriting EVERY reference: vehicles dict keys,
Vehicle.agent_id, vehicle_order, waiting dict keys, every
link queue item, td_map and original_td_map values, pruned
ids and seen_ids. Ids >= base that averaging dropped
(admitted in a profile run but not placed in the average)
are dropped from seen_ids too: they were synthetic profile
ids that never entered any stored node's history, and future
builder ids are assigned sequentially from the new base, so
they cannot be reused by accident. History merging needs a
different step first: two histories' stored sessions both
carry compact sequential ids for different trips, so the
incoming session is offset above the existing session's max
before averaging, and the merged session is then renumbered
to a compact 1..N range (base = 1) and pruned. That is safe
for the same reason: the builder assigns every future id,
and the retained ids after renumbering are exactly the
merged session's current vehicles, waiting vehicles and
pruned completed ids. Invariants tested: ids in a stored
session are unique; averaged ids are unique and do not
collide with future demand for that node; future demand
starts at max + 1 (helper next_agent_id_base(session));
a duplicate of a retained id fails loud in the next segment.
ID growth over 240 epochs is therefore bounded by the total
number of distinct retained vehicles, not by 81 * epochs *
offsets: roughly one sequential id per vehicle ever retained
in a stored state, not per profile run.

## Pruning

After averaging, before storing a node traffic state,
completed vehicles are pruned to an id-only set:
session.pruned_completed_ids holds their ids, the Vehicle
objects leave world.vehicles and world.vehicle_order, and
sim.generated and sim.completed stay as integer counters.
Retained: seen_ids (so a later segment reusing a completed
id still fails loud -- v3 checks seen_ids, not the vehicles
dict); td_map past buckets (the loader only reads the
current step's bucket, a completed vehicle's bucket is in
the past, and the loader guards with world.vehicles.get and
skips a missing vehicle); all in-system and waiting vehicles
in full. Penalty is unaffected: the v3 segment penalty is
computed live from link traversals and reroute deltas, never
from the completed list. average_state accepts pruned
sessions: completed counts come from the sim.completed
counter, completed ids are the union of full completed
vehicles and pruned ids, and donor completed ids that exist
only as pruned ids are carried as ids, never fabricated
back into Vehicle objects. Tested: pruning preserves
counters and the completed id set, duplicate reuse still
fails loud, and averaging a pruned session works.

## Caching

Cache key = (traffic_state_group_id, closed link frozenset,
epoch day of week, epoch starting hour). Minutes are
deliberately NOT in the key: every closed link's
minutes_left is a multiple of 360 and the step is fixed at
360, so any closed link is barred for the whole segment
regardless of its remaining count, and transitions that
differ only in remaining counts (the usual case for branch
nodes of one start action doing nothing next) have identical
penalties and end states. If the step ever stopped being
360, this cache would be wrong and must be revisited.
traffic_state_group_id is shared by branch nodes created
from one transition; a merged node gets a fresh group id.
A cache hit returns a deepcopy of the cached end session,
never the cached object itself. Tested: 5 large-project
branch nodes doing nothing give 4 cache hits and identical
penalties (1 transition computed, not 5). Cache stores
(weighted penalty, averaged end session); callers must not
mutate a cached session directly -- the transition function
deepcopies before running, and cache reads deepcopy.

## Parallel profiles

Profiles are independent (deepcopy per profile, no shared
mutable state), so the transition function is parallelisable.
Base runs sequentially; --workers N (N > 1) runs profiles
through a ProcessPoolExecutor. Sequential and parallel give
identical penalties (exact, deterministic -- bare mode draws
no RNG) and identical averaged link counts (tested on the
toy network). Parallel is NOT the default: each worker job
pickles a whole session, which is expensive on the real
network (see "Runtime" below), so measure before using it
for a full run.

## Optional prune flag

--prune-unfinishable-starts (default off in the base
builder, documented as the first optimisation for a full
run): when True, a project whose minimum duration exceeds
the remaining epochs (T - epoch) is not started, because it
cannot finish by T and starting adds closure penalty without
avoiding terminal penalty. Tested: flag on/off give the same
network except doomed branches removed. It is off by
default because it is a pruning choice, not a definition,
and the unpruned network is the reference.

## Outputs

In-memory Network object: nodes dict (node_id -> node with
epoch, status, traffic session, reaching_weight, group_id,
unfinished_count for terminal nodes), links list, and
methods nodes_at(epoch), outgoing(node_id),
terminal_nodes(). On disk (output_dir, rewritten after
every layer): nodes.csv (node_id, epoch, clock_min,
construction_status JSON, reaching_weight,
unfinished_count for terminal nodes, blank otherwise);
links.csv (link_id, from_node, to_node, action
("do_nothing" or project_id), branch_probability,
link_weight (weighted penalty), closed_set JSON,
profile_count = 81 in production); groups.csv
(node_id,group,x,y, or node_id,group without coordinates);
run_report.json (per-layer nodes, actions, transitions
computed, cache hits, merges, seconds, plus totals and a
per-transition log with epoch, from-node, action, penalty
and seconds); layers/layer_<k>.json holds the same per-layer
summary for layer k alone. Traffic
states are NOT in the CSVs; access them in memory via
node.traffic, or via the optional frontier pickles below.

## Layered expansion and checkpointing

Expansion is epoch by epoch: all nodes at epoch k are
expanded before any node at k+1. Per-layer stats are logged
(and printed with --verbose) and the outputs above are
rewritten after each layer, so a killed run still leaves the
layers completed so far on disk. With --save-frontier, the
frontier traffic states at each layer boundary are also
pickled to frontier_epoch_<k>.pkl, with a printed size
warning. Resume limitations, stated honestly: v1 does NOT
implement resume-from-pickle. The pickle format is
{"epoch": k, "nodes": [(node_id, epoch, status,
reaching_weight, group_id, traffic_session), ...]}, which a
resume implementation could load and continue from, but
without the frontier states -- in memory or from a pickle --
a run cannot be resumed, and the CSVs alone are never
enough, because traffic states are not in the CSVs. A
resume would also need the exact builder inputs (network,
projects, demand, groups, start_date, horizon, flow model,
seed, workers, prune flag) and the cache/group-id counter
state, which are not yet persisted.

## Example mode (deliberate deviation, flagged)

The synthetic example uses k = 2 groups (9 profiles) instead
of production k = 4 (81 profiles), so example_run.py finishes
in tens of seconds instead of minutes. Node/link counts do
not depend on the profile count; only penalty values do.
Production mode is k = 4, always, and the example's groups
are computed by the same k-means code path (not a toy
implementation). The example network is the v3 4-node
network (input_node.csv / input_link.csv) plus
example_coordinates.csv (square corners) for grouping.
Hand-computed example result (horizon 2, projects S small
1/2 at 0.5 each on 1->2, M medium 4 at 1.0 on 1->3): 11 nodes
(1 + 4 + 6), 12 links (4 + 8), 2 merges, transitions
computed 10, cache hits 0. Terminal unfinished counts:
(NS,NS) 2, (completed,NS) 1, (in-progress,NS) 2,
(NS,in-progress) 2, (completed,in-progress) 1,
(NS,in-progress at remaining 2) 2.

## Runtime (measured, not estimated from theory alone)

Toy network (4 nodes, 5 links), sequential: one 360-minute
v3 segment ~0.28 s; one full 81-profile transition ~25 s
(includes 81 deepcopies and the weighted average); the
2-epoch example (10 transitions, 9 profiles each) ~33 s.
Sioux Falls bare network (24 nodes, 76 links), sequential:
one 360-minute segment ~4.0 s (20 cars) and ~4.0 s empty,
~4.6 s with 894 slice cars; one session deepcopy ~1.3-1.6 s
early in a run (growing with the clock, see below). A
measured 2-epoch Sioux Falls slice (real geometry, real
Monday hour 0-1 demand for 10 OD pairs from the DDM Full
Demand sheet, 20 rows, 891 vph total, converted from full
day names to Mon abbreviations; 2 projects S/M; 3-profile
override for speed, NOT 81 profiles) took 188 s for 11
nodes, 12 links, 10 transitions and 30 profile runs, with
epoch-0 do-nothing penalty 27.7, close-1->2 penalty 2774
and close-1->3 penalty 6723. Scaling the measured
per-profile cost (segment + deepcopy ~5.9 s) to a full
81-profile transition gives roughly 81 * 5.9 s + averaging,
about 8 minutes per node-action early in a run, before
averaging and before the session grows. A full
240-epoch, 8-project network has a node count that grows
combinatorially with epochs and projects, and each
transition costs minutes on the real network: do NOT start
a full run on the basis of this builder alone. Measure a
2-epoch slice first, use the prune flag, and consider
parallelism and node-count growth before committing. The
run_report.json per-transition seconds exist so Eli can
extrapolate from his own slice, not from these numbers.

## Limitations and open issues (read before a full run)

- MOE memory growth is a v3 property, not a builder bug,
  but it bounds full runs: a v3 session carries per-link
  per-minute MOE arrays that grow with the clock (horizon
  extends to segment end + 300 each segment). A Sioux Falls
  session pickled at ~12 hours is already ~5.7 MB, and the
  arrays keep growing to the full 86,400-minute horizon.
  Deepcopying and pickling such sessions 81 times per
  transition is the dominant cost late in a run, and
  frontier pickles inherit it. A future v3 revision that
  truncates or compresses MOE history would change the
  full-run arithmetic materially; this builder does not
  modify the vendored engine.
- The averaged session is a synthetic representative state,
  not a state any single run produced (inherited from
  average_state): its future penalty is not expected to
  equal the mean penalty of its inputs, and heavily
  overlapping ids can leave link targets unmet (recorded in
  average_info, with the original warning thresholds). The
  builder's renumbering and offset steps keep ids disjoint
  where it controls them (profiles, merging), but the
  underlying averaging approximation remains.
- Reaching-weight convention: see "Node identity and
  merging" -- the product rule is used throughout; if Eli
  intended first-arrival weight 1.0 for root children, that
  is a one-place change in merge_traffic's callers, flagged
  for the coordinator, not silently assumed.
- Demand file DOW: only integers 0..6 and Mon..Sun
  abbreviations are accepted. A real file using full day
  names must be converted first; the loader fails loud
  rather than guessing.
- The builder does not validate that demand origins are a
  subset of network nodes at load time (generation fails
  loud on an ungrouped origin instead); a demand file with
  an origin outside the network fails at the first
  transition that uses it, not at startup. Group coverage
  for network nodes IS checked at grouping time.
- Parallel workers require the package to be importable in
  worker processes (same directory on sys.path); on a
  cluster this needs the extraction directory on every
  worker's path, which the CLI does not set up.
- No solver, no policy, no terminal penalty value: see the
  top of this file. unfinished_count is stored so the
  analysis stage can apply the penalty.

## Contents

- network_builder/ -- the package: __init__.py, average.py
  (weighted average_state), ids.py (ids, pruning,
  renumbering, offsetting), demand.py (profiles, levels,
  Poisson ppf, generation), groups.py (k-means, groups
  files), projects.py (project CSV validation),
  network.py (nodes, actions, transitions, cache, merging,
  layered expansion, outputs), cli.py, and
  pysim_construction/ (vendored v3, with its README.md).
- example_project.csv, example_demand.csv,
  example_coordinates.csv, example_network/
  (input_node.csv, input_link.csv), example_run.py.
- tests/test_network_builder.py -- the test suite (16
  areas; see the test file for the mapping to the brief).
- This README.md.
