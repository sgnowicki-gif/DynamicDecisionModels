# network_builder v3 — backwards-induction NETWORK construction (no solver)

v3 is the optimised DDM production build (see "What's new
in v3" below). No semantics change from v2 except the
removal of MOE history (see "MOE history removed" in that
section): demand levels, Poisson ppf, k-means, merging,
pruning, agent ids, cache key and penalty are exactly as
in v2, and the three optimisations (path caching, worker
pruning, OD precompute) are penalty-exact by test. v2
checkpoints cannot be resumed into v3 -- restart fresh.

Builds the layered network for Eli's Dynamic Decision Models
backwards-induction analysis. This package constructs the
network only: nodes, actions, branch probabilities and link
weights (weighted penalties). It does NOT solve the Bellman
equation, choose an optimal policy, or apply a terminal
penalty. Those are later work, on top of the CSVs this package
writes.

## What's new in v3

v3 exists for one reason: the v2 full run does not fit.
The investigation (2026-11-19) projected ~2,989 h for a
20-epoch 8-project run even with every approved
optimisation, and frontier sessions in RAM projected
~130 GB by epoch 4 (the run dies around epoch 3 on a
32 GB machine). v3 therefore changes the production
configuration AND the engine, in that order:

Production configuration (Eli, 2026-11-19). 3 projects
only -- S1 small 11->14, S2 small 16->10, M1 medium
16->8 -- from construction_projects_v3.csv (the other
five v2 projects are deferred to future work); k = 3
groups, so 3^3 = 27 demand profiles (not 81);
horizon 28 epochs (one week from Mon 2026-07-06);
--workers 18 on Eli's 20-core / 32 GB Windows machine;
--prune-unfinishable-starts on. No coarsening or binning
of durations, no dominated-node pruning. With path
caching the 27-profile run projects to ~38 h, inside
the two-week deadline. The CLI default is still k = 4;
the DDM run passes --k 3 explicitly. The production
command is:

```
python -m network_builder.cli \
    --network-dir sioux_falls_bare_inputs \
    --projects construction_projects_v3.csv \
    --demand demand_sioux_falls.csv \
    --coordinates original_input_node.csv \
    --k 3 --kmeans-seed 42 \
    --start-date 2026-07-06 --horizon 28 \
    --workers 18 --prune-unfinishable-starts \
    --output-dir network_out
```

Path caching (admission Dijkstra). The vendored engine
recomputes the shortest path for every admitted vehicle
on every segment. In bare mode the travel-time table is
uniform per link (td_time identical in every row),
prevailing_tt == FFTT, and info_updating_flag is False,
so the best path for a given (origin, destination,
barred set) never changes within a segment: it is safe
to cache it. `pysim_construction/sp.py` gains
`find_best_path_cached()`, keyed on (origin,
destination, origin_zone, destination_zone,
demand_type, vot, barred frozenset); only the PATH is
cached, the cost is recomputed from the uniform TD row
on a hit. The cache verifies uniformity by scanning
the TD table once per table and disables itself
(falling back to the uncached path) if the table is
ever non-uniform, so a future non-bare mode cannot
silently get stale paths. All four `find_best_path`
call sites in loading.py (`_reroute_from`,
`assign_paths_iteration_zero`,
`generation_adjustment`, `enroute_adjustment`) use the
cached version. Measured ~2.8x per segment on Sioux
Falls. `--no-path-cache` disables it (spot checks
only); cached and uncached penalties match exactly
(test 17).

Worker pruning. Each profile end session is pruned
(completed vehicles to ids only, as in "Pruning"
below) in the worker before it is pickled back, so the
return pickle drops from ~22 MB to ~5.8 MB per profile
(investigator-measured on Sioux Falls). Sequential
runs prune inline, so sequential == parallel exactly.
`--no-worker-prune` disables it (spot checks only);
pruned and unpruned penalties and averaged link counts
match exactly (test 18).

OD / level precompute. `demand.py` gains
`prepare_epoch_demand()`: the sorted OD list and the
per-hour (low, medium, high) level counts for an epoch
are computed once per transition and shared by all 27
profile count/generate calls, instead of once per
profile. A precomputed context for the wrong epoch,
start day or group set fails loud. Counts and
generated agents are identical with and without the
precompute (test 19).

Disk-spilling frontier sessions. Frontier sessions no
longer live in RAM or inside checkpoint.pkl. Each
node's session is spilled to one file under
`<output_dir>/sessions/` (atomic tmp + replace),
reference-counted by `network_builder/
session_store.py` (`SessionStore`): a node holds a
reference while it is on the frontier, the transition
cache holds one while a cached end session is live,
and the file is deleted when the count reaches zero.
In-memory nodes carry `traffic=None` plus a
`traffic_key`; `Network.traffic_of(node)` loads the
session from disk on demand (and caches it in
`node.traffic`). A resumed run re-spills from the
checkpoint's sessions directory (copied when the
output dir differs) and never needs all sessions in
RAM. checkpoint.pkl is now ~2.5 KB in the example
(was megabytes in v2); the sessions live in
sessions/, not in the checkpoint. Cache entries
created before the current layer are dropped at layer
end, so the checkpoint records node references only.
Orphan session files are swept on resume. The
checkpoint format is v3; a v2 checkpoint (format 2)
is rejected with a fail-loud error -- restart fresh.

MOE history removed. MOE was write-only in the live
v3 path: nothing downstream reads it except dead code
and average.py's averaged MOE, which no analysis
reads. v3 therefore stops allocating it. `Link.
setup_moe` keeps FFTT / prevailing_tt / counters but
sets `link.moe = None`; `travel_time_by_min` returns
FFTT when moe is None (prevailing_tt needs no
fallback -- it is already FFTT in bare mode);
`arrival_flow` / `departure_flow` return 0; the three
MOE write sites in loading.py and the
`extend_horizon` length check are guarded on
`moe is not None`; average.py skips the MOE loop for
MOE-less sessions (the loop is kept for sessions that
do carry MOE). The checkpoint fingerprint gains
`moe_removed: True`, so a v2 checkpoint can never be
mistaken for a v3 one. What is lost: the averaged MOE
history only, a diagnostic -- no penalty, state or
downstream output changes. Penalties match v2
exactly on the same Sioux Falls slice (spot check
below). MOE was ~31% of stored session RAM and the
arrays grew with the clock, so removing it also
removes the session-growth term that bounded v2 runs.

Excluded (future work, deliberately not built).
A flat job queue over all profile runs in a layer,
and parallelising merges: same-target merges must
stay serialised, and neither change is needed -- the
27-profile projection (~38 h with path caching)
already fits the two-week deadline without them.
Both are recorded here so a later revision can pick
them up with a determinism test, not as an oversight.

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

## What's new in v3.2

v3.2 fixes one crash-resume bug and nothing else. In v3,
expanding a node released its session file as soon as the
refcount hit zero -- but the checkpoint written at the
previous layer boundary still listed that node, so a crash
mid-layer left the checkpoint pointing at deleted files
and the resume died later at load_traffic. That is what
happened to the v3 run: it crashed mid-layer in epoch 26,
and the resume failed with "session file missing for key
2868". (WinError 32 is a different failure -- the
os.replace file-lock race the v3.1 rename retry addresses
-- not the cause of the epoch-26 crash.) v3.2 pins every
session file named by the latest checkpoint: SessionStore
keeps a pinned set (the frontier nodes' traffic keys),
release(), cache_release() and sweep_orphans() never delete
a pinned file (refcounts still decrement), the pin set is
persisted in the checkpoint's session_store state ("pinned")
and restored on resume, so a second mid-layer crash is
resumable too, and the sweep after each checkpoint save
deletes files that are unpinned AND unreferenced, so disk
stays bounded to roughly two layers. Resume validation now
checks the session files actually exist on disk, not just
the refcount dict, so a missing file fails fast at resume
with a clear message instead of mid-run at load_traffic.
FORMAT_VERSION stays 3: "pinned" is an optional entry, so
v3 checkpoints remain loadable (their refcount keys are
pinned on restore, which are exactly their frontier keys).
No change to penalties, nodes, links, or any other file
format. No checkpoint.prev.pkl is kept: pinning makes the
single checkpoint resumable after a mid-layer crash, which
is what a second copy would have been for. Because the
deleted v3 files are unrecoverable, a run crashed this way
restarts fresh from epoch 0 -- v3.2 does not resurrect it.
The v3.1 os.replace retry (10 attempts, 0.25 s apart) is
unchanged in session_store.put and checkpoint.save_checkpoint.
The crash simulation is a shipped test:
tests/test_crash_resume.py.

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
    "2026-07-06", horizon=240, output_dir="network_out",
    coords=group_coords)
```

Command line (same inputs, same defaults):

```
python -m network_builder.cli \
    --network-dir sioux_falls_bare_inputs \
    --projects my_projects.csv \
    --demand demand.csv \
    --coordinates original_input_node.csv \
    --start-date 2026-07-06 \
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
flag (default off, see below). `--save-frontier` is accepted
but ignored in v2 (checkpoint.pkl supersedes it; see
"Resume / checkpointing").

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

Demand groups. Node groups via k-means on the original
Sioux Falls x/y, unscaled, deterministic seed 42, k = 4
by CLI default; the v3 DDM production run uses k = 3
(27 profiles -- see "What's new in v3"). K-means is
implemented in numpy (Lloyd, k-means++ initialisation,
empty-cluster re-seeding to the farthest point).
Assignments
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
per-category targets, link counters, travel times; MOE
in v3 is skipped -- sessions carry no MOE history, see
"MOE history removed" in "What's new in v3" -- the
MOE loop is kept only for sessions that do carry MOE).
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

Fixed bug (2026-11-17, found by SF slice testing, pre-existing
v1 code, not a resume bug): a donor session whose completed
vehicles had been pruned to ids only crashed
weighted_average_state with KeyError at the vehicle_order
sort. The pruned donor id (veh is None) was appended to
added_aids although no Vehicle was ever placed in
world.vehicles; in the SF merge 116 consecutive stale ids
(148153..148268, all pruned, none dropped) hit the departure
lookup. The fix is one guard in the donor-completed loop in
average.py: append to added_aids only when a Vehicle was
actually added; the pruned id is still recorded in completed
/ pruned_completed_ids. The hypothesised swap/drop path was
ruled out by instrumentation (no drop of an added aid fired).
Regression test 16-pruned-donor in
tests/test_network_builder.py reproduces the crash on the
unfixed code (KeyError: 11) and passes with the fix. The SF
2-project / 3-profile slice now completes horizon 2 (32
nodes, 40 links, 9 merges), and the resumed horizon-1-to-2
run matches the uninterrupted run exactly (nodes.csv,
links.csv, transition penalties).

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

This is the TRANSITION cache (whole transitions, keyed
above). v3 also adds a PATH cache inside the vendored
engine -- one cached shortest path per (origin,
destination, barred set) within a segment -- which is a
different mechanism with its own uniformity proof; see
"Path caching" in "What's new in v3". The two caches
compose: the path cache speeds up each profile segment,
the transition cache skips whole repeated transitions.

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
for a full run. v3 prunes each profile end session in the
worker before return (~22 MB -> ~5.8 MB per profile on
Sioux Falls), which is what makes --workers 18 viable
for the production run; see "Worker pruning" in "What's
new in v3".

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
states are NOT in the CSVs. In v3 they are not in RAM
either: each node's session lives in one file under
`<output_dir>/sessions/`, and `Network.traffic_of(node)`
loads it on demand (see "Disk-spilling frontier
sessions" in "What's new in v3"). `node.traffic` is
None until loaded, and is released again after the
node's layer is expanded. Resume from
output_dir/checkpoint.pkl (see "Resume /
checkpointing").

## Resume / checkpointing (v2, updated for v3)

v2 adds resume-from-checkpoint and changes nothing else
except the pruned-donor crash fix below (see "Fixed bug"):
no v1 semantics change (demand levels, Poisson ppf,
k-means, merging, pruning, agent ids, cache key, penalty).
What changed is only how a run survives being killed and
restarts.

Checkpoint granularity. ONE checkpoint file only:
`checkpoint.pkl` in the output dir, overwritten after every
completed LAYER (epoch). Never after a node or a branch, and
there is no per-layer checkpoint history. A run killed
mid-layer restarts from the last completed layer; the
partial layer is recomputed from scratch. In v3 the
checkpoint is small (keys and refcounts only, ~2.5 KB
in the example) because the frontier sessions live in
`sessions/`; see "Disk-spilling frontier sessions"
in "What's new in v3".

Why per-layer only: per-node/per-branch saving is
prohibitively slow and large. Measured on the toy network,
pickling one node costs 0.04-0.07 s and 0.27-0.38 MB, and
pickling a whole epoch-2 frontier costs 0.45 s and 2.3 MB;
in v2 a Sioux Falls session was already ~5.7 MB at 12
hours of clock and kept growing (per-link per-minute
MOE arrays grew with the clock). v3 removes MOE, so
that growth term is gone, but saving after every node
would still multiply both the runtime and the disk
footprint by the node count. Per-layer saving costs one
frontier spill per epoch and is the only granularity
that stays affordable.

What checkpoint.pkl contains (v3). The frontier traffic
session KEYS only -- one reference-counted key per
frontier node, pointing at its file in `sessions/` --
plus the session store state (refcounts); nodes at
earlier epochs are stored with traffic=None and no
key, because only frontier nodes are ever expanded
again and write_outputs never touches a node's
traffic. All nodes metadata (id, epoch, status,
reaching_weight, group_id, unfinished_count), all links
(id, from, to, action, branch_probability, link_weight,
closed_set, profile_count), the counters next_node_id /
next_link_id / next_group_id, the totals dict, layer_stats,
a projects summary, and an input fingerprint (below). The
format is built by `network_builder/checkpoint.py`
(`save_checkpoint(output_dir, state)` /
`load_checkpoint(path)`); writes are atomic (temp file +
os.replace), so a kill mid-write never leaves a
half-written checkpoint behind. There is no fsync:
process-kill resume is safe, but power-loss durability is
NOT guaranteed. A kill mid-write leaves an orphan
checkpoint.*.tmp file, deleted at the start of the next
run. After a resume, non-frontier nodes in the returned
in-memory Network carry traffic=None; on-disk outputs are
unaffected (the CSVs never contain traffic).

What it deliberately does NOT contain: the transition
cache. The cache is memoisation only: dropping it cannot
change the NETWORK -- nodes, links, penalties, reaching
weights, unfinished counts and merged traffic states are
identical to an uninterrupted run, because a recomputed
transition runs the same profiles from the same stored
session and produces the same penalty and averaged end
state. What CAN differ is REPORTING, not the network:
totals["transition_seconds"] and the per-entry "seconds"
in totals["transition_log"] are wall-clock measurements,
so they always differ between two runs (resumed or not).
totals["transitions_computed"], "cache_hits",
"profile_runs" and the length of "transition_log" differ
only if a resumed layer would have hit a cache entry
populated by an earlier, completed layer -- which needs
the same (traffic_state_group_id, closed link set, day of
week, starting hour) to recur across layers, and the
(day, hour) pair recurs only every 28 epochs, so it
cannot happen in short runs (verified identical on the
toy resume tests, including a cache-hit resume test with
5 large-project branches). One more thing CAN differ from
an uninterrupted run: group_id values assigned to
transitions recomputed in the resumed layers. group_id is
only a cache key and a stored label, never part of node
identity (identity is epoch + status) or of merging, so a
different group_id on a resumed-layer node changes
nothing downstream except the label itself. Nodes at or
before the checkpoint epoch keep their group_ids exactly.

How to resume. Finish stage 1, then rerun with the same
inputs plus `--resume-from`:

```
python -m network_builder.cli \
    --network-dir sioux_falls_bare_inputs \
    --projects my_projects.csv \
    --demand demand.csv \
    --coordinates original_input_node.csv \
    --start-date 2026-07-06 \
    --horizon 240 \
    --output-dir network_out \
    --resume-from network_out/checkpoint.pkl
```

The CLI still loads projects/demand/groups from the CURRENT
arguments, so the checkpoint fingerprint can be validated
against them, and it passes the same network_dir/projects/
demand/coordinates/start_date/horizon/flow_model/
random_seed/workers/prune settings through. The Python API
equivalent is `build_network(..., resume_from=<path>)`
(plus an optional caller-computed `fingerprint=`; when
omitted it is computed from the loaded objects). The
example flow is `example_resume.py`: horizon 2, then
resume to horizon 4 (29 nodes, 34 links, 30 transitions
computed, 6 merges). Those counts are
profile-count-independent, so they are checked with the
full 9-profile k=2 enumeration in the example itself;
full node/link/penalty equivalence against an
uninterrupted run is verified in tests/test_resume.py --
to horizon 4 with the 3-profile override, and to horizon
2 with the full 9-profile enumeration. Sioux Falls slice
equivalence was verified externally on the 2-project
3-profile slice at horizon 2 only; no Sioux Falls
full-profile equivalence is claimed anywhere.

Fingerprint validation and fail-loud cases. The fingerprint
is a SHA-256 over the builder inputs: the projects file
bytes, the demand file bytes (if any), the groups /
coordinates input identity (file bytes of whichever was
given), the FULL bytes of both bare network files
(input_node.csv and input_link.csv -- hashing only node
ids and link from/to pairs would miss an edited link
attribute such as length or speed, which pysim_construction
reads into the stored sessions), start_date, flow_model,
random_seed, the prune flag, k and the k-means seed (plus
the test-only profile override, when one is used), and in
v3 the build flags `moe_removed: True`,
`sessions_spilled: True`, `path_cache` and
`worker_prune` (so a checkpoint built with the path
cache off cannot be resumed into a run with it on, and
vice versa). The
network directory PATH is deliberately not fingerprinted:
identity comes from the file bytes, so moving an unchanged
network to a new path resumes fine. Resume fails loud
(ValueError, no partial network written) when: the
fingerprint mismatches (any of those inputs changed -- a
changed projects file, a changed start_date, a changed
demand file, a changed network topology, a changed prune
flag, and (via the CLI) a changed link attribute are the
tested cases);
the checkpoint file is missing or corrupt (bad pickle, or a
pickled object of the wrong shape -- including a v1
frontier pickle, which has no fingerprint and cannot be
resumed); a v2 checkpoint (format_version 2, expected
3) is rejected outright -- v2 sessions carry MOE history
and embed sessions inline, so restart fresh rather than
resuming; or the checkpoint epoch is already >= the
requested horizon (nothing left to resume). Horizon itself
is deliberately NOT part of the fingerprint, even though it
is stored in the checkpoint: extending the horizon is the
point of resume (build 2 epochs, resume to 4), so a
fingerprint that included the horizon would make every
extension fail; the epoch >= horizon rule is the horizon
check. Checkpointing is always on when output_dir is
given; there is no separate flag.

`--save-frontier` (v1) is removed as behaviour and kept as
a no-op alias: the flag is still accepted, prints a note
that checkpoint.pkl supersedes the v1
`frontier_epoch_*.pkl` files, and otherwise does nothing.
Stale `frontier_epoch_*.pkl` files in the output dir are
deleted at the start of a run, as are orphan
`checkpoint.*.tmp` files left by a kill mid-write. A
stale `sessions/` directory from an earlier run is
deleted at the start of a FRESH run only; on a resume
the store is restored from the checkpoint and orphan
session files are swept. A pre-existing
`checkpoint.pkl` in the output dir is renamed to
`checkpoint.stale.pkl` (preserved, not deleted) at the
start of a FRESH run only, so a fresh run killed in layer 0
leaves no stale checkpoint behind that could be mistaken
for the new run's; on a resume the checkpoint is left
alone. Add `network_out/` to `.gitignore` (session files
are large binary files and are regenerable from the inputs
by rerunning).

## Layered expansion and checkpointing

Expansion is epoch by epoch: all nodes at epoch k are
expanded before any node at k+1. Per-layer stats are logged
(and printed with --verbose) and the outputs above are
rewritten after each layer, so a killed run still leaves the
layers completed so far on disk. After each completed layer
the single checkpoint.pkl is also overwritten (see "Resume /
checkpointing" above). The CSVs alone are never enough to
resume, because traffic states are not in the CSVs.

## Example mode (deliberate deviation, flagged)

The synthetic example uses k = 2 groups (9 profiles) instead
of production k = 4 (81 profiles), so example_run.py finishes
in tens of seconds instead of minutes. Node/link counts do
not depend on the profile count; only penalty values do.
The v3 DDM production run is k = 3 (27 profiles); the
CLI default remains k = 4. The example's groups
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

## Sioux Falls validation (v3)

Spot checks (one closed-link epoch-0 transition, S1 11->14
for one epoch, full demand_sioux_falls.csv, k = 3 groups,
27 profiles, sandbox sequential). Five variants of the
same transition, penalty / generated agents /
in-system vehicles compared exactly, including the
per-link (loading, entrance, exit) queue counts of the
averaged end session:

- v3 cached + pruned (production): penalty 7615.461691,
  24,726 generated, 2,560 in system, 361.8 s.
- v3 uncached + pruned: penalty 7615.461691, identical
  counts, 868.5 s. Path cache is penalty-exact and
  ~2.4x faster on this transition.
- v3 cached + unpruned: penalty 7615.461691, identical
  counts, 289.4 s. Worker pruning is penalty-exact;
  the pruned run was slower here only because the
  slice job shared the sandbox CPUs at the time.
- v3 uncached + unpruned: penalty 7615.461691,
  identical counts, 735.3 s.
- v2 (pristine, uncached + unpruned, MOE kept): penalty
  7615.461691, identical counts, 786.5 s. MOE removal
  is penalty-exact against v2 on the same slice.

All five agree exactly (penalty to full float precision,
same generated and in-system counts, same per-link
queues). The S1 penalty 7615.461691 also matches the
layer-0 S1 transition in the horizon-1 slice run below,
an independent run of the same transition through the
CLI. The spot-check script is not shipped in this zip
(it hardcodes the sandbox paths); the same
equivalence is covered in the shipped suite by tests
17 (path cache), 18 (worker prune) and 20 (no MOE)
on the toy network.

Horizon-1 slice (3-project file, 27 profiles, k = 3,
full demand, sandbox sequential): 14 nodes, 13 links,
4 transitions computed, 0 cache hits, 0 merges, 108
profile runs, 1,416.4 s. Layer-0 penalties:
do-nothing 1390.828, S1 7615.462, S2 14166.504,
M1 4847.089. checkpoint.pkl 2,702 bytes (format v3);
frontier sessions spilled as 4 files of ~3.0 MB each
(one per transition end state; the 13 epoch-1 nodes
share 4 keys -- S1's 2 branches share one, S2's 2
share one, M1's 8 share one, do-nothing one).

## Limitations and open issues (read before a full run)

- MOE history is REMOVED in v3 (see "MOE history
  removed" in "What's new in v3"): sessions no longer
  carry per-link per-minute MOE arrays, so the v2
  session-growth term is gone. What is lost is the
  averaged MOE history only, a diagnostic; no penalty,
  state or downstream output changes. Sessions still
  grow with the clock through vehicle queues and
  td_map buckets, but far more slowly than the MOE
  term grew.
- Excluded from v3 by design (future work): a flat job
  queue over all profile runs in a layer, and
  parallelising merges (same-target merges must stay
  serialised). See "Excluded" in "What's new in v3".
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
  (weighted average_state; v3: MOE averaging skipped for
  MOE-less sessions), checkpoint.py (v3: checkpoint
  save/load, input fingerprint with the v3 build flags),
  session_store.py (v3: reference-counted disk-spilling
  session store), ids.py (ids, pruning, renumbering,
  offsetting), demand.py (profiles, levels, Poisson ppf,
  generation; v3: prepare_epoch_demand precompute),
  groups.py (k-means, groups files), projects.py
  (project CSV validation), network.py (nodes, actions,
  transitions, cache, merging, layered expansion,
  outputs, resume; v3: disk-backed traffic, worker
  pruning), cli.py (v3: --no-path-cache,
  --no-worker-prune), and pysim_construction/ (vendored
  v3, with its README.md; v3: admission path cache in
  sp.py, MOE guards in loading.py and network.py).
- example_project.csv, example_demand.csv,
  example_coordinates.csv, example_network/
  (input_node.csv, input_link.csv), example_run.py,
  example_resume.py (v2: horizon 2, then resume to 4).
- tests/test_network_builder.py -- the test suite (21
  areas, 88 checks; see the test file for the mapping
  to the brief). v3 adds tests 17-21: path-cache
  equivalence and population, worker-prune
  equivalence, epoch_ctx equivalence and mismatch,
  no-MOE segment and averaging, session spilling
  (sessions/ layout, format v3, small checkpoint,
  frontier keys, traffic_of load).
- tests/test_resume.py -- the v2 resume tests: toy
  resume-vs-uninterrupted equivalence (horizon 2 then
  resume to 4, 3-profile override; plus a full 9-profile
  equivalence to horizon 2), a cache-hit resume test
  (5 large-project branches), fingerprint mismatch
  (start date, demand, projects, network topology, prune
  flag, and a CLI link-attribute change), missing/corrupt
  checkpoint (including malformed totals and duplicate
  node keys), checkpoint epoch >= horizon, next_group_id
  validation, frontier-only checkpoint sessions,
  stale-checkpoint rename and orphan-tmp cleanup on a
  fresh run, and single-file checkpoint overwrite; v3
  adds the format-2 rejection test (a v2 checkpoint
  cannot be resumed into v3) and the resumed-run
  frontier-keys checks (epochs 0-3 hold no session,
  epoch-4 frontier keys load via traffic_of).
- This README.md.
