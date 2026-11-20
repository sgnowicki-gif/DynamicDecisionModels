"""Resume/checkpointing for network_builder v2.

Checkpoint granularity. ONE checkpoint file only:
``checkpoint.pkl`` in the output dir, overwritten after every
completed layer (epoch). No per-layer checkpoint history, no
``frontier_epoch_*.pkl`` files (v1's ``--save-frontier`` output
is gone; see network.py). A run killed mid-layer restarts from
the last completed layer; the partial layer is recomputed.

v3 (FORMAT_VERSION 3). The checkpoint no longer embeds the
frontier sessions: sessions live in per-session pickle files
under <output_dir>/sessions/ (see session_store.py), and the
checkpoint stores each frontier node's session KEY plus the
store's reference counts ("session_store"). A v2 checkpoint
fails loud here (format_version 2 != 3): v2 sessions carry
MOE history and v2 checkpoints embed sessions inline, so a
v2 checkpoint can never be resumed into v3 code -- fresh
restart is the plan. The fingerprint also carries the v3
engine flags (moe_removed, sessions_spilled, path_cache,
worker_prune), so resuming with a different engine setting
fails loud instead of silently mixing.

What a checkpoint contains (minimum to restart exactly):
frontier traffic sessions ONLY -- in v3, the session KEY of
each node at the completed epoch (the session itself lives
in <output_dir>/sessions/sess_<key>.pkl), and traffic_key=None
for every node at an earlier epoch. Only frontier nodes are
ever expanded again, and write_outputs never touches a
node's traffic, so the older sessions are dead weight (a
Sioux Falls checkpoint was already ~58 MB at epoch 1 with
all nodes' sessions stored; a v3 checkpoint is kilobytes
plus the sessions/ directory). All nodes metadata (id, epoch,
status, reaching_weight, group_id, unfinished_count), all
links (id, from, to, action, branch_probability, link_weight,
closed_set, profile_count), the counters next_node_id /
next_link_id / next_group_id, the totals dict, layer_stats,
the session store state (next_key and node reference counts),
and an input fingerprint (see below). The checkpoint also
carries a projects summary so resume can check the project
list itself matches, not just its file hash. Note the
in-memory consequence: after a resume, non-frontier nodes
in the returned Network carry traffic=None and traffic_key=None;
frontier nodes carry traffic_key set and traffic=None until
first expanded (loaded on demand via Network.traffic_of).
On-disk outputs are unaffected (the CSVs never contain
traffic).

What a checkpoint deliberately does NOT contain: the
transition cache. The cache is memoisation only: dropping
it cannot change the NETWORK -- nodes, links, penalties,
reaching weights, unfinished counts and merged traffic
states are identical to an uninterrupted run, because a
recomputed transition runs the same profiles from the same
stored session and produces the same penalty and averaged
end state. What CAN differ is REPORTING, not the network:
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
toy resume tests). One more thing CAN differ from an
uninterrupted run: group_id values assigned to transitions
recomputed in the resumed layers. group_id is only a cache
key and a stored label, never part of node identity
(identity is epoch + status) or of merging, so a different
group_id on a resumed-layer node changes nothing
downstream except the label itself. Nodes at or before the
checkpoint epoch keep their group_ids exactly.

Fingerprint. A SHA-256 over a canonical JSON document
describing the builder inputs. Both paths hash the FULL
bytes of the bare network files (input_node.csv and
input_link.csv), because pysim_construction reads link
attributes (length, lanes, speed, lane capacity,
jam_density, wave_speed) from input_link.csv into stored
sessions: hashing only node ids and link from/to pairs
would let an edited link attribute pass validation and
silently continue on the old network. The CLI builds it
from file bytes (projects file, demand file if any,
coordinates file or groups file if given, both network
files) plus the settings (start_date, flow_model,
random_seed, prune flag, k, kmeans seed). The Python API
path (build_network called directly, no file paths) builds
the same-shaped fingerprint from the loaded objects
instead (projects list, demand dict, groups dict,
coordinates dict) plus the same two network file hashes,
so a direct-API resume is validated the same way within
that path. The two paths hash different representations,
so a checkpoint written through the CLI must be resumed
through the CLI, and one written through the API through
the API; mixing entry points fails loud on fingerprint,
by design. The network DIRECTORY string is deliberately
NOT fingerprinted: identity comes from the file bytes,
so moving an unchanged network to a new path resumes
fine (tested); only the content matters.

Horizon is deliberately NOT part of the fingerprint, even
though it is stored in the checkpoint for provenance:
extending the horizon is the point of resume (build 2
epochs, resume to 4), so a fingerprint that included the
horizon would make every extension fail. What IS enforced
about the horizon is the fail-loud rule in build_network:
a checkpoint whose epoch is already >= the requested
horizon cannot be resumed. start_date, by contrast, is in
the fingerprint and must never change across a resume.

Writes are atomic: the checkpoint is written to a temp file
in the same directory and os.replace()d over checkpoint.pkl,
so a kill mid-write never leaves a half-written checkpoint
behind (the previous completed layer's file stays intact).
There is no fsync: a process kill is safe, but power-loss
durability is NOT guaranteed -- after an OS crash the
checkpoint may be missing or stale, and the run restarts
from the previous completed layer. A kill mid-write leaves
an orphan checkpoint.*.tmp file behind; orphan tmp files
are deleted at the start of the next run (see
prepare_output_dir), so they never accumulate.
"""

import hashlib
import json
import os
import pickle
import tempfile

FORMAT_VERSION = 3
CHECKPOINT_FILENAME = "checkpoint.pkl"


def _canonical(obj):
    """Canonical JSON bytes for hashing: sorted keys, no spaces."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_fingerprint(parts):
    """SHA-256 hex of the canonical JSON of a parts dict.

    Callers put one entry per input/settings item; the exact
    keys are the caller's business, but both the CLI path and
    the API path use the same key names for the same items so
    a fingerprint is comparable within its path.
    """
    if not isinstance(parts, dict):
        raise ValueError("fingerprint parts must be a dict")
    return sha256_bytes(_canonical(parts))


def network_files_sha256(network_dir):
    """SHA-256 of the bare network files' full bytes.

    Both input_node.csv and input_link.csv are hashed in
    full: link attributes (length, lanes, speed, lane
    capacity, jam_density, wave_speed) feed the stored
    sessions, so a byte change in either file must fail
    resume loud. The directory path itself is NOT part of
    the fingerprint (see the module docstring)."""
    return (sha256_file(os.path.join(network_dir, "input_node.csv")),
            sha256_file(os.path.join(network_dir, "input_link.csv")))


def fingerprint_from_files(projects_path, demand_path, coordinates_path,
                           groups_file_path, network_dir, start_date,
                           horizon, flow_model, random_seed,
                           prune_unfinishable_starts, k, kmeans_seed,
                           path_cache=True, worker_prune=True):
    """Fingerprint for the CLI path: hashes of input file bytes.

    demand_path / coordinates_path / groups_file_path may be
    None (recorded as such). The bare network is fingerprinted
    by the full bytes of input_node.csv / input_link.csv in
    network_dir (see network_files_sha256), not by the
    directory path: moving an unchanged network to a new path
    resumes fine, while any byte change -- topology or link
    attributes -- fails resume loud. horizon is accepted but
    intentionally NOT fingerprinted: extending the horizon is
    the point of resume (see the module docstring); the epoch
    >= horizon rule in build_network is the horizon check.
    """
    node_sha, link_sha = network_files_sha256(network_dir)
    parts = {
        "entry": "cli",
        "projects_file_sha256": sha256_file(projects_path),
        "demand_file_sha256": (sha256_file(demand_path)
                               if demand_path else None),
        "coordinates_file_sha256": (sha256_file(coordinates_path)
                                    if coordinates_path else None),
        "groups_file_sha256": (sha256_file(groups_file_path)
                               if groups_file_path else None),
        "network_node_file_sha256": node_sha,
        "network_link_file_sha256": link_sha,
        "start_date": start_date,
        "flow_model": flow_model,
        "random_seed": random_seed,
        "prune_unfinishable_starts": bool(prune_unfinishable_starts),
        "k": k,
        "kmeans_seed": kmeans_seed,
        # v3 engine flags: a resume with a different engine
        # setting fails loud instead of silently mixing (a v2
        # checkpoint fails earlier, on format_version).
        "moe_removed": True,
        "sessions_spilled": True,
        "path_cache": bool(path_cache),
        "worker_prune": bool(worker_prune),
    }
    return make_fingerprint(parts)


def fingerprint_from_objects(projects, demand, groups, coords,
                             network_dir, start_date,
                             horizon, flow_model, random_seed,
                             prune_unfinishable_starts, k, kmeans_seed,
                             profiles=None, path_cache=True,
                             worker_prune=True):
    """Fingerprint for the direct-API path: loaded objects.

    projects: list of Project; demand: {(o, d, dow, h): vph};
    groups: {node_id: group}; coords: {node_id: (x, y)} or None.
    The bare network is fingerprinted by the full bytes of
    input_node.csv / input_link.csv in network_dir (see
    network_files_sha256), not by the directory path or by
    parsed node ids / link pairs. profiles, when given, is
    the test-only profile override. horizon is accepted but
    intentionally NOT fingerprinted (see
    fingerprint_from_files).
    """
    node_sha, link_sha = network_files_sha256(network_dir)
    parts = {
        "entry": "api",
        "projects": [[p.project_id, p.project_type,
                      [int(p.link[0]), int(p.link[1])],
                      [[int(d), float(prob)] for d, prob in p.outcomes]]
                     for p in projects],
        "demand": sorted([[int(o), int(d), int(dow), int(h), float(v)]
                          for (o, d, dow, h), v in demand.items()]),
        "groups": sorted([[int(n), int(g)] for n, g in groups.items()]),
        "coords": (sorted([[int(n), float(x), float(y)]
                           for n, (x, y) in coords.items()])
                   if coords is not None else None),
        "network_node_file_sha256": node_sha,
        "network_link_file_sha256": link_sha,
        "start_date": start_date,
        "flow_model": flow_model,
        "random_seed": random_seed,
        "prune_unfinishable_starts": bool(prune_unfinishable_starts),
        "k": k,
        "kmeans_seed": kmeans_seed,
        "moe_removed": True,
        "sessions_spilled": True,
        "path_cache": bool(path_cache),
        "worker_prune": bool(worker_prune),
        "profiles": (None if profiles is None else
                     [[list(levels), float(prob)]
                      for levels, prob in profiles]),
    }
    return make_fingerprint(parts)


def projects_summary(projects):
    """Small identity record for the project list, stored in the
    checkpoint and checked on resume (in addition to the
    fingerprint)."""
    return [[p.project_id, p.project_type,
             [int(p.link[0]), int(p.link[1])],
             [[int(d), float(prob)] for d, prob in p.outcomes]]
            for p in projects]


def build_state(epoch, horizon, start_date, fingerprint, projects,
                nodes, links, next_node_id, next_link_id,
                next_group_id, totals, layer_stats,
                session_store_state=None):
    """Assemble the checkpoint state dict from live builder data.

    nodes: dict node_id -> _Node (with .traffic sessions); links:
    list of _Link. Only nodes AT the checkpoint epoch (the
    frontier) keep their traffic session in the state; every
    earlier node is stored with traffic=None, because only
    frontier nodes are ever expanded again (expansion
    iterates the frontier; write_outputs never touches a
    node's traffic) -- storing all sessions was the bulk of
    the checkpoint size for no benefit. The frontier sessions
    are pickled by reference into the state (pickle handles
    the deep copy on dump); callers must not mutate a stored
    session afterwards (the builder never does: transitions
    deepcopy first).
    """
    node_records = []
    for node in sorted(nodes.values(), key=lambda n: n.node_id):
        node_records.append({
            "node_id": node.node_id,
            "epoch": node.epoch,
            "status": node.status,
            "reaching_weight": node.reaching_weight,
            "group_id": node.group_id,
            "unfinished_count": node.unfinished_count,
            # v3: the session KEY, not the session. Only nodes AT
            # the checkpoint epoch (the frontier) keep a key;
            # earlier nodes are stored with traffic_key=None.
            "traffic_key": (node.traffic_key
                            if node.epoch == epoch else None),
        })
    link_records = []
    for link in links:
        link_records.append({
            "link_id": link.link_id,
            "from_node": link.from_node,
            "to_node": link.to_node,
            "action": link.action,
            "branch_probability": link.branch_probability,
            "link_weight": link.link_weight,
            "closed_set": dict(link.closed_set),
            "profile_count": link.profile_count,
        })
    return {
        "format_version": FORMAT_VERSION,
        "epoch": epoch,
        "horizon": horizon,
        "start_date": start_date,
        "fingerprint": fingerprint,
        "projects_summary": projects_summary(projects),
        "nodes": node_records,
        "links": link_records,
        "next_node_id": next_node_id,
        "next_link_id": next_link_id,
        "next_group_id": next_group_id,
        "totals": totals,
        "layer_stats": layer_stats,
        # v3: the session store state (next_key + node
        # reference counts) so a resume can rebuild the store.
        "session_store": session_store_state,
    }


def prepare_output_dir(output_dir, resume_from=None):
    """Run-start housekeeping for an output dir (see README).

    Always: delete stale v1 frontier_epoch_*.pkl files and
    orphan checkpoint.*.tmp files left by a kill mid-write.
    On a FRESH run (resume_from is None) only: a pre-existing
    checkpoint.pkl belongs to an earlier run, not this one,
    so it is renamed to checkpoint.stale.pkl (preserved, not
    deleted) -- otherwise a fresh run killed in layer 0
    would leave a stale checkpoint behind that could be
    mistaken for this run's. On a resume the checkpoint is
    left alone: either it IS the resume source (same dir),
    or it is overwritten after the first resumed layer.
    Returns the list of actions taken (paths renamed or
    deleted), for verbose logging by the caller.
    """
    import glob
    actions = []
    os.makedirs(output_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(output_dir,
                                        "frontier_epoch_*.pkl")):
        try:
            os.unlink(stale)
            actions.append("deleted stale %s" % stale)
        except OSError:
            pass
    for orphan in glob.glob(os.path.join(output_dir,
                                         "checkpoint.*.tmp")):
        try:
            os.unlink(orphan)
            actions.append("deleted orphan tmp %s" % orphan)
        except OSError:
            pass
    if resume_from is None:
        ckpt = os.path.join(output_dir, CHECKPOINT_FILENAME)
        if os.path.isfile(ckpt):
            stale_path = os.path.join(output_dir, "checkpoint.stale.pkl")
            os.replace(ckpt, stale_path)
            actions.append("renamed %s to %s" % (ckpt, stale_path))
        # v3: a fresh run owns its sessions/ directory. Spilled
        # sessions from an earlier run in this dir belong to the
        # stale checkpoint and must not be mistaken for this
        # run's (keys restart at 1). On a resume the directory
        # is left alone: it IS the resume source.
        import shutil
        sessions_dir = os.path.join(output_dir, "sessions")
        if os.path.isdir(sessions_dir):
            shutil.rmtree(sessions_dir)
            actions.append("deleted stale %s" % sessions_dir)
    return actions


def save_checkpoint(output_dir, state):
    """Atomically overwrite output_dir/checkpoint.pkl with state.

    Returns the checkpoint path. Raises ValueError if state is
    not a dict carrying the required keys.
    """
    required = ("format_version", "epoch", "horizon", "start_date",
                "fingerprint", "projects_summary", "nodes", "links",
                "next_node_id", "next_link_id", "next_group_id",
                "totals", "layer_stats", "session_store")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state must be a dict")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("checkpoint state is missing keys: %s"
                         % (missing,))
    if state["format_version"] != FORMAT_VERSION:
        raise ValueError("checkpoint format_version %r, expected %d"
                         % (state["format_version"], FORMAT_VERSION))
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CHECKPOINT_FILENAME)
    fd, tmp_path = tempfile.mkstemp(prefix="checkpoint.", suffix=".tmp",
                                    dir=output_dir)
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def load_checkpoint(path):
    """Load and structurally validate a checkpoint file.

    Fails loud (ValueError) on a missing file, an unpicklable /
    corrupt file, a wrong format version, or a state missing
    required keys or with wrong-typed core fields. Semantic
    checks (fingerprint, epoch vs horizon, projects summary)
    are the caller's job, in build_network().
    """
    if not os.path.isfile(path):
        raise ValueError("checkpoint file not found: %s" % path)
    try:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
    except Exception as exc:
        raise ValueError("checkpoint file %s is corrupt or not a "
                         "network_builder checkpoint: %s" % (path, exc)
                         ) from None
    if not isinstance(state, dict):
        raise ValueError("checkpoint file %s does not contain a "
                         "state dict" % path)
    required = ("format_version", "epoch", "horizon", "start_date",
                "fingerprint", "projects_summary", "nodes", "links",
                "next_node_id", "next_link_id", "next_group_id",
                "totals", "layer_stats", "session_store")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("checkpoint file %s is missing keys: %s"
                         % (path, missing))
    if state["format_version"] != FORMAT_VERSION:
        raise ValueError("checkpoint file %s has format_version %r, "
                         "expected %d (v1 frontier pickles and v2 "
                         "checkpoints cannot be resumed into v3: "
                         "v2 sessions carry MOE history and embed "
                         "sessions inline -- restart fresh)"
                         % (path, state["format_version"],
                            FORMAT_VERSION))
    if not isinstance(state["session_store"], dict) \
            or "next_key" not in state["session_store"] \
            or "refcounts" not in state["session_store"]:
        raise ValueError("checkpoint file %s has a bad session_store "
                         "state" % path)
    if not isinstance(state["epoch"], int) \
            or isinstance(state["epoch"], bool):
        raise ValueError("checkpoint file %s has a bad epoch %r"
                         % (path, state["epoch"]))
    if not isinstance(state["nodes"], list) \
            or not isinstance(state["links"], list):
        raise ValueError("checkpoint file %s has bad nodes/links"
                         % path)
    if not isinstance(state["totals"], dict) \
            or not isinstance(state["layer_stats"], list):
        raise ValueError("checkpoint file %s has bad totals/"
                         "layer_stats" % path)
    expected_totals = ("transitions_computed", "cache_hits",
                       "profile_runs", "transition_seconds", "merges",
                       "transition_log")
    missing_totals = [key for key in expected_totals
                      if key not in state["totals"]]
    if missing_totals:
        raise ValueError("checkpoint file %s has totals missing "
                         "keys: %s" % (path, missing_totals))
    # Node identity is (epoch, status): two records with the
    # same pair would restore to the same node, so a
    # checkpoint carrying one is malformed and fails here,
    # not later as a silent overwrite or a confusing merge.
    seen_keys = set()
    for record in state["nodes"]:
        if not isinstance(record, dict) or "epoch" not in record \
                or "status" not in record:
            raise ValueError("checkpoint file %s has a node record "
                             "missing epoch/status" % path)
        try:
            status_key = tuple(tuple(entry) for entry in record["status"])
        except (TypeError, ValueError):
            raise ValueError("checkpoint file %s has a node record "
                             "with a malformed status %r"
                             % (path, record["status"])) from None
        node_key = (record["epoch"], status_key)
        if node_key in seen_keys:
            raise ValueError("checkpoint file %s has duplicate "
                             "node key (epoch, status) %r"
                             % (path, node_key))
        seen_keys.add(node_key)
    for key in ("next_node_id", "next_link_id", "next_group_id"):
        if not isinstance(state[key], int) \
                or isinstance(state[key], bool) or state[key] < 0:
            raise ValueError("checkpoint file %s has a bad counter "
                             "%s=%r" % (path, key, state[key]))
    return state
