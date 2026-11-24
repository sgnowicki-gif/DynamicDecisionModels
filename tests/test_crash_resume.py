# Tests for network_builder v3.2 crash-resume pinning. Run from the
# extraction directory:
#   python tests/test_crash_resume.py
# Prints a count per test and fails loud (non-zero exit) on any
# failure.
#
# The bug (v3). After a frontier node was expanded, build_network
# released its session file as soon as the refcount hit zero --
# but the checkpoint written at the previous layer boundary still
# listed that node. A crash mid-layer (the v3 run crashed this
# way in epoch 26, and the resume then failed with "session file
# missing for key 2868") left the checkpoint pointing at deleted
# files, and the resume died later at load_traffic. (WinError 32
# is a different failure -- the os.replace file-lock race the
# v3.1 rename retry addresses -- not the cause of that crash.)
# v3.2 pins every session file named by the latest
# checkpoint until the next checkpoint replaces it.
#
# The decisive test below simulates that crash on the toy
# network: build to a checkpoint, then load and release every
# frontier node's traffic through the real SessionStore code
# path (the same release() build_network calls after expanding
# a node), and resume from the checkpoint. Without the fix the
# releases delete the files and the resume fails; with the fix
# the files survive and the resumed run completes with results
# identical to an uninterrupted run. A second leg repeats the
# crash after a resume, proving the pin set survives a resume
# (it is persisted in the checkpoint). A fail-fast leg deletes
# one pinned file by hand and checks the resume refuses at
# once with a clear message instead of dying mid-run. A final
# leg strips "pinned" from a v3.2 checkpoint (simulating a v3
# checkpoint, which has no "pinned" entry) and resumes it,
# proving v3 checkpoints stay loadable under v3.2.
import os
import pickle
import shutil
import sys
import tempfile
import time

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
sys.path.insert(0, root)

from network_builder.checkpoint import load_checkpoint
from network_builder.demand import load_demand
from network_builder.groups import compute_groups, load_coordinates
from network_builder.network import build_network, read_bare_network
from network_builder.projects import load_projects
from network_builder.session_store import SESSIONS_DIRNAME, SessionStore

NET = os.path.join(root, "example_network")
EXAMPLE_PROJECT = os.path.join(root, "example_project.csv")
EXAMPLE_DEMAND = os.path.join(root, "example_demand.csv")
EXAMPLE_COORDS = os.path.join(root, "example_coordinates.csv")
OVERRIDE = [(("low", "low"), 0.25), (("medium", "medium"), 0.5),
            (("high", "high"), 0.25)]
START_DATE = "2026-07-06"  # first Monday in July 2026
PASSED = []


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError("FAIL %s %s" % (name, detail))
    PASSED.append(name)
    print("PASS %s %s" % (name, detail))


def example_setup():
    _nodes, link_set = read_bare_network(NET)
    projects = load_projects(EXAMPLE_PROJECT, link_set)
    demand = load_demand(EXAMPLE_DEMAND)
    node_ids, _links = read_bare_network(NET)
    coords = load_coordinates(EXAMPLE_COORDS)
    groups, group_coords = compute_groups(node_ids, coordinates=coords,
                                          k=2, seed=42)
    return projects, demand, groups, group_coords


def node_signature(network):
    return sorted(
        (node.node_id, node.epoch, node.status, node.reaching_weight,
         node.group_id, node.unfinished_count)
        for node in network.nodes.values())


def link_signature(network):
    return sorted(
        (link.link_id, link.from_node, link.to_node, link.action,
         link.branch_probability, link.link_weight,
         tuple(sorted(link.closed_set.items())), link.profile_count)
        for link in network.links)


def layer_signature(layer_stats):
    return [{k: s[k] for k in ("epoch", "nodes_at_epoch", "actions",
                               "transitions_computed", "cache_hits",
                               "merges", "nodes_total", "links_total")}
            for s in layer_stats]


def expect_value_error(name, fn, needle=None):
    try:
        fn()
    except ValueError as exc:
        check(name, needle is None or needle in str(exc),
              "ValueError: %s" % (exc,))
        return
    raise AssertionError("FAIL %s: no ValueError raised" % name)


def frontier_keys(state):
    """Traffic keys of the checkpoint's frontier nodes."""
    return sorted(r["traffic_key"] for r in state["nodes"]
                  if r["epoch"] == state["epoch"]
                  and r["traffic_key"] is not None)


def simulate_mid_layer_crash(out_dir):
    """Simulate a crash part-way through the next layer.

    Loads the checkpoint's store, then for every frontier key
    loads the session (the real load path) and releases it as
    many times as its restored refcount -- one release per
    holder node, exactly what build_network's release_traffic
    does as each frontier node is expanded. Returns the list
    of keys released. Without pinning, every key released to
    zero deletes its file; with pinning, every file survives.
    """
    ckpt = os.path.join(out_dir, "checkpoint.pkl")
    state = load_checkpoint(ckpt)
    store = SessionStore(os.path.join(out_dir, SESSIONS_DIRNAME))
    store.restore_state(state["session_store"])
    refcounts = dict(state["session_store"]["refcounts"])
    keys = frontier_keys(state)
    check("crash-frontier-nonempty", len(keys) > 0,
          "%d frontier keys at epoch %d" % (len(keys), state["epoch"]))
    for key in keys:
        # At least one frontier key must be singly referenced:
        # otherwise releasing could never delete a file and the
        # simulation would not be decisive.
        store.get(key)
        for _ in range(int(refcounts.get(str(key), refcounts.get(key, 0)))):
            store.release(key)
    check("crash-decisive", any(
        int(refcounts.get(str(key), refcounts.get(key, 0))) == 1
        for key in keys),
        "at least one frontier key was singly referenced, so "
        "release-to-zero would have deleted its file without pinning")
    missing = [key for key in keys
               if not os.path.isfile(os.path.join(
                   out_dir, SESSIONS_DIRNAME, "sess_%d.pkl" % key))]
    check("crash-files-survive", not missing,
          "all %d frontier session files still on disk after "
          "releasing every frontier reference (missing: %s)"
          % (len(keys), missing))
    return keys


def main():
    projects, demand, groups, group_coords = example_setup()
    started = time.time()

    # (1) Store-level pin semantics, no build needed. A pinned
    # key released to zero keeps its session loadable; an
    # unpinned key released to zero is deleted; a sweep never
    # deletes a pinned file but does delete an unpinned
    # zero-reference file.
    print("== crash-store-semantics")
    mem = SessionStore(None)
    pinned_key = mem.put({"session": "pinned"})
    mem.add_ref(pinned_key)
    plain_key = mem.put({"session": "plain"})
    mem.add_ref(plain_key)
    mem.set_pinned([pinned_key])
    mem.release(pinned_key)
    check("1-pinned-kept", mem.has(pinned_key)
          and mem.get(pinned_key) == {"session": "pinned"}
          and mem.file_exists(pinned_key),
          "pinned key released to zero stays loadable")
    mem.release(plain_key)
    check("1-unpinned-deleted", not mem.has(plain_key)
          and not mem.file_exists(plain_key),
          "unpinned key released to zero is deleted")
    disk_dir = tempfile.mkdtemp(prefix="crash_store_")
    disk = SessionStore(disk_dir)
    keep = disk.put({"session": "keep"})
    disk.add_ref(keep)
    drop = disk.put({"session": "drop"})
    disk.add_ref(drop)
    disk.set_pinned([keep])
    disk.release(keep)
    disk.release(drop)
    check("1-disk-pinned-kept", disk.file_exists(keep),
          "pinned disk file survives release to zero")
    check("1-disk-unpinned-deleted", not disk.file_exists(drop),
          "unpinned disk file deleted at zero references")
    # A pinned zero-reference file also survives a sweep;
    # unpin it and the sweep deletes it.
    check("1-sweep-keeps-pinned", disk.sweep_orphans() == 0
          and disk.file_exists(keep),
          "sweep deletes nothing while the file is pinned")
    disk.set_pinned([])
    check("1-sweep-after-unpin", disk.sweep_orphans() == 1
          and not disk.file_exists(keep),
          "sweep deletes the file once it is unpinned and "
          "unreferenced")

    # (2) Uninterrupted references, horizons 3 and 4.
    print("== crash-uninterrupted-references")
    out_full3 = tempfile.mkdtemp(prefix="crash_full3_")
    out_full4 = tempfile.mkdtemp(prefix="crash_full4_")
    t0 = time.time()
    net3_full = build_network(NET, projects, demand, groups,
                              START_DATE, horizon=3, output_dir=out_full3,
                              coords=group_coords, profiles=OVERRIDE)
    net4_full = build_network(NET, projects, demand, groups,
                              START_DATE, horizon=4, output_dir=out_full4,
                              coords=group_coords, profiles=OVERRIDE)
    print("    uninterrupted references took %.1fs" % (time.time() - t0))
    check("2-full3-counts", len(net3_full.nodes) == 19
          and len(net3_full.links) == 22,
          "horizon-3 reference: 19 nodes, 22 links")
    check("2-full4-counts", len(net4_full.nodes) == 29
          and len(net4_full.links) == 34,
          "horizon-4 reference: 29 nodes, 34 links")

    # (3) First crash leg: build to horizon 2, check the
    # checkpoint pins exactly its frontier keys, simulate the
    # mid-layer crash, resume to horizon 3.
    print("== crash-leg-1")
    out_crash = tempfile.mkdtemp(prefix="crash_leg1_")
    build_network(NET, projects, demand, groups, START_DATE,
                  horizon=2, output_dir=out_crash,
                  coords=group_coords, profiles=OVERRIDE)
    state2 = load_checkpoint(os.path.join(out_crash, "checkpoint.pkl"))
    check("3-pinned-in-checkpoint",
          state2["session_store"].get("pinned")
          == frontier_keys(state2),
          "checkpoint session_store carries pinned == its %d "
          "frontier keys" % len(frontier_keys(state2)))
    check("3-format-still-3", state2["format_version"] == 3,
          "FORMAT_VERSION stays 3 in v3.2")
    simulate_mid_layer_crash(out_crash)
    t0 = time.time()
    net3_resumed = build_network(
        NET, projects, demand, groups, START_DATE, horizon=3,
        output_dir=out_crash, coords=group_coords, profiles=OVERRIDE,
        resume_from=os.path.join(out_crash, "checkpoint.pkl"))
    print("    leg-1 resume took %.1fs" % (time.time() - t0))
    check("3-node-signature",
          node_signature(net3_resumed) == node_signature(net3_full),
          "resumed horizon-3 node signature identical to "
          "uninterrupted")
    check("3-link-signature",
          link_signature(net3_resumed) == link_signature(net3_full),
          "resumed horizon-3 link signature identical (same "
          "penalties per link)")
    check("3-counters",
          net3_resumed.totals["transitions_computed"]
          == net3_full.totals["transitions_computed"]
          and net3_resumed.totals["merges"]
          == net3_full.totals["merges"]
          and net3_resumed.totals["cache_hits"]
          == net3_full.totals["cache_hits"]
          and net3_resumed.totals["profile_runs"]
          == net3_full.totals["profile_runs"],
          "integer counters identical (computed %d, merges %d, "
          "hits %d, profile runs %d)"
          % (net3_resumed.totals["transitions_computed"],
             net3_resumed.totals["merges"],
             net3_resumed.totals["cache_hits"],
             net3_resumed.totals["profile_runs"]))
    check("3-layer-stats",
          layer_signature(net3_resumed.layer_stats)
          == layer_signature(net3_full.layer_stats),
          "per-layer stats identical (except wall-clock seconds)")
    check("3-sessions-bounded",
          sorted(os.listdir(os.path.join(out_crash, SESSIONS_DIRNAME)))
          == sorted(os.listdir(os.path.join(out_full3,
                                            SESSIONS_DIRNAME))),
          "sessions/ file set after the resumed run is identical "
          "to the uninterrupted run: the swept previous-frontier "
          "files are gone, disk stays bounded")
    state3 = load_checkpoint(os.path.join(out_crash, "checkpoint.pkl"))
    check("3-new-pins",
          state3["epoch"] == 3
          and state3["session_store"].get("pinned")
          == frontier_keys(state3),
          "epoch-3 checkpoint pins exactly its new frontier keys")

    # (4) Fail fast: copy the epoch-3 run aside, delete one
    # pinned session file by hand, and check the resume
    # refuses at once instead of dying mid-run at load_traffic.
    print("== crash-fail-fast")
    out_copy = tempfile.mkdtemp(prefix="crash_copy_")
    shutil.rmtree(out_copy)
    shutil.copytree(out_crash, out_copy)
    victim = frontier_keys(state3)[0]
    os.unlink(os.path.join(out_copy, SESSIONS_DIRNAME,
                           "sess_%d.pkl" % victim))
    expect_value_error(
        "4-missing-pinned-file",
        lambda: build_network(
            NET, projects, demand, groups, START_DATE, horizon=4,
            output_dir=out_copy, coords=group_coords,
            profiles=OVERRIDE,
            resume_from=os.path.join(out_copy, "checkpoint.pkl")),
        needle="missing")

    # (5) Second crash leg: crash again on the epoch-3
    # checkpoint frontier (pins must have survived the leg-1
    # resume, because they are persisted in the checkpoint),
    # then resume to horizon 4 and compare with the
    # uninterrupted run.
    print("== crash-leg-2")
    simulate_mid_layer_crash(out_crash)
    t0 = time.time()
    net4_resumed = build_network(
        NET, projects, demand, groups, START_DATE, horizon=4,
        output_dir=out_crash, coords=group_coords, profiles=OVERRIDE,
        resume_from=os.path.join(out_crash, "checkpoint.pkl"))
    print("    leg-2 resume took %.1fs" % (time.time() - t0))
    check("5-node-signature",
          node_signature(net4_resumed) == node_signature(net4_full),
          "resumed horizon-4 node signature identical to "
          "uninterrupted after a second mid-layer crash")
    check("5-link-signature",
          link_signature(net4_resumed) == link_signature(net4_full),
          "resumed horizon-4 link signature identical (same "
          "penalties per link)")
    check("5-counts", len(net4_resumed.nodes) == 29
          and len(net4_resumed.links) == 34,
          "29 nodes, 34 links after the second crash + resume")
    check("5-sessions-bounded",
          sorted(os.listdir(os.path.join(out_crash, SESSIONS_DIRNAME)))
          == sorted(os.listdir(os.path.join(out_full4,
                                            SESSIONS_DIRNAME))),
          "sessions/ file set after the second resume matches "
          "the uninterrupted run")

    # (7) v3 compatibility: a v3 checkpoint's session_store
    # has no "pinned" entry (pinning is new in v3.2). Strip
    # "pinned" from a fresh v3.2 checkpoint and resume it to
    # horizon 3: the resume must succeed and match the
    # uninterrupted horizon-3 run exactly, and the new
    # checkpoint must pin its own frontier keys again.
    print("== crash-v3-compat")
    out_v3 = tempfile.mkdtemp(prefix="crash_v3_")
    build_network(NET, projects, demand, groups, START_DATE,
                  horizon=2, output_dir=out_v3,
                  coords=group_coords, profiles=OVERRIDE)
    ckpt_v3 = os.path.join(out_v3, "checkpoint.pkl")
    state_v3 = load_checkpoint(ckpt_v3)
    check("7-v3-has-pinned",
          "pinned" in state_v3["session_store"],
          "v3.2 checkpoint pins its %d frontier keys before "
          "stripping" % len(frontier_keys(state_v3)))
    del state_v3["session_store"]["pinned"]
    with open(ckpt_v3, "wb") as fh:
        pickle.dump(state_v3, fh)
    check("7-v3-stripped",
          "pinned" not in load_checkpoint(ckpt_v3)["session_store"],
          "stripped checkpoint carries no pinned entry, like a "
          "v3 one")
    t0 = time.time()
    net3_v3 = build_network(
        NET, projects, demand, groups, START_DATE, horizon=3,
        output_dir=out_v3, coords=group_coords, profiles=OVERRIDE,
        resume_from=ckpt_v3)
    print("    v3-compat resume took %.1fs" % (time.time() - t0))
    check("7-node-signature",
          node_signature(net3_v3) == node_signature(net3_full),
          "v3 checkpoint resumed to horizon 3: node signature "
          "identical to uninterrupted")
    check("7-link-signature",
          link_signature(net3_v3) == link_signature(net3_full),
          "v3 checkpoint resumed to horizon 3: link signature "
          "identical (same penalties per link)")
    check("7-counts",
          len(net3_v3.nodes) == 19 and len(net3_v3.links) == 22
          and net3_v3.totals["transitions_computed"]
          == net3_full.totals["transitions_computed"]
          and net3_v3.totals["merges"]
          == net3_full.totals["merges"]
          and net3_v3.totals["cache_hits"]
          == net3_full.totals["cache_hits"]
          and net3_v3.totals["profile_runs"]
          == net3_full.totals["profile_runs"],
          "19 nodes, 22 links, integer counters identical to "
          "uninterrupted (computed %d, merges %d, hits %d, "
          "profile runs %d)"
          % (net3_v3.totals["transitions_computed"],
             net3_v3.totals["merges"],
             net3_v3.totals["cache_hits"],
             net3_v3.totals["profile_runs"]))
    state3_v3 = load_checkpoint(ckpt_v3)
    check("7-new-pins",
          state3_v3["epoch"] == 3
          and state3_v3["session_store"].get("pinned")
          == frontier_keys(state3_v3),
          "epoch-3 checkpoint written by the v3-compat resume "
          "pins exactly its new frontier keys")

    # (6) Fresh-run housekeeping: a fresh run in a dir with a
    # stale checkpoint and stale sessions/ from a crashed run
    # renames the checkpoint aside and deletes the stale
    # session files (keys restart at 1, so old files must not
    # be mistaken for this run's).
    print("== crash-fresh-housekeeping")
    out_fresh = tempfile.mkdtemp(prefix="crash_fresh_")
    stale_marker = {"marker": "crashed run checkpoint"}
    with open(os.path.join(out_fresh, "checkpoint.pkl"), "wb") as fh:
        pickle.dump(stale_marker, fh)
    stale_sessions = os.path.join(out_fresh, SESSIONS_DIRNAME)
    os.makedirs(stale_sessions)
    # An out-of-range key: the new run reuses keys from 1, so a
    # stale sess_1.pkl would be overwritten anyway and prove
    # nothing; sess_999.pkl can only disappear if the fresh run
    # clears the stale sessions/ directory.
    with open(os.path.join(stale_sessions, "sess_999.pkl"), "wb") as fh:
        fh.write(b"stale session from the crashed run")
    build_network(NET, projects, demand, groups, START_DATE,
                  horizon=1, output_dir=out_fresh,
                  coords=group_coords, profiles=OVERRIDE)
    with open(os.path.join(out_fresh, "checkpoint.stale.pkl"),
              "rb") as fh:
        kept = pickle.load(fh)
    check("6-stale-checkpoint-renamed", kept == stale_marker
          and load_checkpoint(
              os.path.join(out_fresh, "checkpoint.pkl"))["epoch"] == 1,
          "old checkpoint.pkl preserved as checkpoint.stale.pkl; "
          "new checkpoint at epoch 1")
    check("6-stale-sessions-deleted",
          "sess_999.pkl" not in os.listdir(stale_sessions)
          and os.path.isfile(os.path.join(
              stale_sessions,
              "sess_%d.pkl" % frontier_keys(load_checkpoint(
                  os.path.join(out_fresh, "checkpoint.pkl")))[0])),
          "stale sess_999.pkl from the crashed run is gone; "
          "sessions/ holds only this run's files")

    print("ALL CRASH-RESUME TESTS PASSED: %d checks in %.1fs"
          % (len(PASSED), time.time() - started))


if __name__ == "__main__":
    main()
