# Tests for network_builder v2 resume / checkpointing. Run from the
# extraction directory:
#   python tests/test_resume.py
# Prints a count per test and fails loud (non-zero exit) on any
# failure. Timings are printed so the note can quote real numbers.
#
# Most equivalence tests use the toy 4-node network with a
# 3-profile override (low/medium/high at 0.25/0.5/0.25) instead
# of the full 9-profile k=2 enumeration. The override changes
# only the demand profiles averaged inside each transition; the
# resume mechanics under test (restored nodes/links/counters,
# recomputed transitions, merging, reaching weights) are
# identical, and the override keeps the suite at ~6 minutes
# instead of ~10. Both sides of every comparison use the same
# override, so the comparison itself is exact. One test
# (1b) repeats the equivalence with the full 9-profile
# enumeration, so the full-profile case is pinned too.
import csv
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
from network_builder.cli import main as cli_main
from network_builder.demand import load_demand
from network_builder.groups import compute_groups, load_coordinates
from network_builder.network import build_network, read_bare_network
from network_builder.projects import load_projects

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


def write_tmp(content, suffix=".csv"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def example_setup():
    _nodes, link_set = read_bare_network(NET)
    projects = load_projects(EXAMPLE_PROJECT, link_set)
    demand = load_demand(EXAMPLE_DEMAND)
    node_ids, _links = read_bare_network(NET)
    coords = load_coordinates(EXAMPLE_COORDS)
    groups, group_coords = compute_groups(node_ids, coordinates=coords,
                                          k=2, seed=42)
    return projects, demand, groups, group_coords


def expect_value_error(name, fn):
    try:
        fn()
    except ValueError as exc:
        check(name, True, "ValueError: %s" % (exc,))
        return
    raise AssertionError("FAIL %s: no ValueError raised" % name)


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


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    projects, demand, groups, group_coords = example_setup()
    started = time.time()

    # (5a) A horizon-2 run writes exactly one checkpoint.pkl and
    # no v1 frontier pickles.
    out_a = tempfile.mkdtemp(prefix="resume_a_")
    out_b = tempfile.mkdtemp(prefix="resume_b_")
    print("== resume-horizon-2 (out %s)" % out_a)
    net2 = build_network(NET, projects, demand, groups, START_DATE,
                         horizon=2, output_dir=out_a, coords=group_coords,
                         profiles=OVERRIDE)
    ckpt_a = os.path.join(out_a, "checkpoint.pkl")
    check("5-checkpoint-exists", os.path.isfile(ckpt_a),
          "checkpoint.pkl present after a horizon-2 run")
    state2 = load_checkpoint(ckpt_a)
    check("5-checkpoint-epoch", state2["epoch"] == 2
          and state2["horizon"] == 2,
          "checkpoint epoch 2, horizon 2")
    check("5-single-file", len([f for f in os.listdir(out_a)
                                if f.endswith(".pkl")]) == 1
          and not [f for f in os.listdir(out_a)
                   if f.startswith("frontier_epoch_")],
          "one .pkl only, no frontier_epoch_*.pkl")
    ckpt_stat_before = os.stat(ckpt_a)
    print("    horizon-2 run took %.1fs, checkpoint %.2f MB"
          % (time.time() - started, ckpt_stat_before.st_size / 1e6))
    frontier_records = [r for r in state2["nodes"] if r["epoch"] == 2]
    old_records = [r for r in state2["nodes"] if r["epoch"] < 2]
    check("5-frontier-only-sessions",
          len(state2["nodes"]) == 11 and len(frontier_records) == 6
          and all(r["traffic_key"] is not None for r in frontier_records)
          and all(r["traffic_key"] is None for r in old_records)
          and state2["format_version"] == 3
          and "session_store" in state2,
          "11 nodes stored, 6 frontier session keys kept, 5 older "
          "nodes traffic_key=None, format v3 with a session store")
    check("5-sessions-dir",
          os.path.isdir(os.path.join(out_a, "sessions"))
          and [f for f in os.listdir(os.path.join(out_a, "sessions"))
               if f.startswith("sess_") and f.endswith(".pkl")],
          "sessions/ holds the spilled frontier sessions")

    # (1) Uninterrupted horizon-4 reference.
    print("== resume-uninterrupted-horizon-4 (out %s)" % out_b)
    t0 = time.time()
    net4_full = build_network(NET, projects, demand, groups,
                              START_DATE, horizon=4, output_dir=out_b,
                              coords=group_coords, profiles=OVERRIDE)
    print("    uninterrupted horizon-4 took %.1fs" % (time.time() - t0))
    check("1-full-counts", len(net4_full.nodes) == 29
          and len(net4_full.links) == 34,
          "29 nodes, 34 links")
    check("1-full-computed", net4_full.totals["transitions_computed"] == 30
          and net4_full.totals["merges"] == 6
          and net4_full.totals["cache_hits"] == 0,
          "30 computed, 6 merges, 0 cache hits")

    # (1) Resume the horizon-2 checkpoint to horizon 4, in the
    # same output dir (the natural resume flow).
    print("== resume-horizon-2-to-4")
    t0 = time.time()
    net4_resumed = build_network(
        NET, projects, demand, groups, START_DATE, horizon=4,
        output_dir=out_a, coords=group_coords, profiles=OVERRIDE,
        resume_from=ckpt_a)
    print("    resumed layers 2-3 took %.1fs" % (time.time() - t0))
    check("1-node-count", len(net4_resumed.nodes) == len(net4_full.nodes)
          == 29, "29 nodes on both sides")
    check("1-link-count", len(net4_resumed.links) == len(net4_full.links)
          == 34, "34 links on both sides")
    check("1-node-signature",
          node_signature(net4_resumed) == node_signature(net4_full),
          "node id/epoch/status/reaching/group/unfinished identical")
    check("1-link-signature",
          link_signature(net4_resumed) == link_signature(net4_full),
          "link id/from/to/action/branch-prob/weight/closed-set "
          "identical (same penalties per link)")
    check("1-terminal-unfinished",
          sorted((n.node_id, n.unfinished_count)
                 for n in net4_resumed.terminal_nodes())
          == sorted((n.node_id, n.unfinished_count)
                    for n in net4_full.terminal_nodes()),
          "terminal unfinished counts identical")
    first_two = sum(s["transitions_computed"]
                    for s in net4_full.layer_stats[:2])
    # The resumed run restores the first 2 layers' totals from
    # the checkpoint, so its CUMULATIVE counter ends identical
    # to the uninterrupted run (30 == 30); the work actually
    # done DURING the resumed run is the cumulative total minus
    # the restored checkpoint total, and that equals the
    # uninterrupted total minus the first 2 layers (20).
    resumed_increment = (net4_resumed.totals["transitions_computed"]
                         - net2.totals["transitions_computed"])
    check("1-transitions-computed",
          net4_resumed.totals["transitions_computed"]
          == net4_full.totals["transitions_computed"] == 30
          and resumed_increment
          == net4_full.totals["transitions_computed"] - first_two,
          "cumulative computed %d on both sides; resumed-run "
          "increment %d = uninterrupted %d - first-2-layers %d"
          % (net4_resumed.totals["transitions_computed"],
             resumed_increment,
             net4_full.totals["transitions_computed"], first_two))
    check("1-profile-runs",
          net4_resumed.totals["profile_runs"]
          == net4_full.totals["profile_runs"],
          "profile runs identical (%d)"
          % net4_resumed.totals["profile_runs"])
    check("1-merges", net4_resumed.totals["merges"]
          == net4_full.totals["merges"] == 6, "6 merges on both sides")
    check("1-cache-hits", net4_resumed.totals["cache_hits"]
          == net4_full.totals["cache_hits"] == 0,
          "0 cache hits on both sides")
    check("1-layer-stats",
          [{k: s[k] for k in ("epoch", "nodes_at_epoch", "actions",
                              "transitions_computed", "cache_hits",
                              "merges", "nodes_total", "links_total")}
           for s in net4_resumed.layer_stats]
          == [{k: s[k] for k in ("epoch", "nodes_at_epoch", "actions",
                                 "transitions_computed", "cache_hits",
                                 "merges", "nodes_total", "links_total")}
              for s in net4_full.layer_stats],
          "per-layer stats identical (except wall-clock seconds)")
    check("1-csvs-identical",
          read_csv_rows(os.path.join(out_a, "nodes.csv"))
          == read_csv_rows(os.path.join(out_b, "nodes.csv"))
          and read_csv_rows(os.path.join(out_a, "links.csv"))
          == read_csv_rows(os.path.join(out_b, "links.csv")),
          "nodes.csv and links.csv identical between resumed and "
          "uninterrupted runs")
    # Group ids agree on this toy network because it has no
    # cache hits in the resumed layers, so the recomputed
    # transitions take group ids in the same order. In general
    # they may differ (see checkpoint.py); group_id is only a
    # cache key / label, never part of node identity.
    check("1-group-ids",
          [n.group_id for n in sorted(net4_resumed.nodes.values(),
                                      key=lambda n: n.node_id)]
          == [n.group_id for n in sorted(net4_full.nodes.values(),
                                         key=lambda n: n.node_id)],
          "group ids identical on this toy network (no resumed-"
          "layer cache hits)")

    # (5b) The checkpoint was overwritten in place: still one
    # file, now at epoch 4, newer/larger-or-equal than before.
    state4 = load_checkpoint(ckpt_a)
    ckpt_stat_after = os.stat(ckpt_a)
    check("5-checkpoint-overwritten", state4["epoch"] == 4
          and len([f for f in os.listdir(out_a)
                   if f.endswith(".pkl")]) == 1
          and ckpt_stat_after.st_mtime >= ckpt_stat_before.st_mtime,
          "single checkpoint.pkl now at epoch 4, %.2f MB"
          % (ckpt_stat_after.st_size / 1e6))
    check("5-frontier-only-after-resume",
          all(r["traffic_key"] is not None for r in state4["nodes"]
              if r["epoch"] == 4)
          and all(r["traffic_key"] is None for r in state4["nodes"]
                  if r["epoch"] < 4),
          "epoch-4 checkpoint keeps frontier session keys only")
    check("1-resumed-memory-traffic",
          all(n.traffic is None and n.traffic_key is None
              for n in net4_resumed.nodes.values() if n.epoch < 4)
          and all(n.traffic is None and n.traffic_key is not None
                  for n in net4_resumed.nodes.values() if n.epoch == 4),
          "in memory after resume: epochs 0-3 hold no session "
          "(keys released on expansion), epoch-4 frontier holds "
          "keys only until loaded")
    check("1-resumed-traffic-loads",
          all(net4_resumed.traffic_of(n) is not None
              for n in net4_resumed.nodes.values() if n.epoch == 4),
          "every epoch-4 frontier session loads from the store")
    check("5-no-stale-after-resume",
          not os.path.isfile(os.path.join(out_a, "checkpoint.stale.pkl")),
          "a resume does not rename the checkpoint aside")

    # (1b) The same equivalence as (1), but with the FULL
    # 9-profile k=2 enumeration (no override), to horizon 2.
    # The (1) comparison uses the 3-profile override for
    # speed; this one pins the full-profile case.
    print("== resume-full-9-profile-horizon-2")
    out_1b_a = tempfile.mkdtemp(prefix="resume_1b_a_")
    out_1b_b = tempfile.mkdtemp(prefix="resume_1b_b_")
    t0 = time.time()
    net1b_1 = build_network(NET, projects, demand, groups, START_DATE,
                            horizon=1, output_dir=out_1b_a,
                            coords=group_coords)
    net1b_full = build_network(NET, projects, demand, groups, START_DATE,
                               horizon=2, output_dir=out_1b_b,
                               coords=group_coords)
    net1b_resumed = build_network(
        NET, projects, demand, groups, START_DATE, horizon=2,
        output_dir=out_1b_a, coords=group_coords,
        resume_from=os.path.join(out_1b_a, "checkpoint.pkl"))
    print("    full-9-profile horizon-2 took %.1fs"
          % (time.time() - t0))
    check("1b-node-signature",
          node_signature(net1b_resumed) == node_signature(net1b_full),
          "node id/epoch/status/reaching/group/unfinished identical")
    check("1b-link-signature",
          link_signature(net1b_resumed) == link_signature(net1b_full),
          "link id/from/to/action/branch-prob/weight/closed-set "
          "identical (same penalties per link)")
    check("1b-counts", len(net1b_resumed.nodes) == 11
          and len(net1b_resumed.links) == 12
          and net1b_resumed.totals["transitions_computed"] == 10
          and net1b_resumed.totals["cache_hits"] == 0
          and net1b_resumed.totals["merges"] == 2
          and net1b_resumed.totals["profile_runs"] == 90,
          "11 nodes, 12 links, 10 computed, 0 hits, 2 merges, "
          "90 profile runs (10 x 9)")
    check("1b-csvs-identical",
          read_csv_rows(os.path.join(out_1b_a, "nodes.csv"))
          == read_csv_rows(os.path.join(out_1b_b, "nodes.csv"))
          and read_csv_rows(os.path.join(out_1b_a, "links.csv"))
          == read_csv_rows(os.path.join(out_1b_b, "links.csv")),
          "nodes.csv and links.csv identical between resumed and "
          "uninterrupted runs")

    # (6) Resume with cache hits in the resumed layer: v1
    # test 9's 5-duration large-project setup. Resume after
    # epoch 1 to horizon 2 and compare against an
    # uninterrupted horizon-2 run. The NETWORK must be
    # identical (nodes, links, penalties). The integer
    # reporting counters are identical too in this setup:
    # a resumed layer can only hit a cache entry from a
    # completed layer if the same (group_id, closed set,
    # day, hour) recurs across layers, and the (day, hour)
    # pair recurs only every 28 epochs. What DOES shift is
    # the wall-clock reporting: transition_seconds and the
    # per-entry seconds in transition_log always differ
    # between two runs.
    print("== resume-cache-hit-5-duration")
    _nodes, link_set9 = read_bare_network(NET)
    text9 = ("project_id,project_type,from_node_id,to_node_id,"
             "duration_epochs,probability\n")
    for duration in (28, 29, 30, 31, 32):
        text9 += "L,large,1,2,%d,0.2\n" % duration
    projects9 = load_projects(write_tmp(text9), link_set9)
    out_6a = tempfile.mkdtemp(prefix="resume_6a_")
    out_6b = tempfile.mkdtemp(prefix="resume_6b_")
    t0 = time.time()
    net9_1 = build_network(NET, projects9, demand, groups, START_DATE,
                           horizon=1, output_dir=out_6a,
                           coords=group_coords, profiles=OVERRIDE)
    net9_full = build_network(NET, projects9, demand, groups, START_DATE,
                              horizon=2, output_dir=out_6b,
                              coords=group_coords, profiles=OVERRIDE)
    net9_resumed = build_network(
        NET, projects9, demand, groups, START_DATE, horizon=2,
        output_dir=out_6a, coords=group_coords, profiles=OVERRIDE,
        resume_from=os.path.join(out_6a, "checkpoint.pkl"))
    print("    cache-hit setup took %.1fs" % (time.time() - t0))
    check("6-network-identical",
          node_signature(net9_resumed) == node_signature(net9_full)
          and link_signature(net9_resumed) == link_signature(net9_full)
          and len(net9_resumed.nodes) == len(net9_full.nodes) == 14
          and len(net9_resumed.links) == len(net9_full.links) == 17,
          "14 nodes, 17 links, identical node/link signatures "
          "(same penalties per link)")
    check("6-full-counters",
          net9_full.totals["transitions_computed"] == 5
          and net9_full.totals["cache_hits"] == 4
          and net9_full.totals["profile_runs"] == 15
          and net9_full.totals["merges"] == 4,
          "uninterrupted: 5 computed, 4 cache hits, 15 profile "
          "runs, 4 merges")
    check("6-integer-counters",
          net9_resumed.totals["transitions_computed"]
          == net9_full.totals["transitions_computed"] == 5
          and net9_resumed.totals["cache_hits"]
          == net9_full.totals["cache_hits"] == 4
          and net9_resumed.totals["profile_runs"]
          == net9_full.totals["profile_runs"] == 15
          and net9_resumed.totals["merges"]
          == net9_full.totals["merges"] == 4,
          "integer counters identical on both sides (no "
          "cross-layer cache hit is possible here)")
    check("6-timing-counters-shift",
          net9_resumed.totals["transition_seconds"]
          != net9_full.totals["transition_seconds"]
          and [round(e["seconds"], 3)
               for e in net9_resumed.totals["transition_log"]]
          != [round(e["seconds"], 3)
              for e in net9_full.totals["transition_log"]]
          and len(net9_resumed.totals["transition_log"])
          == len(net9_full.totals["transition_log"]) == 5,
          "transition_seconds %.3f vs %.3f; per-entry log "
          "seconds differ, same 5 entries"
          % (net9_resumed.totals["transition_seconds"],
             net9_full.totals["transition_seconds"]))

    # (2) Fingerprint mismatch fails loud. The checkpoint in
    # out_a is now at epoch 4, so resume to horizon 6 (a valid
    # extension) with one input changed at a time.
    print("== resume-fingerprint-mismatch")
    expect_value_error(
        "2-mismatch-start-date",
        lambda: build_network(
            NET, projects, demand, groups, "2026-07-07", horizon=6,
            output_dir=tempfile.mkdtemp(), coords=group_coords,
            profiles=OVERRIDE, resume_from=ckpt_a))
    demand_changed = dict(demand)
    demand_changed[(1, 2, 0, 2)] = 5.0
    expect_value_error(
        "2-mismatch-demand",
        lambda: build_network(
            NET, projects, demand_changed, groups, START_DATE,
            horizon=6, output_dir=tempfile.mkdtemp(),
            coords=group_coords, profiles=OVERRIDE,
            resume_from=ckpt_a))
    _nodes, link_set = read_bare_network(NET)
    projects_changed = load_projects(write_tmp(
        "project_id,project_type,from_node_id,to_node_id,"
        "duration_epochs,probability\n"
        "S,small,1,2,1,0.6\nS,small,1,2,2,0.4\n"
        "M,medium,1,3,4,1.0\n"), link_set)
    expect_value_error(
        "2-mismatch-projects",
        lambda: build_network(
            NET, projects_changed, demand, groups, START_DATE,
            horizon=6, output_dir=tempfile.mkdtemp(),
            coords=group_coords, profiles=OVERRIDE,
            resume_from=ckpt_a))
    # Network topology change: same projects/demand, but a
    # copied network dir with one link (4->2) removed. The
    # fingerprint hashes the full input_link.csv bytes, so
    # the removal fails resume loud.
    net_topo = tempfile.mkdtemp(prefix="resume_nettopo_")
    shutil.copy(os.path.join(NET, "input_node.csv"),
                os.path.join(net_topo, "input_node.csv"))
    with open(os.path.join(NET, "input_link.csv"), "rb") as fh:
        link_bytes = fh.read()
    link_bytes = link_bytes.replace(
        b"4,2,5,2,60,1800,180,12\r\n", b"")
    with open(os.path.join(net_topo, "input_link.csv"), "wb") as fh:
        fh.write(link_bytes)
    expect_value_error(
        "2-mismatch-topology",
        lambda: build_network(
            net_topo, projects, demand, groups, START_DATE,
            horizon=6, output_dir=tempfile.mkdtemp(),
            coords=group_coords, profiles=OVERRIDE,
            resume_from=ckpt_a))
    # Prune-flag change: same inputs, only
    # prune_unfinishable_starts flipped.
    expect_value_error(
        "2-mismatch-prune-flag",
        lambda: build_network(
            NET, projects, demand, groups, START_DATE, horizon=6,
            output_dir=tempfile.mkdtemp(), coords=group_coords,
            profiles=OVERRIDE, resume_from=ckpt_a,
            prune_unfinishable_starts=True))

    # (3) Missing / corrupt checkpoint fails loud.
    print("== resume-missing-corrupt")
    expect_value_error(
        "3-missing",
        lambda: build_network(
            NET, projects, demand, groups, START_DATE, horizon=6,
            output_dir=tempfile.mkdtemp(), coords=group_coords,
            profiles=OVERRIDE,
            resume_from=os.path.join(out_a, "no_such.pkl")))
    garbage = write_tmp("this is not a pickle", suffix=".pkl")
    expect_value_error(
        "3-corrupt-bytes",
        lambda: build_network(
            NET, projects, demand, groups, START_DATE, horizon=6,
            output_dir=tempfile.mkdtemp(), coords=group_coords,
            profiles=OVERRIDE, resume_from=garbage))
    wrong_shape = write_tmp("", suffix=".pkl")
    with open(wrong_shape, "wb") as fh:
        pickle.dump({"epoch": 2}, fh)
    expect_value_error(
        "3-corrupt-shape",
        lambda: build_network(
            NET, projects, demand, groups, START_DATE, horizon=6,
            output_dir=tempfile.mkdtemp(), coords=group_coords,
            profiles=OVERRIDE, resume_from=wrong_shape))

    # (4) Checkpoint epoch >= horizon fails loud.
    print("== resume-epoch-at-horizon")
    expect_value_error(
        "4-epoch-eq-horizon",
        lambda: build_network(
            NET, projects, demand, groups, START_DATE, horizon=4,
            output_dir=tempfile.mkdtemp(), coords=group_coords,
            profiles=OVERRIDE, resume_from=ckpt_a))
    expect_value_error(
        "4-epoch-gt-horizon",
        lambda: build_network(
            NET, projects, demand, groups, START_DATE, horizon=3,
            output_dir=tempfile.mkdtemp(), coords=group_coords,
            profiles=OVERRIDE, resume_from=ckpt_a))

    # (8) CLI file-bytes fingerprint: a changed link ATTRIBUTE
    # (length 5 -> 6 on link 1->2, same topology) must fail a
    # CLI resume loud, and moving an unchanged network to a
    # new path must resume fine (identity is file bytes, not
    # the path string).
    print("== resume-cli-file-bytes")
    out_8 = tempfile.mkdtemp(prefix="resume_8_")
    cli_main(["--network-dir", NET,
              "--projects", EXAMPLE_PROJECT,
              "--demand", EXAMPLE_DEMAND,
              "--coordinates", EXAMPLE_COORDS,
              "--k", "2", "--start-date", START_DATE,
              "--horizon", "1", "--output-dir", out_8])
    ckpt_8 = os.path.join(out_8, "checkpoint.pkl")
    check("8-cli-stage1", os.path.isfile(ckpt_8)
          and load_checkpoint(ckpt_8)["epoch"] == 1,
          "CLI horizon-1 checkpoint written")
    net_attr = tempfile.mkdtemp(prefix="resume_netattr_")
    shutil.copy(os.path.join(NET, "input_node.csv"),
                os.path.join(net_attr, "input_node.csv"))
    with open(os.path.join(NET, "input_link.csv"), "rb") as fh:
        attr_bytes = fh.read()
    attr_bytes = attr_bytes.replace(b"1,2,5,2,60,1800,180,12",
                                    b"1,2,6,2,60,1800,180,12")
    with open(os.path.join(net_attr, "input_link.csv"), "wb") as fh:
        fh.write(attr_bytes)
    expect_value_error(
        "8-cli-link-attribute",
        lambda: cli_main(["--network-dir", net_attr,
                          "--projects", EXAMPLE_PROJECT,
                          "--demand", EXAMPLE_DEMAND,
                          "--coordinates", EXAMPLE_COORDS,
                          "--k", "2", "--start-date", START_DATE,
                          "--horizon", "2", "--output-dir",
                          tempfile.mkdtemp(),
                          "--resume-from", ckpt_8]))
    net_moved = tempfile.mkdtemp(prefix="resume_netmoved_")
    shutil.copy(os.path.join(NET, "input_node.csv"),
                os.path.join(net_moved, "input_node.csv"))
    shutil.copy(os.path.join(NET, "input_link.csv"),
                os.path.join(net_moved, "input_link.csv"))
    out_8b = tempfile.mkdtemp(prefix="resume_8b_")
    cli_main(["--network-dir", net_moved,
              "--projects", EXAMPLE_PROJECT,
              "--demand", EXAMPLE_DEMAND,
              "--coordinates", EXAMPLE_COORDS,
              "--k", "2", "--start-date", START_DATE,
              "--horizon", "2", "--output-dir", out_8b,
              "--resume-from", ckpt_8])
    check("8-cli-path-move",
          len(read_csv_rows(os.path.join(out_8b, "nodes.csv"))) == 11
          and len(read_csv_rows(os.path.join(out_8b, "links.csv"))) == 12,
          "unchanged network at a new path resumes fine: "
          "11 nodes, 12 links")

    # (9) Malformed checkpoints fail loud in load_checkpoint,
    # and a next_group_id that does not exceed the restored
    # group ids fails loud in restore. The crafted states
    # keep the original fingerprint, so the failure is the
    # structural check, not a fingerprint mismatch.
    print("== resume-malformed-checkpoint")
    state4 = load_checkpoint(ckpt_a)
    bad_totals = dict(state4)
    bad_totals["totals"] = dict(state4["totals"])
    del bad_totals["totals"]["merges"]
    bad_path = write_tmp("", suffix=".pkl")
    with open(bad_path, "wb") as fh:
        pickle.dump(bad_totals, fh)
    expect_value_error("9-totals-missing-key",
                       lambda: load_checkpoint(bad_path))
    bad_totals2 = dict(state4)
    bad_totals2["totals"] = ["not", "a", "dict"]
    bad_path2 = write_tmp("", suffix=".pkl")
    with open(bad_path2, "wb") as fh:
        pickle.dump(bad_totals2, fh)
    expect_value_error("9-totals-not-dict",
                       lambda: load_checkpoint(bad_path2))
    bad_dup = dict(state4)
    bad_dup["nodes"] = list(state4["nodes"]) + [dict(state4["nodes"][0])]
    bad_path3 = write_tmp("", suffix=".pkl")
    with open(bad_path3, "wb") as fh:
        pickle.dump(bad_dup, fh)
    expect_value_error("9-duplicate-node-key",
                       lambda: load_checkpoint(bad_path3))
    max_group = max(n["group_id"] for n in state4["nodes"])
    bad_group = dict(state4)
    bad_group["next_group_id"] = max_group  # must EXCEED it
    bad_path4 = write_tmp("", suffix=".pkl")
    with open(bad_path4, "wb") as fh:
        pickle.dump(bad_group, fh)
    expect_value_error(
        "9-next-group-id",
        lambda: build_network(
            NET, projects, demand, groups, START_DATE, horizon=6,
            output_dir=tempfile.mkdtemp(), coords=group_coords,
            profiles=OVERRIDE, resume_from=bad_path4))

    # (9b) A v2 checkpoint (format_version 2) fails loud: v2
    # sessions carry MOE history and embed sessions inline, so
    # they can never be resumed into v3 -- fresh restart only.
    print("== resume-v2-checkpoint-rejected")
    state_v2 = dict(state4)
    state_v2["format_version"] = 2
    bad_v2_path = write_tmp("", suffix=".pkl")
    with open(bad_v2_path, "wb") as fh:
        pickle.dump(state_v2, fh)
    expect_value_error("9b-v2-format",
                       lambda: load_checkpoint(bad_v2_path))

    # (10) A fresh run renames a pre-existing checkpoint.pkl
    # aside (checkpoint.stale.pkl) and deletes orphan tmp /
    # v1 frontier files at run start.
    print("== resume-stale-checkpoint")
    out_10 = tempfile.mkdtemp(prefix="resume_10_")
    stale_marker = {"marker": "previous run checkpoint"}
    with open(os.path.join(out_10, "checkpoint.pkl"), "wb") as fh:
        pickle.dump(stale_marker, fh)
    with open(os.path.join(out_10, "checkpoint.abc123.tmp"),
              "wb") as fh:
        fh.write(b"partial write")
    with open(os.path.join(out_10, "frontier_epoch_0.pkl"),
              "wb") as fh:
        fh.write(b"v1 frontier")
    build_network(NET, projects, demand, groups, START_DATE,
                  horizon=1, output_dir=out_10,
                  coords=group_coords, profiles=OVERRIDE)
    with open(os.path.join(out_10, "checkpoint.stale.pkl"),
              "rb") as fh:
        kept = pickle.load(fh)
    check("10-stale-renamed", kept == stale_marker
          and load_checkpoint(
              os.path.join(out_10, "checkpoint.pkl"))["epoch"] == 1,
          "old checkpoint.pkl preserved as checkpoint.stale.pkl; "
          "new checkpoint at epoch 1")
    check("10-orphans-cleaned",
          not [f for f in os.listdir(out_10) if f.endswith(".tmp")]
          and not [f for f in os.listdir(out_10)
                   if f.startswith("frontier_epoch_")],
          "orphan tmp and v1 frontier files deleted at run start")

    print("ALL RESUME TESTS PASSED: %d checks in %.1fs"
          % (len(PASSED), time.time() - started))


if __name__ == "__main__":
    main()
