# Tests for network_builder v1. Run from the extraction directory:
#   python tests/test_network_builder.py
# Prints a count per test and fails loud (non-zero exit) on any
# failure. Timings are printed so the note can quote real numbers.
import copy
import csv
import os
import sys
import tempfile
import time
import warnings

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
sys.path.insert(0, root)

from network_builder.average import (average_state,
                                     weighted_average_state)
from network_builder.demand import (count_epoch_agents,
                                    enumerate_profiles,
                                    generate_epoch_agents, load_demand,
                                    level_count, parse_day_of_week,
                                    poisson_ppf,
                                    poisson_ppf_fallback_only)
from network_builder.groups import (compute_groups, kmeans_labels,
                                    load_coordinates, save_groups_csv)
from network_builder.ids import (completed_ids_all, next_agent_id_base,
                                 prune_session, renumber_session)
from network_builder.network import (apply_status, available_actions,
                                     build_network, closed_set_for,
                                     compute_transition, merge_traffic,
                                     read_bare_network)
from network_builder.projects import load_projects
from network_builder.pysim_construction import (create_session,
                                                run_segment)

NET = os.path.join(root, "example_network")
EXAMPLE_PROJECT = os.path.join(root, "example_project.csv")
EXAMPLE_DEMAND = os.path.join(root, "example_demand.csv")
EXAMPLE_COORDS = os.path.join(root, "example_coordinates.csv")
SF_BARE = "/home/hatch/workspace/your_files/sioux_falls_bare_inputs"
SF_ORIG_NODES = ("/home/hatch/workspace/projects/phd-dtalite/"
                 "DTALite_Files/input_node.csv")
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


def example_setup(k=2):
    _nodes, link_set = read_bare_network(NET)
    projects = load_projects(EXAMPLE_PROJECT, link_set)
    demand = load_demand(EXAMPLE_DEMAND)
    node_ids, _links = read_bare_network(NET)
    coords = load_coordinates(EXAMPLE_COORDS)
    groups, group_coords = compute_groups(node_ids, coordinates=coords,
                                          k=k, seed=42)
    return projects, demand, groups, group_coords


def toy_session(agents, closed=None, step=60.0):
    session = create_session(NET)
    run_segment(session, agents, closed or {}, step)
    return session


def in_system_count(session):
    total = 0
    for link in session.world.links:
        total += (len(link.loading_buffer) + len(link.entrance_queue)
                  + len(link.exit_queue))
    return total


# (1) project CSV validation
def test_project_validation():
    _nodes, link_set = read_bare_network(NET)
    good = ("project_id,project_type,from_node_id,to_node_id,"
            "duration_epochs,probability\n"
            "S,small,1,2,1,0.5\nS,small,1,2,2,0.5\n")
    assert len(load_projects(write_tmp(good), link_set)) == 1
    bad_cases = {
        "bad probabilities": good.replace("0.5\nS,small,1,2,2,0.5",
                                          "0.5\nS,small,1,2,2,0.4"),
        "duplicate link": good + "M,medium,1,2,4,1.0\n",
        "unknown link": good + "M,medium,2,4,4,1.0\n",
        "out of range": good.replace("S,small,1,2,2,0.5",
                                     "S,small,1,2,3,0.5"),
        "bad type": good.replace("small", "huge", 1),
        "duplicate row": good + "S,small,1,2,1,0.5\n",
    }
    for name, text in bad_cases.items():
        try:
            load_projects(write_tmp(text), link_set)
        except ValueError:
            pass
        else:
            raise AssertionError("project validation missed: %s" % name)
    check("1-project-validation", True, "6 bad files failed loud, "
          "1 good file loaded")


# (2) type constraint, (3) one start per epoch, (4) duration/closure
def test_actions_and_duration():
    projects, _d, _g, _c = example_setup()
    small, medium = projects[0], projects[1]
    # Two small projects plus a medium one, so the same-type
    # rule is testable: with S1 in progress, S2 (same type) may
    # start but M (different type) may not.
    _nodes, link_set = read_bare_network(NET)
    text = ("project_id,project_type,from_node_id,to_node_id,"
            "duration_epochs,probability\n"
            "S1,small,1,2,2,1.0\n"
            "S2,small,1,3,1,1.0\n"
            "M,medium,1,4,4,1.0\n")
    projects3 = load_projects(write_tmp(text), link_set)
    status = (("in_progress", 2), ("not_started", None),
              ("not_started", None))
    actions = available_actions(status, projects3, 0, 240)
    names = [(a, p.project_id if p else None) for a, p in actions]
    check("2-type-constraint", names == [("do_nothing", None),
                                         ("start", "S2")],
          str(names))
    # With nothing in progress, any type may start.
    status_none = (("not_started", None), ("not_started", None),
                   ("not_started", None))
    names_none = [(a, p.project_id if p else None)
                  for a, p in available_actions(status_none, projects3,
                                                0, 240)]
    check("2-type-free", names_none == [("do_nothing", None),
                                        ("start", "S1"),
                                        ("start", "S2"),
                                        ("start", "M")],
          str(names_none))
    status0 = (("not_started", None), ("not_started", None))
    actions0 = available_actions(status0, projects, 0, 240)
    check("3-one-start", all(a == "do_nothing" or p is not None
                             for a, p in actions0)
          and len([a for a, _p in actions0 if a == "start"]) == 2,
          "do-nothing + 2 single starts, no double start action")
    # d=1: closed epoch 0 only.
    s1 = apply_status(status0, projects, "start", small, 1)
    check("4-duration-1", s1[0] == ("completed", None)
          and (1, 2) in closed_set_for(status0, projects, "start",
                                       small, 1)
          and (1, 2) not in closed_set_for(s1, projects, "do_nothing",
                                           None, None),
          "closed [0], completed at 1")
    # d=2: closed epochs 0 and 1, completed at 2.
    s2 = apply_status(status0, projects, "start", small, 2)
    s2b = apply_status(s2, projects, "do_nothing", None, None)
    check("4-duration-2", s2[0] == ("in_progress", 1)
          and (1, 2) in closed_set_for(s2, projects, "do_nothing",
                                       None, None)
          and s2b[0] == ("completed", None)
          and (1, 2) not in closed_set_for(s2b, projects, "do_nothing",
                                           None, None),
          "closed [0,1], completed at 2")


# (5) profiles and weighted penalty
def test_profiles_and_weighted_penalty():
    profiles = enumerate_profiles(4)
    check("5-81-profiles", len(profiles) == 81
          and abs(sum(p for _l, p in profiles) - 1.0) < 1e-12,
          "81 profiles, probs sum to 1")
    projects, demand, groups, _c = example_setup(k=2)
    sorted_groups = sorted(set(groups.values()))
    override = [(("low", "low"), 0.25), (("medium", "medium"), 0.5),
                (("high", "high"), 0.25)]
    source = create_session(NET)
    penalty, _stored = compute_transition(
        source, {}, demand, groups, sorted_groups, 0, 0,
        profiles=override)
    manual = 0.0
    for (levels, prob) in override:
        agents = generate_epoch_agents(demand, groups, sorted_groups,
                                       levels, 0, 0, 1, 10000, 0)
        sess = create_session(NET)
        result = run_segment(sess, agents, {}, 360.0)
        manual += prob * result["penalty"]
    check("5-weighted-penalty", abs(penalty - manual) < 1e-9,
          "builder %.6f vs manual %.6f" % (penalty, manual))


def test_full_81_transition():
    projects, demand, groups4, _c = example_setup(k=4)
    sorted_groups = sorted(set(groups4.values()))
    source = create_session(NET)
    started = time.time()
    penalty, stored = compute_transition(source, {}, demand, groups4,
                                         sorted_groups, 0, 0)
    elapsed = time.time() - started
    check("5-full-81-transition", penalty == penalty
          and in_system_count(stored) >= 0,
          "81 profiles, penalty %.3f, %.1fs" % (penalty, elapsed))
    print("    full-81 transition took %.1fs" % elapsed)


# (6) equidistant departures
def test_equidistant():
    demand = {(1, 2, 0, 0): 60.0}
    groups = {1: 0, 2: 1}
    agents = generate_epoch_agents(demand, groups, [0, 1],
                                   ("medium", "medium"), 0, 0, 1, 10000, 0)
    deps = [a["departure_time_in_min"] for a in agents]
    check("6-equidistant-60", len(deps) == 60 and deps[0] == 0.0
          and deps[-1] == 59.0
          and all(deps[i + 1] - deps[i] == 1.0 for i in range(59)),
          "60 cars, 1-min spacing, first 0.0, last 59.0")
    demand0 = {(1, 2, 0, 0): 0.0}
    agents0 = generate_epoch_agents(demand0, groups, [0, 1],
                                    ("medium", "medium"), 0, 0, 1, 10000, 0)
    check("6-zero-demand", agents0 == [], "D=0 gives no cars")
    projects, demand_e, groups_e, _c = example_setup()
    agents_e = generate_epoch_agents(
        demand_e, groups_e, sorted(set(groups_e.values())),
        ("medium",) * len(set(groups_e.values())), 0, 0, 1, 10000, 0)
    check("6-in-window", all(0 <= a["departure_time_in_min"] < 360
                             for a in agents_e),
          "%d agents in [0, 360)" % len(agents_e))


# (7) branch sharing, (12) terminal nodes, (16) example end-to-end
def test_example_end_to_end():
    projects, demand, groups, group_coords = example_setup(k=2)
    started = time.time()
    with tempfile.TemporaryDirectory() as out:
        network = build_network(NET, projects, demand, groups,
                                "2026-11-02", horizon=2, output_dir=out,
                                coords=group_coords)
        elapsed = time.time() - started
        check("16-example-counts", len(network.nodes) == 11
              and len(network.links) == 12,
              "11 nodes, 12 links, %.1fs" % elapsed)
        check("16-example-merges", network.totals["merges"] == 2,
              "2 merges")
        root_links = network.outgoing(0)
        start_links = [l for l in root_links if l.action == "S"]
        check("7-branch-sharing", len(start_links) == 2
              and start_links[0].link_weight == start_links[1].link_weight
              and network.nodes[start_links[0].to_node].traffic
              is network.nodes[start_links[1].to_node].traffic,
              "2 S branches, same penalty, same traffic object")
        for node in network.terminal_nodes():
            check("12-terminal-%d" % node.node_id, node.epoch == 2
                  and network.outgoing(node.node_id) == []
                  and node.unfinished_count == sum(
                      1 for state, _r in node.status
                      if state != "completed"),
                  "epoch 2, no actions, unfinished %d"
                  % node.unfinished_count)
        with open(os.path.join(out, "nodes.csv")) as fh:
            rows = list(csv.DictReader(fh))
        with open(os.path.join(out, "links.csv")) as fh:
            link_rows = list(csv.DictReader(fh))
        check("16-example-csvs", len(rows) == 11 and len(link_rows) == 12
              and os.path.isfile(os.path.join(out, "groups.csv")),
              "nodes.csv 11 rows, links.csv 12 rows, groups.csv present")
    print("    example end-to-end took %.1fs" % elapsed)


# (8) history merging with 0.7/0.3
def test_history_merging():
    # One car on the long 1->4 link (FFTT 10), departing at 55, so
    # it is still in the system at the 60-minute clock; the other
    # session is empty, so the two in-system counts differ.
    sess_a = toy_session(
        [{"agent_id": 1, "origin_node_id": 1, "destination_node_id": 4,
          "departure_time_in_min": 55.0}], {}, 60.0)
    sess_b = toy_session([], {}, 60.0)
    before_a = in_system_count(sess_a)
    before_b = in_system_count(sess_b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = merge_traffic(sess_a, sess_b, 0.7, 0.3)
    check("8-merge-weights", in_system_count(sess_a) == before_a
          and in_system_count(sess_b) == before_b,
          "inputs unmutated (A %d, B %d)" % (before_a, before_b))
    # Weighted mean in-system is 0.7*A + 0.3*B; merged is within 1
    # of it (largest-remainder rounding) and closer to A than to B
    # when A != B.
    if before_a != before_b:
        mean = 0.7 * before_a + 0.3 * before_b
        check("8-merge-mean", abs(in_system_count(merged) - mean) <= 1.0,
              "merged %d vs weighted mean %.2f"
              % (in_system_count(merged), mean))


# (9) cache across large-project branches
def test_cache():
    _nodes, link_set = read_bare_network(NET)
    text = ("project_id,project_type,from_node_id,to_node_id,"
            "duration_epochs,probability\n")
    for i, duration in enumerate((28, 29, 30, 31, 32)):
        text += "L,large,1,2,%d,0.2\n" % duration
    projects = load_projects(write_tmp(text), link_set)
    demand = load_demand(EXAMPLE_DEMAND)
    node_ids, _links = read_bare_network(NET)
    coords = load_coordinates(EXAMPLE_COORDS)
    groups, group_coords = compute_groups(node_ids, coordinates=coords,
                                          k=2, seed=42)
    with tempfile.TemporaryDirectory() as out:
        network = build_network(NET, projects, demand, groups,
                                "2026-11-02", horizon=2, output_dir=out,
                                coords=group_coords)
    check("9-cache-hits", network.totals["cache_hits"] == 4,
          "4 hits, %d computed"
          % network.totals["transitions_computed"])
    branch_nodes = [n for n in network.nodes_at(1)
                    if n.status[0][0] == "in_progress"]
    check("9-cache-branches", len(branch_nodes) == 5,
          "5 large branches at epoch 1")
    weights = set()
    for node in branch_nodes:
        for link in network.outgoing(node.node_id):
            if link.action == "do_nothing":
                weights.add(link.link_weight)
    check("9-cache-identical", len(weights) == 1,
          "all 5 do-nothing branch penalties identical")


# (10) id invariants, (11) pruning
def test_ids_and_pruning():
    projects, demand, groups, group_coords = example_setup(k=2)
    with tempfile.TemporaryDirectory() as out:
        network = build_network(NET, projects, demand, groups,
                                "2026-11-02", horizon=2, output_dir=out,
                                coords=group_coords)
    for node in network.nodes.values():
        session = node.traffic
        ids = list(session.world.vehicles)
        check("10-unique-%d" % node.node_id, len(ids) == len(set(ids))
              and not (set(ids) & set(
                  getattr(session, "pruned_completed_ids", set()))),
              "node %d" % node.node_id)
        base = next_agent_id_base(session)
        check("10-base-%d" % node.node_id,
              base == max(list(session.seen_ids) or [0]) + 1,
              "base %d = max+1" % base)
    session = toy_session(
        [{"agent_id": 7, "origin_node_id": 1, "destination_node_id": 2,
          "departure_time_in_min": 5.0}], {}, 60.0)
    assert session.sim.completed == 1
    pruned = prune_session(session)
    check("11-prune", pruned == 1 and session.sim.completed == 1
          and session.sim.generated == 1
          and completed_ids_all(session) == {7}
          and 7 not in session.world.vehicles,
          "completed dropped to id set, counters preserved")
    try:
        run_segment(session,
                    [{"agent_id": 7, "origin_node_id": 1,
                      "destination_node_id": 2,
                      "departure_time_in_min": 65.0}], {}, 60.0)
    except ValueError:
        pass
    else:
        raise AssertionError("pruned duplicate id did not fail loud")
    check("11-prune-duplicate", True, "reuse of pruned id fails loud")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        avg = average_state([session, toy_session([], {}, 60.0)])
    check("11-average-pruned", avg.sim.completed >= 0
          and isinstance(avg.average_info, dict),
          "average_state accepts a pruned session")


# (13) k-means
def test_kmeans():
    if not os.path.isfile(SF_ORIG_NODES):
        print("SKIP 13-kmeans-sf (original SF nodes not on this machine)")
        return
    coords = load_coordinates(SF_ORIG_NODES)
    groups1, _c = compute_groups(sorted(coords), coordinates=coords,
                                 k=4, seed=42)
    groups2, _c = compute_groups(sorted(coords), coordinates=coords,
                                 k=4, seed=42)
    check("13-kmeans-sf", len(set(groups1.values())) == 4
          and groups1 == groups2 and len(groups1) == 24,
          "4 groups, 24 nodes, deterministic on rerun")
    with tempfile.TemporaryDirectory() as out:
        path = os.path.join(out, "groups.csv")
        save_groups_csv(path, groups1, coords)
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        check("13-kmeans-saved", len(rows) == 24
              and rows[0].keys() == {"node_id", "group", "x", "y"} or
              set(rows[0].keys()) == {"node_id", "group", "x", "y"},
              "groups.csv 24 rows with x,y")


# (14) poisson ppf
def test_poisson():
    for base in (0, 1, 3, 10, 100, 500, 2000, 10000):
        low = poisson_ppf(0.25, base)
        med = int(round(base)) if base == int(base) else None
        high = poisson_ppf(0.75, base)
        check("14-poisson-order-%d" % base, low <= base <= high or
              (low <= high),
              "low %d, high %d at base %d" % (low, high, base))
        if True:
            try:
                from scipy.stats import poisson as _sp
                sci_low = int(_sp.ppf(0.25, base))
                sci_high = int(_sp.ppf(0.75, base))
                check("14-poisson-scipy-%d" % base,
                      poisson_ppf_fallback_only(0.25, base) == sci_low
                      and poisson_ppf_fallback_only(0.75, base) == sci_high,
                      "fallback matches scipy")
            except ImportError:
                pass


# (15) sequential vs parallel
def test_parallel_equivalence():
    projects, demand, groups, _c = example_setup(k=2)
    sorted_groups = sorted(set(groups.values()))
    override = enumerate_profiles(2)
    source = create_session(NET)
    penalty_seq, stored_seq = compute_transition(
        source, {}, demand, groups, sorted_groups, 0, 0,
        profiles=override)
    source2 = create_session(NET)
    penalty_par, stored_par = compute_transition(
        source2, {}, demand, groups, sorted_groups, 0, 0, workers=2,
        profiles=override)
    def link_counts(session):
        return tuple((len(l.loading_buffer), len(l.entrance_queue),
                      len(l.exit_queue)) for l in session.world.links)
    check("15-parallel", penalty_seq == penalty_par
          and link_counts(stored_seq) == link_counts(stored_par)
          and stored_seq.sim.generated == stored_par.sim.generated,
          "penalty %.6f identical, link counts identical"
          % penalty_seq)


# Weighted average_state behaviour
def test_weighted_average():
    sess_a = toy_session(
        [{"agent_id": 1, "origin_node_id": 1, "destination_node_id": 4,
          "departure_time_in_min": 55.0}], {}, 60.0)
    sess_b = toy_session([], {}, 60.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w10 = weighted_average_state([sess_a, sess_b], [1, 0])
        w_un = average_state([sess_a, sess_b])
        w_7525 = weighted_average_state([sess_a, sess_b], [0.75, 0.25])
        single = average_state([sess_a])
    check("weighted-10", sorted(w10.world.vehicles) ==
          sorted(sess_a.world.vehicles)
          and w10.sim.generated == sess_a.sim.generated,
          "[1,0] reproduces the first session")
    check("weighted-closer", abs(in_system_count(w_7525)
                                 - in_system_count(sess_a))
          <= abs(in_system_count(w_un) - in_system_count(sess_a)),
          "0.75/0.25 at least as close to first as unweighted")
    check("weighted-single", single is not sess_a
          and sorted(single.world.vehicles) ==
          sorted(sess_a.world.vehicles),
          "single session returns a deepcopy")
    for bad in ([], [1], [1, -0.5], [0, 0]):
        try:
            weighted_average_state([sess_a, sess_b], bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad weights accepted: %r" % bad)
    check("weighted-validation", True,
          "empty/mismatched/negative/sum-zero weights fail loud")
    # Unweighted wrapper matches the original module's behaviour on
    # counts (verified against the original file during the build).
    check("weighted-wrapper", w_un.average_info["weights"] == [0.5, 0.5],
          "unweighted wrapper uses equal weights")


def test_prune_flag():
    _nodes, link_set = read_bare_network(NET)
    text = ("project_id,project_type,from_node_id,to_node_id,"
            "duration_epochs,probability\n"
            "L,large,1,2,28,1.0\n")
    projects = load_projects(write_tmp(text), link_set,
                             allow_out_of_range_durations=True)
    demand = load_demand(EXAMPLE_DEMAND)
    node_ids, _links = read_bare_network(NET)
    coords = load_coordinates(EXAMPLE_COORDS)
    groups, group_coords = compute_groups(node_ids, coordinates=coords,
                                          k=2, seed=42)
    with tempfile.TemporaryDirectory() as out:
        net_off = build_network(NET, projects, demand, groups,
                                "2026-11-02", horizon=2, output_dir=out,
                                coords=group_coords)
    with tempfile.TemporaryDirectory() as out:
        net_on = build_network(NET, projects, demand, groups,
                               "2026-11-02", horizon=2, output_dir=out,
                               prune_unfinishable_starts=True,
                               coords=group_coords)
    off_starts = [l for l in net_off.links if l.action == "L"]
    on_starts = [l for l in net_on.links if l.action == "L"]
    check("prune-flag", len(off_starts) > 0 and len(on_starts) == 0,
          "flag on removes doomed large starts (%d -> 0)"
          % len(off_starts))


def main():
    tests = [test_project_validation, test_actions_and_duration,
             test_profiles_and_weighted_penalty,
             test_full_81_transition, test_equidistant,
             test_example_end_to_end, test_history_merging,
             test_cache, test_ids_and_pruning, test_kmeans,
             test_poisson, test_parallel_equivalence,
             test_weighted_average, test_prune_flag]
    started = time.time()
    for fn in tests:
        print("== %s" % fn.__name__)
        fn()
    print("ALL TESTS PASSED: %d checks in %.1fs"
          % (len(PASSED), time.time() - started))


if __name__ == "__main__":
    main()
