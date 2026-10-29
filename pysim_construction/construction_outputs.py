# The three construction-only output files. Bare v2 writes no
# other files: there is no agent, trip or link output writer.
import os

from .construction import _fmt_time
from .cpp import f32

EXCESS_HEADER = (
    "agent_id,origin,destination,departure,arrival,"
    "travel_time_in_min,trip_fftt_actual_in_min,"
    "trip_fftt_closed_in_min,excess_actual_in_min,"
    "excess_closed_in_min,completed\n")

SUMMARY_HEADER = (
    "completed,incomplete,total_excess_closed_in_min,"
    "mean_excess_closed_in_min,total_excess_closed_in_hours,"
    "mean_excess_closed_in_hours,total_excess_actual_in_min,"
    "mean_excess_actual_in_min,total_excess_actual_in_hours,"
    "mean_excess_actual_in_hours\n")

CONSTRUCTION_HEADER = (
    "project_id,project_name,from_node_id,to_node_id,"
    "scheduled_start_in_min,barred_at_in_min,link_cleared_at_in_min,"
    "construction_start_in_min,construction_end_in_min,"
    "duration_in_min,status\n")


def _num(value):
    if value is None:
        return ""
    return "%.4f" % float(value)


def compute_excess(world):
    # Per-vehicle excess, completed vehicles only for the totals.
    # travel_time is PySim's veh.travel_time (arrival minus
    # leaving_buffer_time), never trip_time. trip_fftt_actual is
    # the post-loading trip_fftt over the links actually traversed.
    # trip_fftt_closed is the free-flow total over the original
    # path captured before any closure reroute (preassigned input
    # path, or the iteration-0 free-flow path for pathless agents).
    # Neither excess is clamped: a reroute to a shorter path can
    # legitimately be negative. Incomplete vehicles carry empty
    # excess fields and are excluded from the totals.
    nodes = world.nodes
    rows = []
    for agent_id in sorted(world.vehicles):
        veh = world.vehicles[agent_id]
        origin = nodes[veh.origin_node].number
        dest = nodes[veh.dest_node].number
        if veh.complete:
            travel = float(veh.travel_time)
            fftt_actual = float(veh.trip_fftt)
            fftt_closed = float(veh.closed_fftt)
            rows.append({
                "agent_id": agent_id, "origin": origin,
                "destination": dest, "departure": float(veh.departure),
                "arrival": float(veh.arrival),
                "travel_time": travel, "fftt_actual": fftt_actual,
                "fftt_closed": fftt_closed,
                "excess_actual": travel - fftt_actual,
                "excess_closed": travel - fftt_closed,
                "completed": 1,
            })
        else:
            rows.append({
                "agent_id": agent_id, "origin": origin,
                "destination": dest, "departure": float(veh.departure),
                "arrival": None, "travel_time": None,
                "fftt_actual": None,
                "fftt_closed": float(veh.closed_fftt)
                if veh.node_size >= 2 else None,
                "excess_actual": None, "excess_closed": None,
                "completed": 0,
            })
    completed_rows = [r for r in rows if r["completed"] == 1]
    total_closed = sum(r["excess_closed"] for r in completed_rows)
    total_actual = sum(r["excess_actual"] for r in completed_rows)
    n = len(completed_rows)
    summary = {
        "completed": n,
        "incomplete": len(rows) - n,
        "total_excess_closed_min": total_closed,
        "mean_excess_closed_min": total_closed / n if n else 0.0,
        "total_excess_closed_hours": total_closed / 60.0,
        "mean_excess_closed_hours":
            (total_closed / n / 60.0) if n else 0.0,
        "total_excess_actual_min": total_actual,
        "mean_excess_actual_min": total_actual / n if n else 0.0,
        "total_excess_actual_hours": total_actual / 60.0,
        "mean_excess_actual_hours":
            (total_actual / n / 60.0) if n else 0.0,
    }
    return rows, summary


def write_excess_files(world, excess_path, summary_path):
    rows, summary = compute_excess(world)
    lines = [EXCESS_HEADER]
    for r in rows:
        lines.append(
            "%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%d\n" % (
                r["agent_id"], r["origin"], r["destination"],
                _num(r["departure"]), _num(r["arrival"]),
                _num(r["travel_time"]), _num(r["fftt_actual"]),
                _num(r["fftt_closed"]), _num(r["excess_actual"]),
                _num(r["excess_closed"]), r["completed"]))
    with open(excess_path, "w", newline="", encoding="utf-8") as fh:
        fh.writelines(lines)
    line = (
        "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" % (
            summary["completed"], summary["incomplete"],
            summary["total_excess_closed_min"],
            summary["mean_excess_closed_min"],
            summary["total_excess_closed_hours"],
            summary["mean_excess_closed_hours"],
            summary["total_excess_actual_min"],
            summary["mean_excess_actual_min"],
            summary["total_excess_actual_hours"],
            summary["mean_excess_actual_hours"]))
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(SUMMARY_HEADER)
        fh.write(line)
    return summary


def write_construction_file(manager, path):
    lines = [CONSTRUCTION_HEADER]
    for row in manager.rows_for_output():
        lines.append(
            "%s,%s,%d,%d,%s,%s,%s,%s,%s,%s,%s\n" % (
                row["project_id"], row["project_name"],
                row["from_node_id"], row["to_node_id"],
                _fmt_time(row["scheduled_start"]),
                _fmt_time(row["barred_at"]),
                _fmt_time(row["link_cleared_at"]),
                _fmt_time(row["construction_start"]),
                _fmt_time(row["construction_end"]),
                _fmt_time(row["duration"]), row["status"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.writelines(lines)
