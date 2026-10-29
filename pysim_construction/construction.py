# Construction closures for pysim_construction.
#
# One construction_projects.csv row closes one directed link for one
# project. A project is a set of rows sharing a project_id; the brief
# requires every row of a project to share one start time and one
# duration, so a project starts together and finishes together.
# The manager below owns only that bookkeeping: parsing and
# validation, the barred-link union, and the per-project state
# machine. The simulator (loading.py) owns all vehicle behaviour:
# it asks the manager which links are barred, when a project may
# start, and when it ends, and it performs the reroutes itself.
import csv
import os

from .cpp import ProgramStop

EXPECTED_HEADER = ["project_id", "project_name", "from_node_id",
                   "to_node_id", "start_time_in_min", "duration_in_min"]


def _fmt_time(value):
    if value is None:
        return ""
    if value == int(value):
        return str(int(value))
    return "%.1f" % value


def _parse_number(text, what, where):
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ProgramStop(
            "construction_projects.csv: non-numeric %s %r (%s)"
            % (what, text, where))
    if value != value or value in (float("inf"), float("-inf")):
        raise ProgramStop(
            "construction_projects.csv: non-numeric %s %r (%s)"
            % (what, text, where))
    return value


class ConstructionProject:
    def __init__(self, project_id, project_name, scheduled, duration):
        self.project_id = project_id
        self.project_name = project_name
        self.scheduled = scheduled
        self.duration = duration
        # Rows in file order: (from_number, to_number, link_no).
        self.rows = []
        self.state = "pending"
        self.barred_at = None
        self.link_cleared = {}
        self.construction_start = None
        self.construction_end = None

    def link_nos(self):
        return [row[2] for row in self.rows]

    def status(self):
        if self.state == "completed":
            return "completed"
        if self.state in ("pending", "never_started"):
            return "never_started"
        return "still_closed_at_horizon"


class ConstructionManager:
    def __init__(self, path, world):
        if not os.path.isfile(path):
            raise ProgramStop(
                "construction_projects.csv not found: %s "
                "(this construction version always requires a "
                "construction file; it never runs without closures)"
                % path)
        self.path = path
        self.projects = []
        self._by_id = {}
        seen_rows = set()
        with open(path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                raise ProgramStop(
                    "construction_projects.csv is empty")
            header = [h.strip() for h in header]
            if header != EXPECTED_HEADER:
                raise ProgramStop(
                    "construction_projects.csv header must be exactly "
                    "%s, got %s" % (",".join(EXPECTED_HEADER),
                                    ",".join(header)))
            for line_no, row in enumerate(reader, start=2):
                if not row or all(c.strip() == "" for c in row):
                    continue
                if len(row) != len(EXPECTED_HEADER):
                    raise ProgramStop(
                        "construction_projects.csv line %d: expected %d "
                        "columns, got %d" % (line_no, len(EXPECTED_HEADER),
                                             len(row)))
                pid = row[0].strip()
                name = row[1].strip()
                if pid == "":
                    raise ProgramStop(
                        "construction_projects.csv line %d: empty "
                        "project_id" % line_no)
                try:
                    from_node = int(row[2])
                    to_node = int(row[3])
                except ValueError:
                    raise ProgramStop(
                        "construction_projects.csv line %d: non-integer "
                        "node id" % line_no)
                where = "line %d" % line_no
                start = _parse_number(row[4].strip(),
                                      "start_time_in_min", where)
                duration = _parse_number(row[5].strip(),
                                         "duration_in_min", where)
                if duration <= 0:
                    raise ProgramStop(
                        "construction_projects.csv line %d: "
                        "duration_in_min must be > 0, got %s"
                        % (line_no, row[5].strip()))
                key = (pid, from_node, to_node)
                if key in seen_rows:
                    raise ProgramStop(
                        "construction_projects.csv: duplicate row for "
                        "project %s link %d->%d" % (pid, from_node, to_node))
                seen_rows.add(key)
                link_no = world.link_map.get((from_node, to_node))
                if link_no is None:
                    raise ProgramStop(
                        "construction_projects.csv line %d: unknown "
                        "link %d->%d (no such link in the network)"
                        % (line_no, from_node, to_node))
                proj = self._by_id.get(pid)
                if proj is None:
                    proj = ConstructionProject(pid, name, start, duration)
                    self._by_id[pid] = proj
                    self.projects.append(proj)
                else:
                    if proj.scheduled != start or \
                            proj.duration != duration:
                        raise ProgramStop(
                            "construction_projects.csv: project %s "
                            "has differing start/duration across rows "
                            "(project sync requires one common start "
                            "and duration)" % pid)
                    if proj.project_name == "" and name != "":
                        proj.project_name = name
                proj.rows.append((from_node, to_node, link_no))
        self._barred = set()

    def barred_link_nos(self):
        return self._barred

    def has_active_projects(self):
        return any(p.state in ("barred", "in_construction")
                   for p in self.projects)

    def _rebuild_barred(self):
        barred = set()
        for proj in self.projects:
            if proj.state in ("barred", "in_construction"):
                barred.update(proj.link_nos())
        self._barred = barred

    @staticmethod
    def _link_is_clear(world, link_no):
        link = world.links[link_no]
        if link.cflow_arrival != link.cflow_departure:
            return False
        if link.loading_buffer or link.entrance_queue or link.exit_queue:
            return False
        return True

    def update(self, world, current_time):
        # Called at the start of every simulation step, before any
        # stage runs. Times are compared with a 1e-6 tolerance
        # because step times are start + step * 0.1 in float.
        eps = 1e-6
        for proj in self.projects:
            if proj.state == "pending" and \
                    current_time + eps >= proj.scheduled:
                proj.state = "barred"
                proj.barred_at = proj.scheduled
        self._rebuild_barred()
        for proj in self.projects:
            if proj.state == "barred":
                for _from, _to, link_no in proj.rows:
                    if link_no not in proj.link_cleared and \
                            self._link_is_clear(world, link_no):
                        proj.link_cleared[link_no] = current_time
                if len(proj.link_cleared) == len(proj.rows):
                    cleared = max(proj.link_cleared.values())
                    proj.construction_start = cleared
                    proj.construction_end = cleared + proj.duration
                    proj.state = "in_construction"
        for proj in self.projects:
            if proj.state == "in_construction" and \
                    proj.construction_end is not None and \
                    current_time + eps >= proj.construction_end:
                proj.state = "completed"
        self._rebuild_barred()

    def finalize(self):
        # Called once after the loading loop (including after an
        # early exit). Pending projects never barred: never_started.
        # Barred or in-construction projects at that point are still
        # closed at the horizon, whether or not they had started
        # construction; their recorded start/end keep whatever was
        # already determined (empty if construction never started).
        for proj in self.projects:
            if proj.state == "pending":
                proj.state = "never_started"
            elif proj.state in ("barred", "in_construction"):
                proj.state = "still_closed"
        self._barred = set()

    def rows_for_output(self):
        rows = []
        for proj in self.projects:
            for from_node, to_node, link_no in proj.rows:
                rows.append({
                    "project_id": proj.project_id,
                    "project_name": proj.project_name,
                    "from_node_id": from_node,
                    "to_node_id": to_node,
                    "scheduled_start": proj.scheduled,
                    "barred_at": proj.barred_at,
                    "link_cleared_at": proj.link_cleared.get(link_no),
                    "construction_start": proj.construction_start,
                    "construction_end": proj.construction_end,
                    "duration": proj.duration,
                    "status": proj.status(),
                })
        return rows
