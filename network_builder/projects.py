"""Project CSV loading and validation.

File format. CSV with header exactly
project_id,project_type,from_node_id,to_node_id,duration_epochs,
probability, one row per duration outcome, project_id repeated
across its outcomes. Types are small, medium and large only; no
project count is hardcoded (Eli's real file is expected to hold
about 8 projects, but the builder validates types, not counts).

Duration ranges, enforced unless allow_out_of_range_durations
is True: small 1-2 epochs, medium 4-12, large 28-56. A duration
outside its type's range fails loud by default, because a
mistyped duration (for example a medium project entered in
hours) would silently build a nonsense network.

Other fail-loud rules: probabilities per project must sum to
1.0 within 1e-6, each individual probability must be in (0, 1];
each project has exactly one directed link (rows for one
project_id with different links or types fail); a link may
belong to at most one project (duplicate link across projects
fails); a (project_id, from, to, duration) row may not repeat;
every project link must exist in input_link.csv; a project
link may not be a self-loop.
"""

import csv

RANGES = {"small": (1, 2), "medium": (4, 12), "large": (28, 56)}
HEADER = ["project_id", "project_type", "from_node_id", "to_node_id",
          "duration_epochs", "probability"]


class Project:
    def __init__(self, project_id, project_type, link, outcomes):
        self.project_id = project_id
        self.project_type = project_type
        self.link = link
        self.outcomes = sorted(outcomes)
        self.min_duration = min(d for d, _p in self.outcomes)

    def __repr__(self):
        return "Project(%r, %r, %r, %r)" % (
            self.project_id, self.project_type, self.link, self.outcomes)


def _as_int(value, what):
    if isinstance(value, bool):
        raise ValueError("%s must be an integer, got %r" % (what, value))
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except ValueError:
        raise ValueError("%s must be an integer, got %r" % (what, value)) from None


def load_projects(path, link_set, allow_out_of_range_durations=False):
    """Load and validate projects; returns a list of Project.

    link_set is the set of (from_node_id, to_node_id) pairs from
    input_link.csv. Projects are returned in first-appearance
    order in the file.
    """
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("projects file is empty: %s" % path) from None
        header = [h.strip() for h in header]
        if header != HEADER:
            raise ValueError(
                "projects file header must be exactly %s, got %s"
                % (",".join(HEADER), ",".join(header)))
        by_id = {}
        order = []
        link_owner = {}
        seen_rows = set()
        for line_no, row in enumerate(reader, start=2):
            if not row or all(cell.strip() == "" for cell in row):
                continue
            if len(row) != len(HEADER):
                raise ValueError("projects file line %d: expected %d "
                                 "columns, got %d"
                                 % (line_no, len(HEADER), len(row)))
            cells = [cell.strip() for cell in row]
            project_id, project_type = cells[0], cells[1]
            if not project_id:
                raise ValueError("projects file line %d: empty "
                                 "project_id" % line_no)
            if project_type not in RANGES:
                raise ValueError(
                    "projects file line %d: project_type must be "
                    "small/medium/large, got %r" % (line_no, project_type))
            from_node = _as_int(cells[2], "from_node_id")
            to_node = _as_int(cells[3], "to_node_id")
            duration = _as_int(cells[4], "duration_epochs")
            try:
                prob = float(cells[5])
            except ValueError:
                raise ValueError("projects file line %d: bad probability "
                                 "%r" % (line_no, cells[5])) from None
            if from_node == to_node:
                raise ValueError("projects file line %d: project %s "
                                 "is a self-loop %d->%d"
                                 % (line_no, project_id, from_node, to_node))
            if (from_node, to_node) not in link_set:
                raise ValueError(
                    "projects file line %d: project %s link %d->%d "
                    "is not in input_link.csv"
                    % (line_no, project_id, from_node, to_node))
            if duration < 1:
                raise ValueError("projects file line %d: duration_epochs "
                                 "must be >= 1, got %d" % (line_no, duration))
            if not allow_out_of_range_durations:
                low, high = RANGES[project_type]
                if not low <= duration <= high:
                    raise ValueError(
                        "projects file line %d: %s duration %d outside "
                        "range %d-%d epochs (pass "
                        "allow_out_of_range_durations=True to override)"
                        % (line_no, project_type, duration, low, high))
            if not 0 < prob <= 1:
                raise ValueError("projects file line %d: probability "
                                 "must be in (0, 1], got %r" % (line_no, prob))
            row_key = (project_id, from_node, to_node, duration)
            if row_key in seen_rows:
                raise ValueError("projects file line %d: duplicate row "
                                 "for %s %d->%d duration %d"
                                 % (line_no, project_id, from_node,
                                    to_node, duration))
            seen_rows.add(row_key)
            owner = link_owner.get((from_node, to_node))
            if owner is not None and owner != project_id:
                raise ValueError(
                    "projects file line %d: link %d->%d belongs to "
                    "both %s and %s" % (line_no, from_node, to_node,
                                        owner, project_id))
            link_owner[(from_node, to_node)] = project_id
            if project_id not in by_id:
                by_id[project_id] = {
                    "type": project_type, "link": (from_node, to_node),
                    "outcomes": []}
                order.append(project_id)
            entry = by_id[project_id]
            if entry["type"] != project_type or entry["link"] != (from_node, to_node):
                raise ValueError(
                    "projects file line %d: project %s has more than "
                    "one type/link" % (line_no, project_id))
            entry["outcomes"].append((duration, prob))
    projects = []
    for project_id in order:
        entry = by_id[project_id]
        total = sum(prob for _d, prob in entry["outcomes"])
        if abs(total - 1.0) > 1e-6:
            raise ValueError("project %s probabilities sum to %r, "
                             "not 1.0" % (project_id, total))
        projects.append(Project(project_id, entry["type"], entry["link"],
                                entry["outcomes"]))
    if not projects:
        raise ValueError("projects file %s has no projects" % path)
    return projects
