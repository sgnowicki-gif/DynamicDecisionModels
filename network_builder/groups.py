"""Node groups: deterministic k-means on coordinates, or a file.

Production mode is k = 4 groups from the original Sioux Falls x/y
coordinates, unscaled, exactly as specified. k-means is implemented
here with numpy (Lloyd's algorithm, k-means++ initialisation with
a fixed seed) so sklearn is never required; if sklearn is present
its result is not used, because the numpy path is canonical and
its output is documented here, not delegated to a library whose
version could change the assignments.

Determinism: same point order, same seed, same k gives the same
assignments on rerun (RandomState, not the global RNG, and ties
broken by lowest index). Points are given in node-id order, so
the input order is itself deterministic.

Groups file fallback. For a synthetic or test network without
x/y coordinates, the builder accepts a precomputed groups file
(node_id,group, with optional x,y columns). That path is for
tests and examples only; production runs use coordinates.

Saving. Groups are saved as node_id,group,x,y when coordinates
are known, else node_id,group, sorted by node_id.
"""

import csv

import numpy as np


def _as_int(value, what):
    if isinstance(value, bool):
        raise ValueError("%s must be an integer, got %r" % (what, value))
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except ValueError:
        raise ValueError("%s must be an integer, got %r" % (what, value)) from None


def load_coordinates(path):
    """Load {node_id: (x, y)} from a CSV with node_id, x, y columns.

    Other columns are ignored, so the original DTALite
    input_node.csv can be passed directly. Duplicate node ids
    and non-finite coordinates fail loud.
    """
    coords = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("coordinates file is empty: %s" % path)
        fields = [h.strip() for h in reader.fieldnames]
        for col in ("node_id", "x", "y"):
            if col not in fields:
                raise ValueError("coordinates file %s is missing column "
                                 "%s" % (path, col))
        for line_no, raw in enumerate(reader, start=2):
            row = {}
            for key, value in raw.items():
                if key is None:
                    continue
                row[key.strip()] = value.strip() if isinstance(
                    value, str) else value
            if all(v in (None, "") for v in row.values()):
                continue
            node = _as_int(row["node_id"], "node_id")
            try:
                x = float(row["x"])
                y = float(row["y"])
            except (TypeError, ValueError):
                raise ValueError("coordinates file line %d: bad x/y %r/%r"
                                 % (line_no, row["x"], row["y"])) from None
            if x != x or y != y or x in (float("inf"), float("-inf")) \
                    or y in (float("inf"), float("-inf")):
                raise ValueError("coordinates file line %d: non-finite "
                                 "x/y" % line_no)
            if node in coords:
                raise ValueError("coordinates file line %d: duplicate "
                                 "node %d" % (line_no, node))
            coords[node] = (x, y)
    return coords


def load_groups_file(path):
    """Load {node_id: group} (and coords if present) from a file.

    Header must contain node_id and group; x and y columns, if
    present, are carried through to the saved groups.csv.
    Duplicate node ids fail loud.
    """
    groups = {}
    coords = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("groups file is empty: %s" % path)
        fields = [h.strip() for h in reader.fieldnames]
        for col in ("node_id", "group"):
            if col not in fields:
                raise ValueError("groups file %s is missing column %s"
                                 % (path, col))
        has_xy = "x" in fields and "y" in fields
        for line_no, raw in enumerate(reader, start=2):
            row = {}
            for key, value in raw.items():
                if key is None:
                    continue
                row[key.strip()] = value.strip() if isinstance(
                    value, str) else value
            if all(v in (None, "") for v in row.values()):
                continue
            node = _as_int(row["node_id"], "node_id")
            group = _as_int(row["group"], "group")
            if node in groups:
                raise ValueError("groups file line %d: duplicate node %d"
                                 % (line_no, node))
            groups[node] = group
            if has_xy:
                try:
                    coords[node] = (float(row["x"]), float(row["y"]))
                except (TypeError, ValueError):
                    raise ValueError("groups file line %d: bad x/y"
                                     % line_no) from None
    return groups, (coords if has_xy else None)


def kmeans_labels(points, k, seed=42, max_iter=100):
    """Lloyd k-means on an (n, 2) array; returns (labels, centroids).

    k-means++ initialisation with numpy RandomState(seed): the
    first centroid is a uniform draw, each later centroid is drawn
    with probability proportional to squared distance to the
    nearest existing centroid. Assignment ties go to the lowest
    centroid index. An empty cluster is re-seeded to the point
    farthest from its assigned centroid. Iteration stops when no
    label changes or max_iter is reached. k must not exceed the
    number of points; k == n gives one point per cluster.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("kmeans needs an (n, 2) point array")
    n = points.shape[0]
    if not isinstance(k, int) or k < 1 or k > n:
        raise ValueError("k must be between 1 and %d, got %r" % (n, k))
    rng = np.random.RandomState(seed)
    centroids = [points[rng.randint(n)]]
    closest_sq = ((points - centroids[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = closest_sq.sum()
        if total <= 0:
            choice = rng.randint(n)
        else:
            choice = rng.choice(n, p=closest_sq / total)
        centroids.append(points[choice])
        dist_sq = ((points - centroids[-1]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, dist_sq)
    centroids = np.array(centroids, dtype=float)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dist = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1)
        changed = not np.array_equal(new_labels, labels)
        labels = new_labels
        new_centroids = centroids.copy()
        for j in range(k):
            members = points[labels == j]
            if len(members) == 0:
                farthest = int(np.argmax(dist[np.arange(n), labels]))
                new_centroids[j] = points[farthest]
            else:
                new_centroids[j] = members.mean(axis=0)
        centroids = new_centroids
        if not changed:
            break
    return labels, centroids


def compute_groups(node_ids, coordinates=None, groups_file=None, k=4,
                   seed=42):
    """Return (groups dict, coords dict or None) for node_ids.

    groups_file takes precedence when given. Otherwise coordinates
    must cover every node id and k-means runs with the given k
    and seed. Groups are relabelled 0..k-1 in first-appearance
    order over node-id-sorted points, so labels are stable and
    readable (k-means' internal centroid order is arbitrary).
    """
    node_ids = sorted(node_ids)
    if groups_file is not None:
        groups, coords = load_groups_file(groups_file)
        missing = [node for node in node_ids if node not in groups]
        if missing:
            raise ValueError("groups file %s is missing nodes %s"
                             % (groups_file, missing[:10]))
        return {node: groups[node] for node in node_ids}, coords
    if coordinates is None:
        raise ValueError("compute_groups needs coordinates or a "
                         "groups file")
    missing = [node for node in node_ids if node not in coordinates]
    if missing:
        raise ValueError("coordinates are missing nodes %s" % (missing[:10],))
    points = np.array([coordinates[node] for node in node_ids], dtype=float)
    raw_labels, _centroids = kmeans_labels(points, k, seed=seed)
    remap = {}
    groups = {}
    for node, label in zip(node_ids, raw_labels):
        label = int(label)
        if label not in remap:
            remap[label] = len(remap)
        groups[node] = remap[label]
    return groups, {node: coordinates[node] for node in node_ids}


def save_groups_csv(path, groups, coords=None):
    """Write groups.csv sorted by node_id."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if coords is not None:
            writer.writerow(["node_id", "group", "x", "y"])
            for node in sorted(groups):
                x, y = coords[node]
                writer.writerow([node, groups[node], x, y])
        else:
            writer.writerow(["node_id", "group"])
            for node in sorted(groups):
                writer.writerow([node, groups[node]])
