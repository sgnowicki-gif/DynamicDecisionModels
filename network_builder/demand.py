"""Demand profiles, levels and equidistant agent generation.

Demand file (a parameter, never a hardcoded path). CSV with header
origin_node_id,destination_node_id,day_of_week,hour,
average_demand_vph. Extra columns are ignored; missing columns
fail loud. average_demand_vph is a float (vehicles per hour) and
may be non-integer: it is an average over days, not a count.

day_of_week, canonical form. The builder accepts exactly two
forms, and nothing else: an integer 0 = Monday .. 6 = Sunday
(CSV strings "0".."6" included), or a three-letter abbreviation
Mon, Tue, Wed, Thu, Fri, Sat, Sun (any capitalisation). Full
names ("Monday"), other abbreviations and out-of-range numbers
fail loud at load time. The builder maps a start_date
(YYYY-MM-DD) to the same 0..6 form with datetime.weekday(),
which is Monday = 0, so the two agree by construction. An OD
pair with no row for a needed (day, hour) is treated as zero
demand for that hour; the demand dict is sparse by design, so
an example file may cover two hours without listing the rest
as zeros.

Levels. Per origin group only, independent across groups, held
for a whole epoch. Level probabilities: low 0.25, medium 0.50,
high 0.25. With 4 groups there are 3^4 = 81 profiles; a profile
is one level per group, its probability is the product over
groups, and the 81 probabilities sum to 1. In example mode with
fewer groups the count is 3^(number of groups) -- see README.

Hourly counts. For each hour h of the epoch, base is the average
demand for that OD pair, that hour, that day of week. Counts:
medium = round_half_up(base); high = round_half_up(poisson_ppf
(0.75, base)); low = round_half_up(poisson_ppf(0.25, base)).
round_half_up means floor(x + 0.5) for x >= 0, documented here
because Python's round() is banker's rounding and would give a
different count at .5. poisson_ppf uses scipy if it is installed
and otherwise a mode-anchored cumulative-pmf search written here
(deterministic; checked against scipy to base 50000 -- see the
tests). The ppf result is a
count, so it is rounded half up as well; for integer-valued ppf
outputs the rounding is a no-op.

Departures. D cars in an hour depart equidistantly: hour_start +
i * 60 / D for i = 0 .. D - 1. The first car is exactly at the
hour start, the last is strictly before the next hour, and D = 0
means no cars. Departures are NOT rounded here; v3 buckets a
departure to int(departure * 10) at 0.1-minute resolution, so two
departures less than 0.1 minutes apart share a bucket in the
simulator. That is v3 behaviour, disclosed here, not a builder
rounding choice. All departures are absolute minutes from
simulation start (epoch k starts at k * 360), and generation
splits exactly by epoch: every generated departure lies in
[epoch_start, epoch_start + 360).
"""

import csv
import itertools
import math
from datetime import date

LEVEL_PROBS = {"low": 0.25, "medium": 0.50, "high": 0.25}
_LEVELS = ("low", "medium", "high")

try:
    from scipy.stats import poisson as _scipy_poisson
    _HAS_SCIPY = True
except Exception:
    _scipy_poisson = None
    _HAS_SCIPY = False

_ppf_cache = {}


def has_scipy():
    return _HAS_SCIPY


def round_half_up(value):
    """Round a non-negative float to the nearest int, halves up."""
    if value < 0 or value != value or value in (float("inf"), float("-inf")):
        raise ValueError("round_half_up needs a finite non-negative "
                         "number, got %r" % (value,))
    return int(math.floor(value + 0.5))


def _poisson_ppf_fallback(q, lam):
    if lam == 0:
        return 0
    # Mode-anchored recurrence. The naive recurrence starting at
    # exp(-lam) underflows to 0 for lam above ~745, so it can
    # never reach q. Instead anchor at the mode, whose pmf is
    # about 1/sqrt(2*pi*lam) and never underflows, then walk
    # down to 0 (storing pmfs) and up as far as needed. Correct
    # for lam up to at least tens of thousands; the tests check
    # it against scipy at 0, 1, 3, 10, 100, 500, 2000 and 10000.
    mode = int(math.floor(lam))
    log_pmf_mode = -lam + mode * math.log(lam) - math.lgamma(mode + 1)
    pmf_mode = math.exp(log_pmf_mode)
    pmfs = [0.0] * (mode + 1)
    pmfs[mode] = pmf_mode
    pmf = pmf_mode
    for k in range(mode, 0, -1):
        pmf = pmf * k / lam
        pmfs[k - 1] = pmf
    cdf = 0.0
    for k in range(mode + 1):
        cdf += pmfs[k]
        if cdf >= q:
            return k
    pmf = pmf_mode
    k = mode
    while cdf < q:
        k += 1
        pmf = pmf * lam / k
        cdf += pmf
        if k > mode + 100 * math.sqrt(lam) + 10000:
            raise ValueError("poisson ppf fallback did not converge "
                             "for q=%s, lam=%s" % (q, lam))
    return k


def poisson_ppf(q, lam):
    """Smallest k with P(Poisson(lam) <= k) >= q, for 0 < q < 1."""
    if not 0 < q < 1:
        raise ValueError("poisson_ppf quantile must be in (0, 1), "
                         "got %r" % (q,))
    if lam < 0 or lam != lam or lam in (float("inf"), float("-inf")):
        raise ValueError("poisson_ppf needs a finite non-negative "
                         "mean, got %r" % (lam,))
    key = (q, float(lam))
    if key in _ppf_cache:
        return _ppf_cache[key]
    if _HAS_SCIPY:
        result = int(_scipy_poisson.ppf(q, lam))
    else:
        result = _poisson_ppf_fallback(q, lam)
    _ppf_cache[key] = result
    return result


def poisson_ppf_fallback_only(q, lam):
    """The no-scipy path, exposed so tests can compare the two."""
    if not 0 < q < 1:
        raise ValueError("quantile must be in (0, 1)")
    if lam < 0:
        raise ValueError("mean must be non-negative")
    return _poisson_ppf_fallback(q, lam)


_ABBR = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4,
         "sat": 5, "sun": 6}


def parse_day_of_week(value):
    """Parse one day_of_week cell to 0 = Monday .. 6 = Sunday."""
    if isinstance(value, bool):
        raise ValueError("bad day_of_week %r" % (value,))
    if isinstance(value, int):
        if 0 <= value <= 6:
            return value
        raise ValueError("day_of_week int out of range 0..6: %r" % (value,))
    if isinstance(value, float) and value.is_integer():
        return parse_day_of_week(int(value))
    text = str(value).strip()
    if text in ("0", "1", "2", "3", "4", "5", "6"):
        return int(text)
    key = text.lower()
    if key in _ABBR:
        return _ABBR[key]
    raise ValueError(
        "day_of_week must be 0..6 (Monday..Sunday) or Mon..Sun, "
        "got %r" % (value,))


def parse_start_date(text):
    """Parse YYYY-MM-DD and return (date object, weekday 0..6)."""
    if not isinstance(text, str):
        raise ValueError("start_date must be a YYYY-MM-DD string, "
                         "got %r" % (text,))
    parts = text.strip().split("-")
    if len(parts) != 3:
        raise ValueError("start_date must be YYYY-MM-DD, got %r" % (text,))
    try:
        day = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise ValueError("start_date %r is not a real date: %s"
                         % (text, exc)) from exc
    return day, day.weekday()


def _as_int(value, what):
    if isinstance(value, bool):
        raise ValueError("%s must be an integer, got %r" % (what, value))
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        raise ValueError("%s must be an integer, got %r" % (what, value)) from None


def load_demand(path):
    """Load a demand CSV into {(origin, dest, dow, hour): avg_vph}.

    Fails loud on a missing file, missing columns, non-integer
    node/hour cells, an hour outside 0..23, a negative or
    non-finite average, a bad day_of_week cell, or a duplicated
    (origin, dest, dow, hour) row.
    """
    demand = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("demand file is empty: %s" % path)
        fields = [h.strip() for h in reader.fieldnames]
        required = ["origin_node_id", "destination_node_id",
                    "day_of_week", "hour", "average_demand_vph"]
        for col in required:
            if col not in fields:
                raise ValueError("demand file %s is missing column %s"
                                 % (path, col))
        for line_no, raw in enumerate(reader, start=2):
            row = {}
            for key, value in raw.items():
                if key is None:
                    continue
                row[key.strip()] = value.strip() if isinstance(
                    value, str) else value
            if all(v in (None, "") for v in row.values()):
                continue
            origin = _as_int(row["origin_node_id"], "origin_node_id")
            dest = _as_int(row["destination_node_id"],
                           "destination_node_id")
            dow = parse_day_of_week(row["day_of_week"])
            hour = _as_int(row["hour"], "hour")
            if not 0 <= hour <= 23:
                raise ValueError("demand file line %d: hour %d outside "
                                 "0..23" % (line_no, hour))
            try:
                avg = float(row["average_demand_vph"])
            except (TypeError, ValueError):
                raise ValueError("demand file line %d: bad "
                                 "average_demand_vph %r"
                                 % (line_no, row["average_demand_vph"])) from None
            if avg < 0 or avg != avg or avg in (float("inf"), float("-inf")):
                raise ValueError("demand file line %d: average_demand_vph "
                                 "must be finite and non-negative, got %r"
                                 % (line_no, avg))
            key = (origin, dest, dow, hour)
            if key in demand:
                raise ValueError("demand file line %d: duplicate row for "
                                 "OD %d->%d, dow %d, hour %d"
                                 % (line_no, origin, dest, dow, hour))
            demand[key] = avg
    return demand


def enumerate_profiles(n_groups):
    """All demand profiles for n_groups groups, in fixed order.

    Returns a list of (levels_tuple, probability) where levels are
    one of 'low'/'medium'/'high' per group, in sorted-group order.
    Group 0 varies slowest (itertools.product order), so profile
    index 0 is all-low and the last is all-high. Probabilities
    are products of LEVEL_PROBS and sum to 1.
    """
    if not isinstance(n_groups, int) or n_groups < 1:
        raise ValueError("n_groups must be a positive int, got %r"
                         % (n_groups,))
    profiles = []
    for levels in itertools.product(_LEVELS, repeat=n_groups):
        prob = 1.0
        for level in levels:
            prob *= LEVEL_PROBS[level]
        profiles.append((levels, prob))
    total = sum(prob for _levels, prob in profiles)
    if abs(total - 1.0) > 1e-12:
        raise ValueError("profile probabilities sum to %r, not 1" % total)
    return profiles


def level_count(level, base):
    """Cars in one hour for one OD pair at the given level."""
    if level == "medium":
        return round_half_up(base)
    if level == "high":
        return round_half_up(poisson_ppf(0.75, base))
    if level == "low":
        return round_half_up(poisson_ppf(0.25, base))
    raise ValueError("unknown demand level %r" % (level,))


def count_epoch_agents(demand, origin_group, sorted_groups, levels,
                       epoch, start_dow):
    """Total agents in one epoch for one profile, without building them.

    Same counting rules as generate_epoch_agents (level per origin
    group, hourly base for the epoch's day, missing rows = 0).
    Used to size the profile id stride before generation.
    """
    if len(levels) != len(sorted_groups):
        raise ValueError("levels %r do not match %d groups"
                         % (levels, len(sorted_groups)))
    group_pos = {group: i for i, group in enumerate(sorted_groups)}
    day_index = epoch // 4
    dow = (start_dow + day_index) % 7
    total = 0
    for (origin, dest) in sorted({(o, d) for (o, d, _w, _h)
                                  in demand}):
        if origin not in origin_group:
            raise ValueError("demand origin %d has no group assignment"
                             % origin)
        level = levels[group_pos[origin_group[origin]]]
        for hour_offset in range(6):
            hour_of_day = (epoch * 6 + hour_offset) % 24
            base = demand.get((origin, dest, dow, hour_of_day), 0.0)
            if base > 0:
                total += level_count(level, base)
    return total


def generate_epoch_agents(demand, origin_group, sorted_groups, levels,
                          epoch, start_dow, id_base, stride,
                          profile_index):
    """Generate one profile's agents for one epoch.

    demand: dict from load_demand. origin_group: origin node id ->
    group label. sorted_groups: sorted unique group labels, so a
    level tuple indexes groups in that order. levels: one level
    per group. epoch: epoch index k, starting at k * 360 minutes.
    start_dow: weekday of epoch 0's day, 0 = Monday. id_base and
    stride: see ids.py (profile p's ids start at
    id_base + profile_index * stride). Returns a list of agent
    dicts in deterministic order (OD sorted, hour ascending,
    departure ascending), each with agent_id, origin_node_id,
    destination_node_id and departure_time_in_min, all departures
    in [epoch_start, epoch_start + 360).
    """
    if len(levels) != len(sorted_groups):
        raise ValueError("levels %r do not match %d groups"
                         % (levels, len(sorted_groups)))
    group_pos = {group: i for i, group in enumerate(sorted_groups)}
    epoch_start = epoch * 360
    day_index = epoch // 4
    dow = (start_dow + day_index) % 7
    agents = []
    counter = 0
    for (origin, dest) in sorted({(o, d) for (o, d, _w, _h)
                                  in demand}):
        if origin not in origin_group:
            raise ValueError("demand origin %d has no group assignment"
                             % origin)
        level = levels[group_pos[origin_group[origin]]]
        for hour_offset in range(6):
            hour_of_day = (epoch * 6 + hour_offset) % 24
            base = demand.get((origin, dest, dow, hour_of_day), 0.0)
            count = level_count(level, base) if base > 0 else 0
            hour_start = epoch_start + hour_offset * 60
            for i in range(count):
                departure = hour_start + i * 60.0 / count
                if not (epoch_start <= departure < epoch_start + 360):
                    raise ValueError("generated departure %r outside "
                                     "epoch %d" % (departure, epoch))
                agents.append({
                    "agent_id": id_base + profile_index * stride + counter,
                    "origin_node_id": origin,
                    "destination_node_id": dest,
                    "departure_time_in_min": departure,
                })
                counter += 1
    return agents
