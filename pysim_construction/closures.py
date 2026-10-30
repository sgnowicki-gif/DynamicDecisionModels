# Per-segment construction closures for stepwise v3.
#
# This replaces the v2 project state machine entirely. There are
# no projects, no scheduled starts, no clear-first rule and no
# state carried across segments. Each run_segment call receives
# a dict mapping each closed directed link to the minutes left
# in its construction, and that dict is the authority for that
# segment only:
#
# - A link with minutes_left > 0 is barred for ENTRY from the
#   segment start for min(minutes_left, step_minutes) minutes.
#   It reopens at segment_start + minutes_left if that falls
#   inside the segment, so a mid-segment reopening takes
#   effect for entry and rerouting from the next 0.1-minute
#   step onward. Otherwise it is still barred at segment end.
# - A link absent from the dict, or with minutes_left == 0,
#   is open for the whole segment.
# - Cars already ON a barred link finish traversing it; they
#   are never ejected. Only entry is barred, exactly as in v2.
# - The countdown starts at the segment start regardless of
#   cars on the link. This is Eli's approved simplification:
#   v2 waited for each link to clear before starting the
#   construction clock, v3 does not.
# - The next segment's dict is passed fresh by the caller. The
#   outside tracker subtracts the elapsed step_minutes; PySim
#   does not carry construction across segments, and it does
#   not report finish times (finish time = segment start +
#   minutes_left, computed by the caller, not by PySim).
#
# The loader (loading.py) only needs two methods from this
# object: barred_link_nos() and update(world, current_time),
# called at the start of every step. has_active_projects and
# finalize exist only so the unused v2 network_loading path
# would not crash if ever called; neither is used by sessions.
class SegmentClosures:
    def __init__(self, world, closed_links, segment_start_time):
        # closed_links: {(from_node_id, to_node_id): minutes_left}
        # already validated by session.py (known links, finite
        # non-negative numbers). Reopen times are absolute.
        self.reopen = {}
        for (from_node, to_node), minutes_left in closed_links.items():
            if minutes_left > 0:
                link_no = world.link_map[(from_node, to_node)]
                self.reopen[link_no] = (segment_start_time +
                                        float(minutes_left))
        self._barred = set()

    def barred_link_nos(self):
        return self._barred

    def update(self, world, current_time):
        # A step at exactly the reopen time is open: the 1e-9
        # margin absorbs float step-time error (step times are
        # step_count * 0.1) without moving a genuine reopening
        # by a whole step.
        self._barred = set(
            link_no for link_no, reopen in self.reopen.items()
            if current_time < reopen - 1e-9)

    def has_active_projects(self):
        return bool(self._barred)

    def finalize(self):
        self._barred = set()
