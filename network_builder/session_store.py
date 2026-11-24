"""Disk-spilling frontier session store (network_builder v3).

Why this module exists. v2 kept every node's traffic session in
RAM forever: at the end of epoch 3 that is ~1,285 sessions,
~28 GB on the Sioux Falls network, and the run dies of memory
around epoch 3 on a 32 GB machine before time becomes the
binding constraint. v3 spills sessions to per-session pickle
files under <output_dir>/sessions/ and keeps only references
in RAM: a node's traffic is loaded from disk when the node is
expanded (or merged into) and freed again afterwards, so live
memory is roughly workers x session size plus one layer's
transient sessions, not the whole tree.

Keys and references. put(session) writes one file,
sess_<key>.pkl, and returns a fresh integer key. A key is kept
alive by references: add_ref(key) when a node (or the
transition cache) starts pointing at it, release(key) when the
pointer goes away; the file is deleted when the count reaches
zero. Node references and cache references are tracked
separately (cache_refs): the transition cache holds one
reference per cached transition, because a cached end session
can still be needed after every branch node that pointed at
it has been merged away. Checkpoint state records NODE
references only (checkpoint_state): the cache is memoisation
and is dropped on resume, so a cache-only key must not keep
its file alive across a resume -- the resumed run recomputes
that transition instead.

Backends. With a directory, files are written atomically
(temp file + os.replace) and loaded with pickle. Without a
directory (output_dir is None: in-memory-only builds, used by
tests), sessions live in a dict; put() stores a deepcopy, so
callers can never mutate a stored session through the object
they handed in. The two backends share the reference-count
logic, so one code path serves both.

Resume. The checkpoint stores {"next_key": int, "refcounts":
{key: node_refs}, "pinned": [key, ...]} (see checkpoint.py).
restore_state rebuilds the counts and the pin set;
sweep_orphans deletes sess files the restored state does not
reference (orphans from a layer that was killed mid-write).
When a run resumes into a DIFFERENT output dir than the
checkpoint's, build_network copies the checkpoint's
sessions/ directory into the new output dir first, so the
resumed run owns its files.

Pinning (v3.2). A node's session file must survive until the
next checkpoint stops referencing it. The bug v3.2 fixes:
after a node was expanded, release() deleted its file as
soon as the refcount hit zero -- but the checkpoint written
at the previous layer boundary still listed that node, so a
crash mid-layer left the checkpoint pointing at deleted
files and the resume died later at load_traffic. The store
therefore keeps a pinned set: the traffic keys of the nodes
in the latest checkpoint. release(), cache_release() and
sweep_orphans() NEVER delete a pinned file (refcounts still
decrement, and a pinned key whose count reaches zero keeps
its entry at zero so the file stays loadable). The pin set
is updated when a new checkpoint is saved and persisted in
the checkpoint's session_store state, so a second mid-layer
crash is resumable too; the sweep after each checkpoint
save deletes files that are unpinned AND unreferenced, so
disk stays bounded to roughly two layers' sessions. A
checkpoint written by v3 (no "pinned" entry) restores with
its refcount keys pinned, which are exactly its frontier
keys, so v3 checkpoints stay loadable and protected.
"""

import copy
import os
import pickle
import shutil
import tempfile
import time

SESSIONS_DIRNAME = "sessions"


class SessionStore:
    """Reference-counted store of traffic sessions, disk or memory."""

    def __init__(self, directory=None):
        self.directory = directory
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
        self._next_key = 1
        self._refcounts = {}
        self._cache_refs = {}
        self._memory = {}
        # v3.2: keys pinned by the latest checkpoint (see the
        # module docstring). A pinned file is never deleted by
        # release(), cache_release() or sweep_orphans(), even
        # at zero references.
        self._pinned = set()

    def _path(self, key):
        return os.path.join(self.directory, "sess_%d.pkl" % key)

    def put(self, session):
        """Store session, return a fresh key with zero references.

        The caller adds a reference for every pointer it creates
        (add_ref for a node, cache_hold for a cache entry).
        """
        key = self._next_key
        self._next_key += 1
        self._refcounts[key] = 0
        if self.directory is not None:
            path = self._path(key)
            fd, tmp_path = tempfile.mkstemp(
                prefix="sess_%d." % key, suffix=".tmp",
                dir=self.directory)
            try:
                with os.fdopen(fd, "wb") as fh:
                    pickle.dump(session, fh,
                                protocol=pickle.HIGHEST_PROTOCOL)
                # Windows (WinError 32): another process -- usually
                # antivirus / indexer -- can hold the just-written
                # temp file open for a moment, and os.replace then
                # fails. The file is fully written and closed here,
                # so retry a few times instead of crashing the run.
                for _attempt in range(10):
                    try:
                        os.replace(tmp_path, path)
                        break
                    except PermissionError:
                        if _attempt == 9:
                            raise
                        time.sleep(0.25)
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        else:
            self._memory[key] = copy.deepcopy(session)
        return key

    def get(self, key):
        """Load the session for key; a fresh object every call.

        A missing key or file fails loud: a live reference must
        always resolve, so a missing file means the store and
        its references have diverged (a bug, not a cache miss).
        """
        if key not in self._refcounts:
            raise ValueError(
                "session store: unknown session key %r" % (key,))
        if self.directory is not None:
            path = self._path(key)
            if not os.path.isfile(path):
                raise ValueError(
                    "session store: session file missing for key %r "
                    "(%s)" % (key, path))
            with open(path, "rb") as fh:
                return pickle.load(fh)
        return copy.deepcopy(self._memory[key])

    def has(self, key):
        return key in self._refcounts

    def file_exists(self, key):
        """True when the session for key is actually on disk
        (or in memory for the memory backend). Resume
        validation uses this, not has(): has() only checks
        the refcount dict, which says nothing about whether
        the file survived a crash."""
        if key not in self._refcounts:
            return False
        if self.directory is None:
            return key in self._memory
        return os.path.isfile(self._path(key))

    def session_path(self, key):
        """Disk path of the session file, or None for the
        memory backend. Used in resume-failure messages."""
        if self.directory is None:
            return None
        return self._path(key)

    def pinned_keys(self):
        """Sorted list of the keys pinned by the latest
        checkpoint (v3.2)."""
        return sorted(self._pinned)

    def set_pinned(self, keys):
        """Replace the pin set with the given keys.

        Called when a new checkpoint is saved, with the
        traffic keys of the new frontier nodes. This alone
        deletes nothing: the caller sweeps afterwards, so
        files that are now unpinned AND unreferenced get
        deleted then."""
        pinned = set()
        for key in keys:
            if not isinstance(key, int) or isinstance(key, bool) \
                    or key < 1:
                raise ValueError(
                    "session store: bad pinned key %r" % (key,))
            pinned.add(key)
        self._pinned = pinned

    def add_ref(self, key):
        if key not in self._refcounts:
            raise ValueError(
                "session store: add_ref on unknown key %r" % (key,))
        self._refcounts[key] += 1

    def release(self, key):
        """Drop one reference; delete the file at zero references.

        v3.2: a PINNED key is never deleted, even at zero
        references -- its file is still named by the latest
        checkpoint, so a crash mid-layer must be able to
        resume from it. The count still drops to zero and
        the entry is kept (at zero) so the file stays
        loadable through get()/has(); the file is deleted
        later by sweep_orphans(), once the key is unpinned
        by a newer checkpoint and still unreferenced.
        """
        if key not in self._refcounts:
            raise ValueError(
                "session store: release on unknown key %r" % (key,))
        self._refcounts[key] -= 1
        if self._refcounts[key] < 0:
            raise ValueError(
                "session store: negative refcount for key %r" % (key,))
        if self._refcounts[key] == 0:
            if key in self._pinned:
                self._cache_refs.pop(key, None)
                return
            del self._refcounts[key]
            self._cache_refs.pop(key, None)
            if self.directory is not None:
                try:
                    os.unlink(self._path(key))
                except OSError:
                    pass
            else:
                self._memory.pop(key, None)

    def cache_hold(self, key):
        """Add a reference held by the transition cache."""
        self.add_ref(key)
        self._cache_refs[key] = self._cache_refs.get(key, 0) + 1

    def cache_release(self, key):
        """Drop a reference held by the transition cache."""
        held = self._cache_refs.get(key, 0)
        if held < 1:
            raise ValueError(
                "session store: cache_release with no cache "
                "reference for key %r" % (key,))
        self._cache_refs[key] = held - 1
        if self._cache_refs[key] == 0:
            del self._cache_refs[key]
        self.release(key)

    def live_keys(self):
        return sorted(self._refcounts)

    def checkpoint_state(self):
        """State for the checkpoint: node references only.

        Cache references are excluded: the cache is dropped on
        resume, so a cache-only key records zero references
        here and its file is swept on resume. Keys with zero
        node references are omitted. v3.2: the pin set is
        recorded too ("pinned": the traffic keys of the
        frontier nodes in this checkpoint), so a resume
        restores the same protection against a second
        mid-layer crash.
        """
        refs = {}
        for key, total in self._refcounts.items():
            node_refs = total - self._cache_refs.get(key, 0)
            if node_refs > 0:
                refs[key] = node_refs
        return {"next_key": self._next_key, "refcounts": refs,
                "pinned": sorted(self._pinned)}

    def restore_state(self, state):
        """Restore counts from a checkpoint state dict."""
        if not isinstance(state, dict):
            raise ValueError("session store state must be a dict")
        if "next_key" not in state or "refcounts" not in state:
            raise ValueError(
                "session store state is missing next_key/refcounts")
        next_key = state["next_key"]
        refcounts = state["refcounts"]
        if not isinstance(next_key, int) or isinstance(next_key, bool) \
                or next_key < 1:
            raise ValueError(
                "session store state has a bad next_key %r" % (next_key,))
        if not isinstance(refcounts, dict):
            raise ValueError(
                "session store state refcounts must be a dict")
        restored = {}
        for raw_key, count in refcounts.items():
            key = int(raw_key)
            if not isinstance(count, int) or isinstance(count, bool) \
                    or count < 1:
                raise ValueError(
                    "session store state has a bad refcount for key "
                    "%r: %r" % (raw_key, count))
            restored[key] = count
        # v3.2: restore the pin set. A v3 checkpoint has no
        # "pinned" entry; its refcount keys ARE its frontier
        # keys, so pin those -- a v3 checkpoint resumes with
        # the same protection as a v3.2 one. A pinned key
        # missing from refcounts (it can only be at zero
        # references) is re-added at zero so its file stays
        # loadable through get().
        raw_pinned = state.get("pinned")
        if raw_pinned is None:
            pinned = set(restored)
        else:
            if not isinstance(raw_pinned, (list, tuple)):
                raise ValueError(
                    "session store state has a bad pinned set %r"
                    % (raw_pinned,))
            pinned = set()
            for raw_key in raw_pinned:
                if not isinstance(raw_key, int) \
                        or isinstance(raw_key, bool) or raw_key < 1:
                    raise ValueError(
                        "session store state has a bad pinned key "
                        "%r" % (raw_key,))
                pinned.add(raw_key)
        for key in pinned:
            if key not in restored:
                restored[key] = 0
        self._next_key = next_key
        self._refcounts = restored
        self._pinned = pinned
        self._cache_refs = {}
        if self.directory is None:
            # A memory-backend store cannot be restored from a
            # disk checkpoint: the sessions themselves are gone.
            raise ValueError(
                "session store: cannot restore a checkpoint into "
                "an in-memory store (resume needs an output_dir)")

    def sweep_orphans(self):
        """Delete sess files no live reference or pin points at.

        Disk backend only. Orphans come from a layer killed
        mid-write: keys at or past next_key, or keys whose
        references were all released, never get reused, so the
        files are dead weight. v3.2: a file is deleted only
        when it is BOTH unpinned and unreferenced -- a pinned
        key keeps its file even at zero references (its entry
        is kept at zero by release(), so it is found here and
        skipped), and a zero-reference unpinned entry is
        removed from the counts as its file is deleted.
        Returns the number deleted.
        """
        if self.directory is None:
            return 0
        deleted = 0
        for name in os.listdir(self.directory):
            if not (name.startswith("sess_")
                    and name.endswith(".pkl")):
                continue
            try:
                key = int(name[len("sess_"):-len(".pkl")])
            except ValueError:
                continue
            if key in self._pinned:
                continue
            if key not in self._refcounts \
                    or self._refcounts[key] == 0:
                try:
                    os.unlink(os.path.join(self.directory, name))
                    deleted += 1
                except OSError:
                    pass
                self._refcounts.pop(key, None)
                self._cache_refs.pop(key, None)
        return deleted

    def copy_from(self, source_dir):
        """Replace this store's directory with source_dir's files.

        Used when resuming into a different output dir than the
        checkpoint's: the resumed run copies the checkpoint's
        sessions/ directory so it owns its files. No-op when
        source and destination are the same directory.
        """
        if self.directory is None:
            raise ValueError(
                "session store: cannot copy sessions into an "
                "in-memory store")
        if not os.path.isdir(source_dir):
            return 0
        if os.path.abspath(source_dir) == os.path.abspath(self.directory):
            return 0
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory)
        shutil.copytree(source_dir, self.directory)
        count = len([name for name in os.listdir(self.directory)
                     if name.startswith("sess_")
                     and name.endswith(".pkl")])
        return count
