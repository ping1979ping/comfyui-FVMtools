"""Progress bookkeeping for the batch loader.

The loader hands out one file per queue run, so it needs to remember what it
already handed out. A plain integer index would be the obvious way and it is the
wrong one: when the pipeline is set to MOVE its files into pass/reject folders,
the source listing shrinks under the loader's feet and an index skips every
other picture. So the state records **which files are done**, by name, and the
next file is simply the first one not in that set. That works whether files stay
put, get copied, or get moved away, and it survives a ComfyUI restart.

The state lives in a JSON file inside the source directory, so a batch is
resumable, inspectable and deletable with the folder it belongs to. Several
independent runs over one directory are kept apart by a ``tracker`` key.
"""

from __future__ import annotations

import json
import os
import tempfile

#: Written into the source directory. Dot-prefixed so it sorts out of the way,
#: and skipped by the loader's own file scan.
STATE_FILENAME = ".fvm_batch_state.json"

#: Extensions the loader treats as images. Everything else in the directory —
#: text files, sidecars, the state file itself — is ignored, per the contract
#: that non-image files are simply not part of the batch.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")


def state_path(directory):
    return os.path.join(directory, STATE_FILENAME)


def load_state(directory):
    """Read the whole state file. Returns ``{}`` when absent or unreadable.

    A corrupt state file must not stop a batch: losing the progress marker costs
    a re-run, raising here would cost the whole queue.
    """
    try:
        with open(state_path(directory), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(directory, state):
    """Write the state file atomically.

    Atomic because the queue may be interrupted at any moment, and a half-written
    state file would read as "nothing done" and re-run the whole directory.
    """
    target = state_path(directory)
    handle = None
    try:
        fd, temp_path = tempfile.mkstemp(dir=directory, prefix=".fvm_batch_", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=1, ensure_ascii=False)
        os.replace(temp_path, target)
        return True
    except OSError:
        return False


def get_done(directory, tracker="default"):
    """Filenames already handed out for this tracker, as a set."""
    entry = load_state(directory).get(tracker)
    if isinstance(entry, dict):
        done = entry.get("done")
    else:
        done = entry
    return set(done) if isinstance(done, list) else set()


def set_done(directory, done, tracker="default", extra=None):
    """Replace this tracker's done-list, leaving other trackers untouched."""
    state = load_state(directory)
    entry = {"done": sorted(done)}
    if extra:
        entry.update(extra)
    state[tracker] = entry
    return save_state(directory, state)


def clear_done(directory, tracker="default"):
    """Forget this tracker's progress — the loader's ``reset`` switch."""
    state = load_state(directory)
    state.pop(tracker, None)
    if state:
        return save_state(directory, state)
    try:                                   # nothing left to remember
        os.remove(state_path(directory))
    except OSError:
        pass
    return True


def list_images(directory, include_subdirs=False, sort_by="name", exclude_dirs=()):
    """Image files in ``directory``, as paths relative to it, in stable order.

    Relative rather than absolute so the state file stays portable and readable,
    and so a moved parent folder does not invalidate a half-finished batch.

    ``exclude_dirs`` names top-level subfolders to skip when walking — the
    loader's own pass/reject targets, so sorted pictures do not come back in.
    """
    if not os.path.isdir(directory):
        return []

    found = []
    if include_subdirs:
        skip = {os.path.normcase(d.strip().strip("/\\")) for d in exclude_dirs if d and d.strip()}
        for root, dirnames, filenames in os.walk(directory):
            # A batch writes its results into subfolders of its own source
            # directory; walking into them would feed the output back in.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if root == directory:
                dirnames[:] = [d for d in dirnames if os.path.normcase(d) not in skip]
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    full = os.path.join(root, filename)
                    found.append(os.path.relpath(full, directory).replace("\\", "/"))
    else:
        try:
            entries = os.listdir(directory)
        except OSError:
            return []
        for filename in entries:
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            if os.path.isfile(os.path.join(directory, filename)):
                found.append(filename)

    if sort_by == "modified":
        def mtime(name):
            try:
                return os.path.getmtime(os.path.join(directory, name))
            except OSError:
                return 0.0
        found.sort(key=lambda name: (mtime(name), name.lower()))
    else:
        found.sort(key=lambda name: name.lower())
    return found


def next_file(directory, done, include_subdirs=False, sort_by="name", loop=False,
              exclude_dirs=()):
    """Pick the next file to hand out.

    Returns ``(filename, done, wrapped)``. ``filename`` is None when the batch is
    finished and ``loop`` is off. ``wrapped`` reports that the done-list was
    cleared and the batch restarted, so the caller can say so in its status line.
    """
    available = list_images(directory, include_subdirs, sort_by, exclude_dirs)
    remaining = [name for name in available if name not in done]

    wrapped = False
    if not remaining:
        if not loop or not available:
            return None, done, False
        # Round two: everything has been handed out at least once. Only sensible
        # when files stay put — with move mode there is nothing left to loop over.
        done, remaining, wrapped = set(), available, True

    return remaining[0], done, wrapped


def progress(directory, done, include_subdirs=False, sort_by="name", exclude_dirs=()):
    """Counts for the node's status line: ``(processed, total, remaining)``.

    ``processed`` is simply how many files have been handed out — so a caller
    that has just added the current file to ``done`` reads its own position, and
    one that has not reads how many came before it.

    ``total`` counts files still in the directory plus those already moved out,
    so the denominator does not shrink as a move-mode batch empties the folder.
    """
    available = list_images(directory, include_subdirs, sort_by, exclude_dirs)
    present = set(available)
    moved_away = [name for name in done if name not in present]
    total = len(available) + len(moved_away)
    remaining = len([name for name in available if name not in done])
    return min(len(done), total), total, remaining
