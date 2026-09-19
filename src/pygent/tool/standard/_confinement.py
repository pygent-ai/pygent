"""Workspace write confinement via Linux Landlock (LSM) for interactive
terminal process trees.

Linux Landlock (5.13+, Python 3.11+) allows an unprivileged process to
restrict its own filesystem access.  Once applied, the restriction affects
the current process tree: the shell started by ``PtyProcess`` can read and
execute the system it needs to start (``/bin/bash``, dynamic linker and
shared libraries, ``/etc`` configuration) but can only *write* inside the
designated workspace root.

This is *workspace write confinement*, not a full sandbox: reads and
executes stay allowed across the whole filesystem, and network, IPC,
``/dev``, ``/proc`` and process access are not restricted.  The
``sandbox_profile="workspace"`` claim is therefore accurate only about
filesystem mutation, never about general containment of the process.

The confinement is best-effort: if Landlock is not available (older kernel,
missing ``os.landlock_*``, or ``CAP_SYS_ADMIN`` required for the first
user in a user namespace) the call silently returns ``False`` and the caller
can decide whether to fall back or refuse.
"""

from __future__ import annotations

import os


def _landlock_available() -> bool:
    """Return ``True`` when the Python ``os.landlock_*`` API is present."""
    return hasattr(os, "landlock_create_ruleset") and callable(
        os.landlock_create_ruleset  # type: ignore[attr-defined]
    )


def confine_workspace(workspace_root: str) -> bool:
    """Apply Landlock filesystem confinement to the calling process.

    After this call the process may write, create or remove files only under
    *workspace_root* (recursively).  Reads and executes are allowed across the
    whole filesystem so the shell binary, dynamic loader and ``/etc``
    configuration remain usable, while nothing outside the workspace can be
    modified.  ``cd ..`` from the workspace therefore cannot damage files
    outside it.

    Returns ``True`` when confinement was applied; ``False`` if Landlock
    is not available or the kernel refused the ruleset.
    """
    if not _landlock_available():
        return False

    landlock = os  # the os.landlock_* functions live on the os module

    handled_fs = (
        landlock.LANDLOCK_ACCESS_FS_EXECUTE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_WRITE_FILE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_READ_FILE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_READ_DIR  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_REMOVE_DIR  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_REMOVE_FILE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_MAKE_DIR  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_MAKE_REG  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_MAKE_SYM  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_TRUNCATE  # type: ignore[attr-defined]
    )
    read_exec = (
        landlock.LANDLOCK_ACCESS_FS_EXECUTE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_READ_FILE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_READ_DIR  # type: ignore[attr-defined]
    )

    try:
        ruleset_fd = landlock.landlock_create_ruleset(  # type: ignore[attr-defined]
            handled_fs,
            0,  # handled_access_net — none for now
        )
    except OSError:
        return False

    try:
        # Whole filesystem: read + execute only.  The shell needs this to
        # start (binary, dynamic loader, libraries, /etc configuration).
        landlock.landlock_add_rule(  # type: ignore[attr-defined]
            ruleset_fd,
            landlock.LANDLOCK_RULE_PATH_BENEATH,  # type: ignore[attr-defined]
            "/",
            read_exec,
        )
        # Workspace: full access.  More specific rule, so it wins inside.
        landlock.landlock_add_rule(  # type: ignore[attr-defined]
            ruleset_fd,
            landlock.LANDLOCK_RULE_PATH_BENEATH,  # type: ignore[attr-defined]
            workspace_root,
            handled_fs,
        )
    except (OSError, PermissionError):
        os.close(ruleset_fd)
        return False

    try:
        landlock.landlock_restrict_self(  # type: ignore[attr-defined]
            ruleset_fd, handled_fs
        )
    except OSError:
        os.close(ruleset_fd)
        return False

    os.close(ruleset_fd)
    return True


def _probe_confine_supported(workspace_root: str) -> bool:
    """Probe whether Landlock confinement *can* be applied for the workspace.

    Creates the ruleset and adds both rules in the calling (parent) process
    without calling ``landlock_restrict_self``, so the probe itself does not
    confine anything.  The child re-runs the same ruleset creation before
    ``execvp``; a successful probe here is a strong signal the child will
    succeed there.
    """
    if not _landlock_available():
        return False

    landlock = os

    handled_fs = (
        landlock.LANDLOCK_ACCESS_FS_EXECUTE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_WRITE_FILE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_READ_FILE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_READ_DIR  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_REMOVE_DIR  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_REMOVE_FILE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_MAKE_DIR  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_MAKE_REG  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_MAKE_SYM  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_TRUNCATE  # type: ignore[attr-defined]
    )
    read_exec = (
        landlock.LANDLOCK_ACCESS_FS_EXECUTE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_READ_FILE  # type: ignore[attr-defined]
        | landlock.LANDLOCK_ACCESS_FS_READ_DIR  # type: ignore[attr-defined]
    )

    try:
        ruleset_fd = landlock.landlock_create_ruleset(  # type: ignore[attr-defined]
            handled_fs, 0
        )
    except OSError:
        return False
    try:
        landlock.landlock_add_rule(  # type: ignore[attr-defined]
            ruleset_fd,
            landlock.LANDLOCK_RULE_PATH_BENEATH,  # type: ignore[attr-defined]
            "/",
            read_exec,
        )
        landlock.landlock_add_rule(  # type: ignore[attr-defined]
            ruleset_fd,
            landlock.LANDLOCK_RULE_PATH_BENEATH,  # type: ignore[attr-defined]
            workspace_root,
            handled_fs,
        )
    except (OSError, PermissionError):
        os.close(ruleset_fd)
        return False
    os.close(ruleset_fd)
    return True


__all__ = ["_landlock_available", "_probe_confine_supported", "confine_workspace"]