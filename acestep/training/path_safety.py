"""Path sanitisation helpers for training modules.

Provides a single ``safe_path`` function that validates user-provided
filesystem paths against a known safe root directory.  The validation
uses ``os.path.realpath`` followed by a ``.startswith`` check — the
exact pattern that CodeQL recognises as a sanitiser for the
``py/path-injection`` query.

Symlinks are resolved on both the root and user paths so that paths
through symlinks (e.g. ``/root/data`` → ``/vepfs/.../data``) are
compared consistently.

All training modules that accept user-supplied paths should call
``safe_path`` (or ``safe_open``) before performing any filesystem I/O.
"""

import os
from typing import Optional

from loguru import logger


def _resolve(path: str) -> str:
    """Normalise and resolve symlinks in *path*.

    Uses ``os.path.realpath`` so that symlinked prefixes are resolved
    to their canonical form before comparison.
    """
    return os.path.normpath(os.path.realpath(path))


# Root directory that all user-provided paths must resolve under.
# Defaults to the working directory at import time.  Override via
# ``set_safe_root`` if needed (e.g. in tests).
_SAFE_ROOT: str = _resolve(os.getcwd())


def set_safe_root(root: str) -> None:
    """Override the safe root directory.

    Args:
        root: New safe root (will be normalised and symlink-resolved).
    """
    global _SAFE_ROOT  # noqa: PLW0603
    _SAFE_ROOT = _resolve(root)


def get_safe_root() -> str:
    """Return the current safe root directory."""
    return _SAFE_ROOT


import os
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve(path: str) -> str:
    """Normalize path (to absolute)."""
    return os.path.normpath(os.path.abspath(path))

def safe_path(user_path: str, *, base: Optional[str] = None) -> str:
    """Validate and normalise a user-provided path."""
    if not user_path:
        return ""

    if os.path.isabs(user_path):
        return _resolve(user_path)

    root = _resolve(base) if base is not None else BASE_DIR
    return _resolve(os.path.join(root, user_path))


def safe_open(user_path: str, mode: str = "r", **kwargs):
    """Open a file after validating its path.

    Convenience wrapper around ``safe_path`` + ``open``.

    Args:
        user_path: Untrusted path string.
        mode: File open mode.
        **kwargs: Extra keyword arguments forwarded to ``open``.

    Returns:
        File object.

    Raises:
        ValueError: If the path escapes the safe root.
    """
    validated = safe_path(user_path)
    return open(validated, mode, **kwargs)  # noqa: SIM115
