"""Environment configuration with optional ``.env`` file support.

``load_env()`` reads a local ``.env`` file into the environment (project-root
convention: ``python-dotenv`` searches the current working directory upward).
Existing environment variables are never overridden, so ``.env`` is purely
additive — handy for checkout-local setup without touching the shell profile.
After ``load_env()`` has run, plain ``os.getenv`` sees ``.env`` values.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def load_env() -> bool:
    """Read a ``.env`` file into the environment once per process.

    Returns ``True`` when a file was found and loaded.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv not installed: env vars only
        return False
    return load_dotenv(override=False)


__all__ = ["load_env"]
