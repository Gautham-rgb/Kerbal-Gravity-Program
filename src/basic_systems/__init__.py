"""Kerbal Gravity Program — core simulation and planning library.

This package provides the physics, orbital mechanics, and mission-planning
core used by the ``kgrp`` REPL (see :mod:`CLI_use`). The heavy 3D renderer
(``basic_systems.renderer``) is an optional extra; import it only when you
actually want to draw something.
"""

from importlib import resources
import os

__version__ = "0.2.0"


def example_system_path(name: str) -> str:
    """Return the absolute path to a bundled example system JSON.

    Example systems shipped with the package live under
    ``basic_systems/data``. Pass either a bare name (``"planets_ksp"``) or a
    full filename (``"planets_ksp.json"``); if the file isn't found on disk
    the bundled copy is returned.

    Useful for getting started without wrangling your own system file::

        from basic_systems import example_system_path
        # then in the REPL:  load system <that path>
    """
    if not name.endswith(".json"):
        name = name + ".json"
    try:
        return str(resources.files("basic_systems") / "data" / name)
    except Exception:  # pragma: no cover - fallback for unusual environments
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)
