"""Shared fixtures for the KGRP test suite."""

from __future__ import annotations

import math

import pytest

from basic_systems import example_system_path
from basic_systems.builder import System
from basic_systems.orbit_pred import Spacecraft
from CLI_use.ticket import Ticket


def _find(node, key, by="name"):
    if getattr(node, by, None) == key:
        return node
    for moon in getattr(node, "moons", []):
        found = _find(moon, key, by)
        if found:
            return found
    return None


@pytest.fixture
def system():
    return System.load(example_system_path("planets_ksp"))


@pytest.fixture
def kerbin(system):
    body = _find(system.root, "Kerbin")
    assert body is not None, "Kerbin not found in bundled system"
    return body


@pytest.fixture
def jool(system):
    body = _find(system.root, "6")
    if body is None:
        body = _find(system.root, "Jool")
    assert body is not None, "Jool not found in bundled system"
    return body


def _circular_vessel(parent, altitude, dry=1000.0, wet=4000.0, name="Vessel"):
    r = parent.radius + altitude
    v = math.sqrt(parent.mu / r)
    sc = Spacecraft(
        name=name,
        r0=[r, 0.0, 0.0],
        v0=[0.0, v, 0.0],
        t0=0.0,
        parent=parent,
        dry_mass=dry,
        wet_mass=wet,
        hull_mesh=None,
    )
    sc.add_engine(20000.0, 300.0, [0.0, 0.0, 0.0], name="engine")
    return sc


@pytest.fixture
def kerbin_ticket(kerbin, system):
    return Ticket("K", _circular_vessel(kerbin, 200_000.0), system, name="K")


@pytest.fixture
def jool_ticket(jool, system):
    sc = _circular_vessel(jool, jool.radius * 0.2 + jool.atm_height, dry=1000.0, wet=20000.0)
    return Ticket("J", sc, system, name="J")


@pytest.fixture
def pol(system):
    body = _find(system.root, "Pol")
    assert body is not None, "Pol not found in bundled system"
    return body


@pytest.fixture
def pol_ticket(pol, system):
    sc = _circular_vessel(pol, 10_000.0, dry=1000.0, wet=20000.0)
    return Ticket("P", sc, system, name="P")
