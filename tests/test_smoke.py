"""Smoke + unit tests for the KGRP public API."""

from __future__ import annotations

import math

import pytest

from basic_systems import example_system_path
from basic_systems.orbit_pred import parse_time_string, Spacecraft
from CLI_use.go import (
    apply_go,
    apply_go_interplanetary,
    go_plan,
)
from CLI_use.ticket import Ticket


# --- Time parsing -----------------------------------------------------------

def test_parse_duration():
    assert parse_time_string("21600") == 21600.0
    assert parse_time_string("2d 5h") == 2 * 21600.0 + 5 * 3600.0
    assert parse_time_string("90s") == 90.0


def test_parse_calendar():
    # Year 1, Day 1, 00:00:00 == UT 0 in Kerbal time.
    assert parse_time_string("Year 1, Day 1") == 0.0


def test_parse_unknown_unit_is_friendly():
    for bad in ("5 eras", "eras", "5 bananas"):
        with pytest.raises(ValueError):
            parse_time_string(bad)


def test_parse_relative_to_base():
    assert parse_time_string("1d", base=100.0) == 100.0 + 21600.0


# --- Bundled data -----------------------------------------------------------

def test_example_system_path_resolves():
    import os

    path = example_system_path("planets_ksp")
    assert os.path.exists(path)


# --- go planning ------------------------------------------------------------

def test_go_escape_needs_burn(kerbin_ticket):
    plan = go_plan(kerbin_ticket, "current", "escape")
    assert plan.burn_count >= 1


def test_go_same_orbit_no_burn(kerbin_ticket):
    plan = go_plan(kerbin_ticket, "current", "current")
    assert plan.burn_count == 0


def test_go_unknown_orbit_quips(kerbin_ticket):
    with pytest.raises(ValueError):
        go_plan(kerbin_ticket, "current", "not_a_real_orbit")


def test_apply_go_schedules(kerbin_ticket):
    applied = apply_go(kerbin_ticket, "current", "escape", mode="gradual")
    assert applied.burn_count >= 1
    assert len(kerbin_ticket.events) >= 1


# --- interplanetary ---------------------------------------------------------

def test_go_interplanetary(jool_ticket):
    plan = apply_go_interplanetary(jool_ticket, "moon:3", mode="instant")
    assert plan.total_dv > 0.0
    assert plan.burn_count >= 2
    parent = jool_ticket.spacecraft.orbit.parent
    assert parent is not None
    assert parent.name != "Jool"
    assert jool_ticket.spacecraft.orbit.eccen < 1.0
    assert jool_ticket.spacecraft.orbit.semi_major_axis > 0.0


def test_go_interplanetary_unknown_target(jool_ticket):
    with pytest.raises(ValueError):
        apply_go_interplanetary(jool_ticket, "moon:999", mode="gradual")


# --- tickets / events -------------------------------------------------------

def test_ticket_coast_and_advance(kerbin_ticket):
    from CLI_use.ticket import CoastEvent

    start = kerbin_ticket.cursor_ut
    kerbin_ticket.add_event(CoastEvent(start, start + 1000.0))
    kerbin_ticket.advance_to(start + 500.0)
    assert kerbin_ticket.cursor_ut == pytest.approx(start + 500.0)
