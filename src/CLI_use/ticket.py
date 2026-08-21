"""Mission timeline events.

The REPL and renderer should advance a :class:`Ticket`; they should never
directly mutate a spacecraft to perform a scheduled mission action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import json
from pathlib import Path

TICKET_EXTENSION = ".kgrp.json"

from basic_systems.RKF45 import RKF45
from basic_systems.builder import System
from basic_systems.orbit_pred import (
    Body,
    Engine,
    GuidanceState,
    ManeuverNode,
    Part,
    ResourceTank,
    Spacecraft,
    VesselSnapshot,
)


class Propagator(Protocol):
    def propagate(self, spacecraft: Spacecraft, start_ut: float, end_ut: float) -> tuple[np.ndarray, np.ndarray]: ...


class KeplerPropagator:
    """Two-body propagation through the spacecraft's current orbital elements."""

    def propagate(self, spacecraft: Spacecraft, start_ut: float, end_ut: float) -> tuple[np.ndarray, np.ndarray]:
        del start_ut
        return spacecraft.state_at(end_ut)


def _tank_to_json(tank: ResourceTank) -> dict:
    return {
        "resource": tank.resource,
        "amount": tank.amount,
        "capacity": tank.capacity,
        "crossfeed": tank.crossfeed
    }


def _tank_from_json(data: dict) -> ResourceTank:
    return ResourceTank(data["resource"], data["amount"], data["capacity"], data.get("crossfeed", True))


def _engine_to_json(engine: Engine) -> dict:
    return {
        "name": engine.name,
        "max_thrust": engine.max_thrust,
        "vacuum_isp": engine.vacuum_isp,
        "propellants": engine.propellants,
        "atmospheric_isp": engine.atmospheric_isp,
        "active": engine.active,
        "throttle": engine.throttle,
        "gimbal_limit": engine.gimbal_limit,
        "offset": engine.offset.tolist()
    }


def _engine_from_json(data: dict) -> Engine:
    return Engine(
        data["name"],
        data["max_thrust"],
        data["vacuum_isp"],
        data.get("propellants", {"LiquidFuel": 1.0}),
        data.get("atmospheric_isp"),
        data.get("active", True),
        data.get("throttle", 1.0),
        data.get("gimbal_limit", 0.0),
        np.array(data.get("offset", [0.0, 0.0, 0.0]))
    )


def _part_to_json(part: Part) -> dict:
    return {
        "identifier": part.identifier,
        "dry_mass": part.dry_mass,
        "attached": part.attached,
        "tanks": [_tank_to_json(tank) for tank in part.tanks],
        "engines": [_engine_to_json(engine) for engine in part.engines]
    }


def _part_from_json(data: dict) -> Part:
    return Part(
        data["identifier"],
        data["dry_mass"],
        [_tank_from_json(tank) for tank in data.get("tanks", [])],
        [_engine_from_json(engine) for engine in data.get("engines", [])],
        data.get("attached", True)
    )


def _guidance_to_json(guidance: GuidanceState) -> dict:
    return {
        "mode": guidance.mode,
        "attitude": guidance.attitude.tolist(),
        "angular_velocity": guidance.angular_velocity.tolist(),
        "target": guidance.target
    }


def _guidance_from_json(data: dict) -> GuidanceState:
    return GuidanceState(
        data.get("mode", "inertial"),
        np.array(data.get("attitude", [1.0, 0.0, 0.0, 0.0])),
        np.array(data.get("angular_velocity", [0.0, 0.0, 0.0])),
        data.get("target")
    )


def _vessel_to_json(spacecraft: Spacecraft, snapshot: VesselSnapshot, mesh_key: str | None) -> dict:
    return {
        "identifier": spacecraft.identifier,
        "name": spacecraft.name,
        "render_color": spacecraft.render_color,
        "mesh_key": mesh_key,
        "parent": snapshot.parent.identifier,
        "ut": snapshot.ut,
        "position": snapshot.position.tolist(),
        "velocity": snapshot.velocity.tolist(),
        "parts": [_part_to_json(part) for part in snapshot.parts],
        "guidance": _guidance_to_json(snapshot.guidance),
        "surface_body": snapshot.surface_body.identifier if snapshot.surface_body else None,
        "science_data": snapshot.science_data
    }


def _vessel_from_json(data: dict, bodies: dict[str, Body], mesh_factory: Callable[[str], Any] | None) -> Spacecraft:
    parent = bodies[data["parent"]]
    mesh_key = data.get("mesh_key")
    hull_mesh = mesh_factory(mesh_key) if mesh_factory and mesh_key else None
    spacecraft = Spacecraft(
        data["name"], np.array([1.0, 0.0, 0.0]), np.zeros(3), data["ut"], parent,
        0.0, 0.0, hull_mesh, identifier=data["identifier"], render_color=data.get("render_color", "#ffffff")
    )
    snapshot = VesselSnapshot(
        data["ut"], parent, np.array(data["position"]), np.array(data["velocity"]),
        [_part_from_json(part) for part in data["parts"]],
        _guidance_from_json(data["guidance"]),
        bodies[data["surface_body"]] if data.get("surface_body") else None,
        data.get("science_data", 0.0)
    )
    spacecraft.restore(snapshot)
    return spacecraft


def _collect_bodies(root: Body) -> dict[str, Body]:
    lookup: dict[str, Body] = {}

    def visit(body: Body) -> None:
        lookup[body.identifier] = body
        for moon in body.moons:
            visit(moon)

    visit(root)
    return lookup


@dataclass
class TicketContext:
    bodies: dict[str, Body] | None = None
    vessels: dict[str, Spacecraft] | None = None
    cursor_ut: float = float("inf")

    def body(self, identifier: str | None, start_ut: float = 0.0) -> Body | None:
        if identifier is None:
            return None
        found = self.bodies.get(identifier) if self.bodies else None
        if found is None and start_ut <= self.cursor_ut:
            raise KeyError(f"No body lookup entry for '{identifier}'.")
        return found

    def vessel(self, identifier: str | None, start_ut: float = 0.0) -> Spacecraft | None:
        if identifier is None:
            return None
        found = self.vessels.get(identifier) if self.vessels else None
        if found is None and start_ut <= self.cursor_ut:
            raise KeyError(f"No vessel lookup entry for '{identifier}'.")
        return found


class TicketEvent(ABC):
    """Base class for a scheduled mission action on a :class:`Ticket`.

    Subclasses set ``type`` (a string key used for (de)serialisation) and
    implement :meth:`update`, which advances the spacecraft's state through
    ``ut``. ``start_ut``/``end_ut`` bound the event window in seconds; events
    register themselves in :attr:`registry` on definition so they can be loaded
    from JSON.
    """

    type = "event"
    registry: dict[str, type["TicketEvent"]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        TicketEvent.registry[cls.type] = cls

    def __init__(self, start_ut: float, end_ut: float | None = None):
        self.start_ut = float(start_ut)
        self.end_ut = float(start_ut if end_ut is None else end_ut)
        if self.end_ut < self.start_ut:
            raise ValueError("Event end UT must not precede its start UT.")
        self.completed = False
        self.result: str = ""
        self.active_now = True

    def active(self, ut: float) -> bool:
        self.active_now = self.start_ut <= ut <= self.end_ut 
        return self.start_ut <= ut <= self.end_ut

    def finished(self, ut: float | None = None) -> bool:
        """True when the event is complete, or its window has elapsed by ``ut``."""
        if self.completed:
            return True
        if ut is None:
            return False
        return ut > self.end_ut

    def reset(self) -> None:
        self.completed = False
        self.result = ""

    @abstractmethod
    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        """Apply this event through ``ut``. ``ut`` can be beyond event end."""

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "TicketEvent":
        del ctx
        return cls(data["start_ut"], data.get("end_ut"))




class CoastEvent(TicketEvent):
    """A passive coast: do nothing between ``start_ut`` and ``end_ut`` (seconds)."""

    type = "coast"

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        del spacecraft
        self.completed = ut >= self.end_ut



class ManeuverEvent(TicketEvent):
    """An instantaneous delta-v burn applied at the node's UT (seconds).

    Wraps a :class:`~basic_systems.orbit_pred.ManeuverNode`; on execution the
    node's delta-v vector (m/s) is applied to the spacecraft in one impulse.
    """

    type = "maneuver"

    def __init__(self, maneuver_node: ManeuverNode):
        super().__init__(maneuver_node.ut)
        self.node = maneuver_node

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            operation = spacecraft.apply_impulse(self.node.delta_v_vector, self.start_ut)
            if not operation.success:
                raise RuntimeError(operation.reason)
            self.result = f"Applied {self.node.total_mag:.3f} m/s impulse."
            self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "delta_v": self.node.delta_v_vector.tolist(),
            "total_mag": self.node.total_mag
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "ManeuverEvent":
        del ctx
        delta_v = np.array(data["delta_v"])
        zero = np.zeros(3)
        node = ManeuverNode(data["start_ut"], delta_v, zero, zero, zero, data["total_mag"])
        return cls(node)


class PropagationEvent(TicketEvent):
    """Advance state from ``start_ut`` to ``end_ut`` using a propagator.

    Defaults to two-body Keplerian propagation; supply a custom ``propagator``
    (e.g. an n-body integrator) to override. Pure state advance, no manoeuvre.
    """

    type = "propagation"

    def __init__(self, start_ut: float, end_ut: float, propagator: Propagator | None = None):
        super().__init__(start_ut, end_ut)
        self.propagator = propagator or KeplerPropagator()
        self.last_ut = self.start_ut

    def reset(self) -> None:
        super().reset()
        self.last_ut = self.start_ut

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        target = min(ut, self.end_ut)
        if target > self.last_ut:
            position, velocity = self.propagator.propagate(spacecraft, self.last_ut, target)
            spacecraft.set_state(position, velocity, target)
            self.last_ut = target
        self.completed = target >= self.end_ut



class RKF45Event(TicketEvent):
    """Full n-body propagation over [``start_ut``, ``end_ut``] via RKF45.

    Integrates the spacecraft against its mission system's gravity field with
    the given ``tolerance``; more accurate than :class:`PropagationEvent` for
    multi-body environments at the cost of compute.
    """

    type = "rkf45"

    def __init__(self, start_ut: float, end_ut: float, system: System, tolerance: float = 1e-6,):
        super().__init__(start_ut, end_ut)
        self.tolerance = tolerance
        self.last_ut = start_ut
        self.integrator: RKF45 | None = None
        self.mission_system = system

    def reset(self) -> None:
        super().reset()
        self.last_ut = self.start_ut
        self.integrator = None

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        target = min(ut, self.end_ut)
        if target <= self.last_ut:
            return
        if self.integrator is None:
            system = self.mission_system
            gravity = lambda pos, root_mu, now: RKF45.n_body_grav_system(pos, root_mu, now, system)
            self.integrator = RKF45(spacecraft, gravity, self.tolerance, system)
        self.integrator.propagate(target - self.last_ut, 1.0, self.last_ut)
        self.last_ut = target
        self.completed = target >= self.end_ut

    def to_json(self) -> dict:
        return {
            "type": self.type, 
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "tol": self.tolerance
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "RKF45Event":
        del ctx
        return cls(data["start_ut"], data["end_ut"], data.get("tol", 1e-6))


class FiniteBurnEvent(TicketEvent):
    """A constant-direction burn from ``start_ut`` to ``end_ut`` (seconds).

    ``direction`` is the thrust unit vector; thrust and Isp come from the active
    engines, with ``pressure`` (Pa) used for atmospheric Isp interpolation.
    Propellant is consumed over the burn duration.
    """

    type = "finite-burn"

    def __init__(self, start_ut: float, end_ut: float, direction: np.ndarray, pressure: float = 0.0):
        super().__init__(start_ut, end_ut)
        self.direction = np.asarray(direction, dtype=float)
        self.pressure = pressure
        self.last_ut = start_ut

    def reset(self) -> None:
        super().reset()
        self.last_ut = self.start_ut

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        target = min(ut, self.end_ut)
        if target <= self.last_ut:
            return
        operation = spacecraft.advance_burn(target - self.last_ut, self.direction, self.last_ut, self.pressure)
        if not operation.success:
            raise RuntimeError(operation.reason)
        self.last_ut = target
        self.result = operation.reason
        self.completed = target >= self.end_ut or bool(operation.reason)

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "direction": self.direction.tolist(),
            "pressure": self.pressure
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "FiniteBurnEvent":
        del ctx
        return cls(data["start_ut"], data["end_ut"], np.array(data["direction"]), data.get("pressure", data.get("thrust", 0.0)))


class EngineControlEvent(TicketEvent):
    """Toggle engines and/or set throttle at ``ut`` (seconds).

    ``active`` turns the selected engines on/off; ``throttle`` sets the fraction
    (0-1) of max thrust; ``engine_indices`` restricts which engines are affected
    (default: all). At least one of ``active``/``throttle`` must be given.
    """

    type = "engine-control"

    def __init__(self, ut: float, active: bool | None = None, throttle: float | None = None, engine_indices: list[int] | None = None):
        if active is None and throttle is None:
            raise ValueError("Engine control requires an active state or throttle.")
        super().__init__(ut)
        self.target_active, self.throttle, self.engine_indices = active, throttle, engine_indices

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if self.completed or ut < self.start_ut:
            return
        indices = self.engine_indices or list(range(len(spacecraft.engines)))
        if self.target_active is not None:
            for index in indices:
                spacecraft.set_engine_state(index, bool(self.target_active))
        if self.throttle is not None:
            spacecraft.set_throttle(self.throttle, indices)
        self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "active": self.target_active,
            "throttle": self.throttle,
            "indices": self.engine_indices
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "EngineControlEvent":
        del ctx
        return cls(data["start_ut"], data.get("active"), data.get("throttle"), data.get("indices"))


class AttitudeEvent(TicketEvent):
    """Set the vessel's guidance mode/attitude at ``ut`` (seconds).

    ``mode`` selects the guidance behaviour, ``attitude`` is an optional attitude
    quaternion (must be non-zero) and ``target`` an optional target name.
    """

    type = "attitude"

    def __init__(self, ut: float, mode: str, attitude: np.ndarray | None = None, target: str | None = None):
        super().__init__(ut)
        self.mode, self.attitude, self.target = mode, attitude, target

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            spacecraft.set_guidance(self.mode, self.attitude, self.target)
            self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "mode": self.mode,
            "attitude": self.attitude.tolist() if self.attitude is not None else None,
            "target": self.target if self.target else None
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "AttitudeEvent":
        del ctx
        attitude = np.array(data["attitude"]) if data.get("attitude") is not None else None
        return cls(data["start_ut"], data["mode"], attitude, data.get("target"))


class RCSTranslationEvent(TicketEvent):
    """Apply a small MonoPropellant translation impulse at ``ut`` (seconds).

    ``delta_v_vector`` is the delta-v (m/s); ``isp`` is the RCS specific impulse
    in seconds (default 260). Consumes MonoPropellant from the vessel.
    """

    type = "rcs-translation"

    def __init__(self, ut: float, delta_v_vector: np.ndarray, isp: float = 260.0):
        super().__init__(ut)
        self.delta_v_vector = np.asarray(delta_v_vector, dtype=float)
        self.isp = isp

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            operation = spacecraft.apply_rcs_impulse(self.delta_v_vector, self.start_ut, self.isp)
            if not operation.success:
                raise RuntimeError(operation.reason)
            self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "delta_v": self.delta_v_vector.tolist(),
            "isp": self.isp
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "RCSTranslationEvent":
        del ctx
        return cls(data["start_ut"], np.array(data["delta_v"]), data.get("isp", 260.0))

class StageEvent(TicketEvent):
    """Detach the named parts from the vessel at ``ut`` (seconds).

    Each id in ``part_ids`` is detached (the ``"core"`` part cannot be staged),
    reducing mass; detached parts' resources are dropped with them.
    """

    type = "stage"

    def __init__(self, ut: float, part_ids: list[str]):
        super().__init__(ut)
        self.part_ids = part_ids

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            operation = spacecraft.stage(self.part_ids)
            if not operation.success:
                raise RuntimeError(operation.reason)
            self.result = ", ".join(operation.detached_parts)
            self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "part_id": self.part_ids
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "StageEvent":
        del ctx
        return cls(data["start_ut"], data["part_id"])



class ResourceTransferEvent(TicketEvent):
    """Transfer ``amount`` of ``resource`` to another vessel at ``ut`` (seconds).

    Moves propellant/resource from this vessel into ``target`` up to its spare
    capacity; fails if nothing transferable is available.
    """

    type = "resource-transfer"

    def __init__(self, ut: float, target: Spacecraft, resource: str, amount: float):
        super().__init__(ut)
        self.target, self.resource, self.amount = target, resource, amount

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            operation = spacecraft.transfer_resource(self.target, self.resource, self.amount)
            if not operation.success:
                raise RuntimeError(operation.reason)
            self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "resource": self.resource,
            "amount": self.amount,
            "target": self.target.identifier
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "ResourceTransferEvent":
        return cls(data["start_ut"], ctx.vessel(data["target"], data["start_ut"]), data["resource"], data["amount"]) #type: ignore


class DockEvent(TicketEvent):
    """Dock with ``target`` vessel at ``ut`` (seconds).

    Requires both vessels to share a reference body and be within docking range;
    merges the target's parts into this vessel and conserves momentum.
    """

    type = "dock"

    def __init__(self, ut: float, target: Spacecraft):
        super().__init__(ut)
        self.target = target

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            operation = spacecraft.dock(self.target, self.start_ut)
            if not operation.success:
                raise RuntimeError(operation.reason)
            self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "target": self.target.identifier
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "DockEvent":
        return cls(data["start_ut"], ctx.vessel(data["target"], data["start_ut"])) #type: ignore

class UndockEvent(TicketEvent):
    """Spawn a new vessel from ``part_ids`` at ``ut`` (seconds).

    Detaches the parts, creating a new vessel named ``vessel_name`` with their
    mass; ``on_spawn`` is an optional callback receiving the spawned vessel.
    """

    type = "undock"

    def __init__(self, ut: float, part_ids: list[str], vessel_name: str, on_spawn: Callable[[Spacecraft], None] | None = None):
        super().__init__(ut)
        self.part_ids, self.vessel_name, self.on_spawn = part_ids, vessel_name, on_spawn
        self.spawned_vessel: Spacecraft | None = None

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            operation = spacecraft.undock(self.part_ids, self.vessel_name, self.start_ut)
            if not operation.success or operation.spawned_vessel is None:
                raise RuntimeError(operation.reason)
            self.spawned_vessel = operation.spawned_vessel
            if self.on_spawn:
                self.on_spawn(self.spawned_vessel)
            self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "part_ids": self.part_ids,
            "vessel_name": self.vessel_name
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "UndockEvent":
        del ctx
        return cls(data["start_ut"], data["part_ids"], data["vessel_name"])


class ReferenceBodyEvent(TicketEvent):
    """Change the spacecraft's reference (parent) body at ``ut`` (seconds).

    Re-bases the vessel's relative position/velocity onto ``target`` without
    moving it in absolute space; used when crossing a sphere of influence.
    """

    type = "reference-body"

    def __init__(self, ut: float, target: Body):
        super().__init__(ut)
        self.target = target

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            operation = spacecraft.change_reference_body(self.target, self.start_ut)
            if not operation.success:
                raise RuntimeError(operation.reason)
            self.completed = True

    def to_json(self) -> dict:
        return {
                "type": self.type,
                "start_ut": self.start_ut,
                "end_ut": self.end_ut,
                "target": self.target.identifier
            }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "ReferenceBodyEvent":
        return cls(data["start_ut"], ctx.body(data["target"], data["start_ut"])) #type: ignore


class SurfaceOperationEvent(TicketEvent):
    """Land on or launch from a body at ``ut`` (seconds).

    ``operation`` is ``"land"`` (requires a ``body``) or ``"launch"``; records
    or clears the vessel's ``surface_body``. Pure bookkeeping, no state change.
    """

    type = "surface-operation"

    def __init__(
        self,
        ut: float,
        operation: str,
        body: Body | None = None,
    ):
        super().__init__(ut)

        if operation not in {"land", "launch"}:
            raise ValueError(
                "Surface operation must be 'land' or 'launch'."
            )

        if operation == "land" and body is None:
            raise ValueError(
                "A body is required for a landing operation."
            )

        self.operation = operation
        self.body = body

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if self.completed or ut < self.start_ut:
            return

        if self.operation == "land":
            outcome = spacecraft.land(self.body) #type: ignore
        else:
            outcome = spacecraft.launch()

        if not outcome.success:
            raise RuntimeError(outcome.reason)

        self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "operation": self.operation,
            "body": self.body.identifier if self.body else None,
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "SurfaceOperationEvent":
        body = ctx.body(data["body"], data["start_ut"]) if data.get("body") else None
        return cls(data["start_ut"], data["operation"], body)


class ScienceEvent(TicketEvent):
    """Collect or transmit science data at ``ut`` (seconds).

    ``operation`` is ``"collect"`` (gather ``amount`` units of data, defaulting
    to 100) or ``"transmit"`` (send ``amount`` units, required for transmit).
    """

    type = "science"

    def __init__(self, ut: float, operation: str, amount: float | None = None):
        super().__init__(ut)
        if operation not in {"collect", "transmit"}:
            raise ValueError("Science operation must be 'collect' or 'transmit'.")
        self.operation, self.amount = operation, amount

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if self.completed or ut < self.start_ut:
            return
        if self.operation == "collect":
            amount = self.amount if self.amount is not None else 100.0
            outcome = spacecraft.collect_science(amount)
        else:
            outcome = spacecraft.transmit_science(self.amount)
        if not outcome.success:
            raise RuntimeError(outcome.reason)
        self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "operation": self.operation,
            "amount": self.amount
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "ScienceEvent":
        del ctx
        return cls(data["start_ut"], data["operation"], data.get("amount"))


class CheckpointEvent(TicketEvent):
    """Record a vessel snapshot at ``ut`` (seconds) for later restore.

    Captures the full spacecraft state under ``name``; useful as a save point
    within a mission timeline.
    """

    type = "checkpoint"

    def __init__(self, ut: float, name: str):
        super().__init__(ut)
        self.name = name
        self.snapshot: VesselSnapshot | None = None

    def reset(self) -> None:
        super().reset()
        self.snapshot = None

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if not self.completed and ut >= self.start_ut:
            self.snapshot = spacecraft.snapshot(self.start_ut)
            self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "name": self.name
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "CheckpointEvent":
        del ctx
        return cls(data["start_ut"], data["name"])


@dataclass
class TicketSummary:
    identifier: str
    vessel: str
    event_count: int
    cursor_ut: float


class Ticket:
    """An ordered mission timeline for a single spacecraft.

    Holds the vessel, its originating :class:`~basic_systems.builder.System` and
    a sorted list of :class:`TicketEvent`\\ s. Use :meth:`add_event` to schedule,
    :meth:`advance_to` to play events up to a UT (seconds), and :meth:`reset` to
    replay. Persist with :meth:`save`/ :meth:`load` (``.kgrp.json``), which
    serialise the vessel and events and rebuild them via :class:`TicketContext`.
    """

    def __init__(self, identifier: str, spacecraft: Spacecraft, system: System, name: str = "", mesh_key: str | None = None):
        self.identifier = identifier
        self.name = name or identifier
        self.spacecraft = spacecraft
        self.system = system
        self.mission_system = system
        self.mesh_key = mesh_key
        self.events: list[TicketEvent] = []
        self.cursor_ut = spacecraft.t0
        self.initial_snapshot = spacecraft.snapshot(self.cursor_ut)

    def add_event(self, event: TicketEvent) -> None:
        if event.start_ut < self.cursor_ut:
            raise ValueError("Cannot schedule an event before the ticket cursor.")
        self.events.append(event)
        self.events.sort(key=lambda item: (item.start_ut, item.end_ut))

    def remove_event(self, index: int) -> TicketEvent:
        if self.events[index].start_ut < self.cursor_ut:
            raise ValueError("Cannot remove an event in the executed history.")
        return self.events.pop(index)

    def clear(self) -> None:
        if any(event.start_ut < self.cursor_ut for event in self.events):
            raise ValueError("Reset the ticket before clearing executed history.")
        self.events.clear()

    def advance_to(self, ut: float) -> None:
        if ut < self.cursor_ut:
            raise ValueError("Tickets only advance forward; use reset for a new run.")
        for event in self.events:
            if not event.completed and event.start_ut <= ut:
                event.update(self.spacecraft, ut)
        self.cursor_ut = float(ut)

    def reset(self) -> None:
        self.spacecraft.restore(self.initial_snapshot)
        self.cursor_ut = self.initial_snapshot.ut
        for event in self.events:
            event.reset()

    def to_json(self) -> dict:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "cursor_ut": self.cursor_ut,
            "vessel": _vessel_to_json(self.spacecraft, self.initial_snapshot, self.mesh_key),
            "events": [event.to_json() for event in self.events]
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_dir():
            path = path / f"{self.identifier}{TICKET_EXTENSION}"
        if not path.name.endswith(TICKET_EXTENSION):
            path = path.with_name(path.stem + TICKET_EXTENSION)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path

    @staticmethod
    def collect_bodies(root: Body) -> dict[str, Body]:
        return _collect_bodies(root)

    @classmethod
    def load(
        cls,
        path: str | Path,
        system: System,
        bodies: dict[str, Body],
        vessels: dict[str, Spacecraft] | None = None,
        mesh_factory: Callable[[str], Any] | None = None,
    ) -> "Ticket":
        path = Path(path)
        if not path.name.endswith(TICKET_EXTENSION):
            raise ValueError(f"Ticket files must use the '{TICKET_EXTENSION}' extension, got '{path.name}'.")
        data = json.loads(path.read_text())
        return cls.from_dict(data, system, bodies, vessels, mesh_factory)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        system: System,
        bodies: dict[str, Body],
        vessels: dict[str, Spacecraft] | None = None,
        mesh_factory: Callable[[str], Any] | None = None,
    ) -> "Ticket":
        spacecraft = _vessel_from_json(data["vessel"], bodies, mesh_factory)
        ticket = cls(data["identifier"], spacecraft, system, data.get("name", ""), data["vessel"].get("mesh_key"))
        ctx = TicketContext(bodies, vessels, data["cursor_ut"])
        for event_data in data["events"]:
            event_cls = TicketEvent.registry[event_data["type"]]
            ticket.events.append(event_cls.from_json(event_data, ctx))
        ticket.events.sort(key=lambda item: (item.start_ut, item.end_ut))
        ticket.advance_to(data["cursor_ut"])
        return ticket