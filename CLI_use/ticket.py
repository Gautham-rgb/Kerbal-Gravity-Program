"""Mission timeline events.

The REPL and renderer should advance a :class:`Ticket`; they should never
directly mutate a spacecraft to perform a scheduled mission action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import json
from pathlib import Path

from basic_systems.RKF45 import RKF45
from basic_systems.builder import System
from basic_systems.orbit_pred import Body, ManeuverNode, Spacecraft, VesselSnapshot


class Propagator(Protocol):
    def propagate(self, spacecraft: Spacecraft, start_ut: float, end_ut: float) -> tuple[np.ndarray, np.ndarray]: ...


class KeplerPropagator:
    """Two-body propagation through the spacecraft's current orbital elements."""

    def propagate(self, spacecraft: Spacecraft, start_ut: float, end_ut: float) -> tuple[np.ndarray, np.ndarray]:
        del start_ut
        return spacecraft.state_at(end_ut)


@dataclass
class TicketContext:
    bodies: dict[str, Body] | None = None
    vessels: dict[str, Spacecraft] | None = None

    def body(self, identifier: str | None) -> Body | None:
        if identifier is None:
            return None
        if not self.bodies or identifier not in self.bodies:
            raise KeyError(f"No body lookup entry for '{identifier}'.")
        return self.bodies[identifier]

    def vessel(self, identifier: str | None) -> Spacecraft | None:
        if identifier is None:
            return None
        if not self.vessels or identifier not in self.vessels:
            raise KeyError(f"No vessel lookup entry for '{identifier}'.")
        return self.vessels[identifier]


class TicketEvent(ABC):
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

    def reset(self) -> None:
        self.completed = False
        self.result = ""

    @abstractmethod
    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        """Apply this event through ``ut``. ``ut`` can be beyond event end."""

    @abstractmethod
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
    type = "coast"

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        del spacecraft
        self.completed = ut >= self.end_ut



class ManeuverEvent(TicketEvent):
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
    type = "rkf45"

    def __init__(self, start_ut: float, end_ut: float, tolerance: float = 1e-6):
        super().__init__(start_ut, end_ut)
        self.tolerance = tolerance
        self.last_ut = start_ut
        self.integrator: RKF45 | None = None

    def reset(self) -> None:
        super().reset()
        self.last_ut = self.start_ut
        self.integrator = None

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        target = min(ut, self.end_ut)
        if target <= self.last_ut:
            return
        if self.integrator is None:
            system = getattr(spacecraft, "mission_system", None)
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
            "thrust": self.pressure
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "FiniteBurnEvent":
        del ctx
        return cls(data["start_ut"], data["end_ut"], np.array(data["direction"]), data.get("thrust", 0.0))


class EngineControlEvent(TicketEvent):
    type = "engine-control"

    def __init__(self, ut: float, active: bool | None = None, throttle: float | None = None, engine_indices: list[int] | None = None):
        if active is None and throttle is None:
            raise ValueError("Engine control requires an active state or throttle.")
        super().__init__(ut)
        self.active_now, self.throttle, self.engine_indices = active, throttle, engine_indices

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if self.completed or ut < self.start_ut:
            return
        indices = self.engine_indices or list(range(len(spacecraft.engines)))
        if self.active_now is not None:
            for index in indices:
                spacecraft.set_engine_state(index, bool(self.active_now))
        if self.throttle is not None:
            spacecraft.set_throttle(self.throttle, indices)
        self.completed = True

    def to_json(self) -> dict:
        return {
            "type": self.type,
            "start_ut": self.start_ut,
            "end_ut": self.end_ut,
            "active": self.active_now,
            "throttle": self.throttle,
            "indices": self.engine_indices
        }

    @classmethod
    def from_json(cls, data: dict, ctx: TicketContext) -> "EngineControlEvent":
        del ctx
        return cls(data["start_ut"], data.get("active"), data.get("throttle"), data.get("indices"))


class AttitudeEvent(TicketEvent):
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
        return cls(data["start_ut"], ctx.vessel(data["target"]), data["resource"], data["amount"]) #type: ignore


class DockEvent(TicketEvent):
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
        return cls(data["start_ut"], ctx.vessel(data["target"])) #type: ignore

class UndockEvent(TicketEvent):
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
        return cls(data["start_ut"], ctx.body(data["target"])) #type: ignore


class SurfaceOperationEvent(TicketEvent):
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
        body = ctx.body(data["body"]) if data.get("body") else None
        return cls(data["start_ut"], data["operation"], body)


class ScienceEvent(TicketEvent):
    type = "science"

    def __init__(self, ut: float, operation: str, amount: float | None = None):
        super().__init__(ut)
        if operation not in {"collect", "transmit"}:
            raise ValueError("Science operation must be 'collect' or 'transmit'.")
        self.operation, self.amount = operation, amount

    def update(self, spacecraft: Spacecraft, ut: float) -> None:
        if self.completed or ut < self.start_ut:
            return
        outcome = spacecraft.collect_science(self.amount or 0.0) if self.operation == "collect" else spacecraft.transmit_science(self.amount)
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
    def __init__(self, identifier: str, spacecraft: Spacecraft, system: System, name: str = ""):
        self.identifier = identifier
        self.name = name or identifier
        self.spacecraft = spacecraft
        self.system = system
        self.mission_system = system
        self.events: list[TicketEvent] = []
        self.cursor_ut = spacecraft.t0
        self.initial_snapshot = spacecraft.snapshot(self.cursor_ut)

    def add_event(self, event: TicketEvent) -> None:
        if event.start_ut < self.cursor_ut:
            raise ValueError("Cannot schedule an event before the ticket cursor.")
        self.events.append(event)
        self.events.sort(key=lambda item: (item.start_ut, item.end_ut))

    def remove_event(self, index: int) -> TicketEvent:
        if self.events[index].start_ut <= self.cursor_ut:
            raise ValueError("Cannot remove an event in the executed history.")
        return self.events.pop(index)

    def clear(self) -> None:
        if any(event.start_ut <= self.cursor_ut for event in self.events):
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
            "events": [event.to_json() for event in self.events]
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(
        cls,
        path: str | Path,
        spacecraft: Spacecraft,
        system: System,
        bodies: dict[str, Body] | None = None,
        vessels: dict[str, Spacecraft] | None = None,
    ) -> Ticket:
        
        data = json.loads(Path(path).read_text())
        ticket = cls(data["identifier"], spacecraft, system, data.get("name", ""))
        ctx = TicketContext(bodies, vessels)
        for event_data in data["events"]:
            event_cls = TicketEvent.registry[event_data["type"]]
            ticket.events.append(event_cls.from_json(event_data, ctx))
        ticket.events.sort(key=lambda item: (item.start_ut, item.end_ut))
        ticket.advance_to(data["cursor_ut"])
        return ticket