from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from basic_systems.builder import System
from basic_systems.orbit_pred import Body

@dataclass
class ValidationResult:
    warnings: list[str]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_system(system: System) -> ValidationResult:
    result = ValidationResult([], [])
    seen_ids: set[str] = set()

    def visit(body: Body):
        if body.identifier is not None:
            if body.identifier in seen_ids:
                result.errors.append(
                    f"Duplicate body ID '{body.identifier}' "
                    f"(body '{body.name}')."
                )
            else:
                seen_ids.add(body.identifier)

        if not np.isfinite(body.mu):
            result.errors.append(
                f"Body '{body.name}' has a non-finite μ."
            )
        elif body.mu < 0:
            result.errors.append(
                f"Body '{body.name}' has a negative μ ({body.mu})."
            )

        if not np.isfinite(body.radius):
            result.errors.append(
                f"Body '{body.name}' has a non-finite radius."
            )
        elif body.radius < 0:
            result.errors.append(
                f"Body '{body.name}' has a negative radius ({body.radius})."
            )

        if not np.isfinite(body.atm_height):
            result.errors.append(
                f"Body '{body.name}' has a non-finite atmosphere height."
            )
        elif body.atm_height < 0:
            result.errors.append(
                f"Body '{body.name}' has a negative atmosphere height."
            )

        rotation_period = getattr(body, "rotation_period_s", 0.0)

        if not np.isfinite(rotation_period):
            result.errors.append(
                f"Body '{body.name}' has a non-finite rotation period."
            )
        elif rotation_period < 0:
            result.errors.append(
                f"Body '{body.name}' has a negative rotation period "
                f"({rotation_period})."
            )
        elif rotation_period == 0:
            result.warnings.append(
                f"Body '{body.name}' has no rotation period; "
                f"synchronous-orbit planning is unavailable for it."
            )

        if body.orbit is not None:
            orbit = body.orbit

            if orbit.parent is not None:
                if orbit.parent is body:
                    result.errors.append(
                        f"Body '{body.name}' cannot orbit itself."
                    )

                if not np.isfinite(orbit.semi_major_axis):
                    result.errors.append(
                        f"Body '{body.name}' has a non-finite semi-major axis."
                    )
                elif orbit.semi_major_axis <= 0:
                    result.errors.append(
                        f"Body '{body.name}' has an invalid semi-major axis "
                        f"({orbit.semi_major_axis})."
                    )

                if not np.isfinite(orbit.eccen):
                    result.errors.append(
                        f"Body '{body.name}' has a non-finite eccentricity."
                    )
                elif not (0 <= orbit.eccen < 1):
                    result.errors.append(
                        f"Body '{body.name}' has an invalid eccentricity "
                        f"({orbit.eccen})."
                    )

                periapsis = orbit.semi_major_axis * (1 - orbit.eccen)

                if periapsis <= orbit.parent.radius:
                    result.errors.append(
                        f"Body '{body.name}' intersects its parent at periapsis."
                    )

            if not np.isfinite(orbit.inclination):
                result.errors.append(
                    f"Body '{body.name}' has a non-finite inclination."
                )

            for field in ("arg_p", "lon_of_asc", "MA_at_t0"):
                value = getattr(orbit, field)

                if not np.isfinite(value):
                    result.errors.append(
                        f"Body '{body.name}' has a non-finite {field}."
                    )

        for moon in body.moons:
            visit(moon)

    visit(system.root)
    return result
