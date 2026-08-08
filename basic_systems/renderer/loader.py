import json
from pathlib import Path

from basic_systems.builder import System
from basic_systems.orbit_pred import Body, Orbit


def load_system(json_name: str = "planets.json") -> System:
    """Load a planetary system from a JSON configuration file."""

    target_file = Path(json_name)


    with target_file.open("r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from '{target_file}': {e}"
            ) from e

    title = payload.get("system_name", "Unknown System")
    bodies_data: dict = payload.get("bodies", {})

    if not bodies_data:
        raise ValueError("No bodies found in configuration.")

    root_keys = [
        name
        for name, attrs in bodies_data.items()
        if attrs.get("is_root", False)
    ]

    if len(root_keys) != 1:
        raise ValueError(
            "Exactly one body must have 'is_root': true."
        )

    root_key = root_keys[0]
    active_system: System | None = None

    def parse_body(
        name: str,
        attrs: dict,
        parent: Body | None = None,
    ) -> Body:
        nonlocal active_system

        try:
            mu = attrs["mu"]
            radius = attrs["radius"]
        except KeyError as e:
            raise ValueError(
                f"{name}: missing required key '{e.args[0]}'"
            ) from None

        orbit = None
        if parent is not None:
            orbit = Orbit(
                a=attrs["a"],
                e=attrs["e"],
                arg_p=attrs["arg_p"],
                lon_of_asc=attrs["lon_of_asc"],
                MA_at_t0=attrs["MA_at_t0"],
                inclination=attrs["inc"],
                parent=parent,
            )

        body = Body(
            name=name,
            mu=mu,
            radius=radius,
            atm_height=attrs.get("atm_height", 0.0),
            orbit=orbit,
            render_color=attrs.get("color", "#808080"),
            identifier=str(attrs.get("identifier")),
        )

        if parent is None:
            active_system = System(
                name=title,
                root_obj=body,
            )
        else:
            active_system.add_child(parent, body)  # type: ignore

        for moon_name, moon_attrs in attrs.get("moons", {}).items():
            parse_body(moon_name, moon_attrs, body)

        return body

    root = parse_body(root_key, bodies_data[root_key])

    for name, attrs in bodies_data.items():
        if name == root_key:
            continue

        parse_body(name, attrs, root)

    if active_system is None:
        raise RuntimeError("Failed to create system.")

    return active_system