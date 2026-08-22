from __future__ import annotations

import sys, pathlib
# Add the project root to sys.path
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from basic_systems.orbit_pred import Body, Orbit
from typing import Any
from pathlib import Path
import json

class System:
    """A celestial system: a tree of :class:`~basic_systems.orbit_pred.Body` objects.

    The ``root`` body anchors the hierarchy and ``get``/``__getitem__`` look up
    children by name or identifier. Build one from JSON with :meth:`load` (a
    ``planets.json`` mapping bodies, orbits, radii and mu values) and persist it
    with :meth:`save`. ``add_child``/``remove_child`` mutate the tree, while
    ``tree`` renders an ASCII view for inspection.
    """

    def __init__(self, name: str, root_obj: Body):
        self.root = root_obj
        self.name = name
        
    def add_child(self, parent: Body, obj_to_add: Body):
        parent.moons.append(obj_to_add)
        if hasattr(obj_to_add, 'orbit') and obj_to_add.orbit:
            obj_to_add.orbit.parent = parent
    
    def remove_child(self, obj_to_remove: Body):
        if obj_to_remove == self.root:
            return
            
        parent = getattr(getattr(obj_to_remove, 'orbit', None), 'parent', None)
        
        if parent and obj_to_remove in parent.moons:
            parent.moons.remove(obj_to_remove)
            obj_to_remove.orbit.parent = None

    def get_all_obj_in_system(self) -> dict[Body, list[Body]]:
        system_dict = {}
        
        def traverse(current: Body):
            if not current: 
                return
            system_dict[current] = list(current.moons) 
            for moon in current.moons:
                traverse(moon)
                
        traverse(self.root)
        return system_dict

    def tree(self, start_body: Body | None = None) -> str:
        if self.root is None:
            return "System(empty)"

        start = start_body if start_body else self.root

        def build_tree(body: Body, prefix: str = "", is_last: bool = True) -> list[str]:
            connector = "└── " if is_last else "├── "

            label = body.name
            if body.identifier is not None:
                label += f" ({body.identifier})"

            lines = [prefix + connector + label]

            child_prefix = prefix + ("    " if is_last else "│   ")

            for i, moon in enumerate(body.moons):
                lines.extend(
                    build_tree(
                        moon,
                        child_prefix,
                        i == len(body.moons) - 1,
                    )
                )

            return lines

        tree = build_tree(start)
        tree[0] = tree[0][4:]  # Remove connector from root.

        return "\n".join(tree)

    def __repr__(self) -> str:
        return f"System: {self.name} \n{self.tree()}"
    
    def get(self, name: str, default: None | Any = None) -> Body | Any:
        target_str = name.strip().lower()
        for body in self.get_all_obj_in_system():
            if getattr(body, "name", "").strip().lower() == target_str or getattr(body, "identifier", "").strip().lower() == target_str:
                return body
        return default

    def __getitem__(self, key):
        body = self.get(key)
        if not body:
            raise KeyError(f"Body {key} not found in the {self.name} system")
        return body

    @classmethod
    def load(cls, json_name: str | Path = "planets.json") -> System:
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
                rotation_period_s=attrs.get("rotation_period_s", 0.0),
            )

            if parent is None:
                active_system = System(
                    name=title,
                    root_obj=body,
                )
            else:
                active_system.add_child(parent, body) #type: ignore

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
    
    def save(self, json_name: str | Path = "planets.json"):
        path_str = str(json_name)
        is_directory_path = path_str.endswith(("/", "\\"))
        target_file = Path(json_name)

        if target_file.is_dir() or is_directory_path:
            target_file = target_file / "planets.json"

        if any(part.endswith(".json") for part in target_file.parent.parts):
            raise ValueError(
                f"Invalid path structure: '{target_file.parent}' contains a '.json' directory name."
            )

        target_file.parent.mkdir(parents=True, exist_ok=True)

        def serialize_body(body: Body):
            body_data = {
                "mu": body.mu,
                "radius": body.radius,
                "rotation_period_s": getattr(body, "rotation_period_s", 0.0),
                "atm_height": getattr(body, "atm_height", 0.0),
                "color": getattr(body, "render_color", "#808080"),
                "identifier": getattr(body, "identifier", None)
            }

            if body == self.root:
                body_data["is_root"] = True

            if getattr(body, "orbit", None) is not None:
                body_data.update({
                    "a": body.orbit.semi_major_axis,
                    "e": body.orbit.eccen,
                    "arg_p": body.orbit.arg_p,
                    "lon_of_asc": body.orbit.lon_of_asc,
                    "MA_at_t0": body.orbit.MA_at_t0,
                    "inc": body.orbit.inclination,
                })

            if body.moons:
                body_data["moons"] = {
                    moon.name: serialize_body(moon)
                    for moon in body.moons
                }

            return body_data
        
        payload = {
            "system_name": self.name,
            "bodies": {
                self.root.name: serialize_body(self.root)
            }
        }

        with target_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)



