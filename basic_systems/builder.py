import sys, pathlib
# Add the project root to sys.path
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from basic_systems.orbit_pred import Body
from typing import Any

class System:
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

    def __repr__(self) -> str:
        if self.root is None:
            return "System(empty)"

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

        tree = build_tree(self.root)
        tree[0] = tree[0][4:]  # Remove connector from root.

        return (
            f"System: {self.name}\n"
            + "\n".join(tree)
        )

    def get(self, name: str, default: None | Any = None) -> Body | Any:
        target_name = name.strip().lower()
        for body in self.get_all_obj_in_system():
            if getattr(body, "name", "").strip().lower() == target_name:
                return body
        return default

    def __getitem__(self, key):
        body = self.get(key)
        if not body:
            raise KeyError(f"Body {key} not found in the {self.name} system")
        return body

