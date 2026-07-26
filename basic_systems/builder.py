import sys, pathlib
sys.path.extend([str(pathlib.Path(__file__).resolve().parents[1]), str(pathlib.Path(__file__).resolve().parents[1] / "basic_systems")])

from orbit_pred import Body
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
        if not self.root:
            return "System(Empty)"
            
        def build_tree_str(current: Body, level: int = 0) -> str:
            indent = "  " * level
            prefix = f"{indent}└── " if level > 0 else ""
            
            body_name = getattr(current, 'name', str(current))
            lines = [f"{prefix}{body_name}"]
            
            for moon in current.moons:
                lines.append(build_tree_str(moon, level + 1))
                
            return "\n".join(lines)
            
        return f"System Structure:\nName: The {self.name}\n{build_tree_str(self.root)}"

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

