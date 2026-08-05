import json
from pathlib import Path

from basic_systems.builder import System
from basic_systems.orbit_pred import Body, Orbit


def load_system(json_name: str = "planets.json") -> System:
    """Load a planetary system from a JSON configuration file.
    
    The JSON should have a 'system_name' and a 'bodies' dictionary.
    One body in 'bodies' must have a semi-major axis 'a' of 0.0 to be the root.
    """
    target_file = Path(json_name)
    
    # If not found directly, try relative to the script's parent and grandparent
    if not target_file.exists():
        current_dir = Path(__file__).resolve().parent
        # Try basic_systems/planets.json
        target_file = current_dir / json_name
        if not target_file.exists():
            # Try KerbalGravityProg/planets.json
            target_file = current_dir.parent / json_name
            if not target_file.exists():
                # Try relative to CWD
                target_file = Path.cwd() / json_name
                if not target_file.exists():
                    raise FileNotFoundError(f"Configuration file '{json_name}' not found at any expected location.")

    with open(target_file, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {target_file}: {e}")

    title = payload.get("system_name", "Unknown System")
    bodies_data = payload.get("bodies", {})
    if not bodies_data:
        raise ValueError("No 'bodies' found in configuration.")

    active_system: System | None = None

    def parse_body(name: str, attrs: dict, parent_node: Body | None = None) -> Body:
        nonlocal active_system
        
        # Validation
        mu = attrs.get("mu", 0.0)
        radius = attrs.get("radius", 1.0)
        
        curr_orbit = Orbit(
            a=attrs.get("a", 0.0), 
            e=attrs.get("e", 0.0), 
            arg_p=attrs.get("arg_p", 0.0),
            lon_of_asc=attrs.get("lon_of_asc", 0.0), 
            MA_at_t0=attrs.get("MA_at_t0", 0.0),
            inclination=attrs.get("inclination", 0.0), 
            parent=parent_node
        )
        
        curr_node = Body(
            name=name, 
            mu=mu, 
            radius=radius,
            atm_height=attrs.get("atm_height", 0.0), 
            orbit=curr_orbit,
            render_color=attrs.get("color", "#808080"),
            identifier=attrs.get("identifier", "X")
        )
        
        if parent_node is None:
            if active_system is not None:
                raise ValueError(f"Multiple root bodies detected. '{name}' and '{active_system.root.name}' both have a=0.")
            active_system = System(name=title, root_obj=curr_node)
        else:
            if active_system is not None:
                active_system.add_child(parent_node, curr_node)

        for moon_name, moon_attrs in attrs.get("moons", {}).items():
            parse_body(moon_name, moon_attrs, curr_node)
            
        return curr_node

    # Find the root body (a=0)
    root_keys = [k for k, v in bodies_data.items() if v.get("a", 0.0) == 0.0 or v.get("true_root", False)]
    if not root_keys:
        raise ValueError("No root body (with a=0.0) found in configuration.")
    
    root_key = root_keys[0]
    root_attrs = bodies_data.pop(root_key)
    
    # Any other top-level bodies are treated as direct children of the root
    # unless they are already in the root's moons list.
    if "moons" not in root_attrs:
        root_attrs["moons"] = {}
    
    root_attrs["moons"].update(bodies_data)
    
    parse_body(root_key, root_attrs, None)
    
    if active_system is None:
        raise ValueError("Failed to initialize system hierarchy.")
        
    return active_system