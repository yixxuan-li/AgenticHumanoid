"""Typed predicate vocabulary and explicit bindings."""
import re

ARITY = {"visible": 1, "hand_free": 1, "object_in_region": 2, "drawer_open": 1, "holding": 2}


def parse(atom):
    if not isinstance(atom, str):
        raise ValueError("Predicate must be a string")
    match = re.fullmatch(r"([a-z_]+)\(([a-zA-Z0-9_,?]+)\)", atom)
    if not match:
        raise ValueError("Invalid predicate syntax")
    name, args = match[1], tuple(match[2].split(","))
    if name not in ARITY or len(args) != ARITY[name]:
        raise ValueError("Unknown predicate or wrong arity")
    return name, args


def bind(atom, bindings):
    name, args = parse(atom)
    bound = []
    for arg in args:
        if arg.startswith("?") and arg not in bindings:
            raise ValueError(f"Unbound variable {arg}")
        bound.append(bindings.get(arg, arg))
    result = f"{name}({','.join(bound)})"
    parse(result)
    return result


def geometric_reachable(root_position, target_position, radius=.65):
    """Conservative geometric screen, never a learned symbolic predicate."""
    import numpy as np
    return bool(np.linalg.norm(np.asarray(root_position)-np.asarray(target_position)) <= radius)
