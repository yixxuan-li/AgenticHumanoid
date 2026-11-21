"""Finite grounded BFS: all preconditions, bindings, versions and effects matter."""
from collections import deque
from dataclasses import dataclass
from itertools import product
from .predicates import bind, parse
from .knowledge import check_rule
from humanoid_skill_lab.types import Truth


@dataclass(frozen=True)
class Operator:
    skill: str
    version: int
    bindings: tuple
    preconditions: frozenset
    add: frozenset
    delete: frozenset


def ground(rule, domains, accepted_versions):
    check_rule(rule)
    if rule["status"] != "validated" or accepted_versions.get(rule["skill"]) != rule["skill_version"]:
        return []
    variables = sorted({a for key in ("preconditions", "add_effects", "delete_effects") for term in rule[key]
                        for a in parse(term)[1] if a.startswith("?")})
    if any(v not in domains or not domains[v] for v in variables):
        raise ValueError("Every variable requires an explicit finite domain")
    result = []
    for values in product(*(domains[v] for v in variables)):
        binding = dict(zip(variables, values))
        terms = [frozenset(bind(a, binding) for a in rule[key]) for key in ("preconditions", "add_effects", "delete_effects")]
        if terms[1] & terms[2]:
            continue
        result.append(Operator(rule["skill"], rule["skill_version"], tuple(binding.items()), *terms))
    return result


def plan(state, goals, operators, max_depth=8, max_nodes=10000):
    true = frozenset(k for k,v in state.items() if v == Truth.TRUE)
    goals = frozenset(goals)
    if not consistent(true):
        raise ValueError("Initial symbolic state contains conflicting effects")
    frontier = deque([(true, [])])
    seen = {true}
    while frontier:
        current, path = frontier.popleft()
        if goals <= current:
            return path
        if len(path) >= max_depth:
            continue
        for op in operators:
            if not op.preconditions <= current:
                continue
            successor = (current-op.delete) | op.add
            if not consistent(successor):
                continue
            if successor in seen:
                continue
            seen.add(successor)
            if len(seen) > max_nodes:
                raise ValueError("Symbolic search budget exhausted")
            frontier.append((successor, path+[op]))
    return None


def consistent(atoms):
    held = {}
    free = set()
    for atom in atoms:
        name, args = parse(atom)
        if name == "hand_free":
            free.add(args[0])
        elif name == "holding":
            if args[0] in held and held[args[0]] != args[1]:
                return False
            held[args[0]] = args[1]
    return not (free & held.keys())


def apply_effects(state, operator):
    result = dict(state)
    for atom in operator.delete:
        result[atom] = Truth.FALSE
    for atom in operator.add:
        result[atom] = Truth.TRUE
    if not consistent({a for a,v in result.items() if v == Truth.TRUE}):
        raise ValueError("Effects violate hand occupancy constraints")
    return result
