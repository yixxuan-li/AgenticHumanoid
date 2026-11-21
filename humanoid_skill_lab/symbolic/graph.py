"""Inspectable dependencies; edges do not replace conjunctive planning."""


def build_graph(rules):
    nodes, edges = {}, []
    for rule in rules:
        skill = f"skill:{rule['skill']}@{rule['skill_version']}"
        nodes[skill] = {"type": "skill", "status": rule["status"]}
        for kind, terms in (("requires", rule["preconditions"]), ("produces", rule["add_effects"]), ("invalidates", rule["delete_effects"])):
            for term in terms:
                nodes[term] = {"type": "predicate"}
                edges.append({"source": skill, "target": term, "type": kind})
    for left in rules:
        for right in rules:
            if set(left["add_effects"]) & set(right["preconditions"]):
                edges.append({"source": f"skill:{left['skill']}@{left['skill_version']}",
                              "target": f"skill:{right['skill']}@{right['skill_version']}", "type": "may_enable"})
    return {"nodes": nodes, "edges": edges, "warning": "Dependencies require grounded AND-condition checks"}
