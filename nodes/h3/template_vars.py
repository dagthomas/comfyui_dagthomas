# APNext H3 template variables
#
# Lets the free-text boxes of an H3 node (idea, direction, extra_instructions,
# wardrobe, ...) refer to what is wired into the node by name, so a brief can
# say "{character1} walks into {character2}'s kitchen" and the model receives
# the real names. Variables come from the node's connected sockets:
#
#   {character1} {actor1} {franchise1} {cast1} {wardrobe1} - the k-th H3 Characters node
#       feeding this node (cast_N / context_N sockets, chained cast_in included),
#       numbered in socket order
#   {characters}                                  - "A, B and C"
#   {cast}                                        - every cast line, one per line
#   {context1} / {context_1}, {cast_1}, ...       - the raw text on that socket
#
# Unknown {names} are left untouched, so JSON-ish braces in a brief survive.
# The frontend (web/js/h3_template_vars.js) shows the same list on the node.

import re

from ...utils.apnext_context import HIDDEN_GRAPH, HIDDEN_NODE_ID

VAR_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CAST_LINE = re.compile(r"^(?P<name>.+?)(?: \(played by (?P<actor>.+?)\))? from (?P<show>.+)$")
_CHARACTERS_CLASS = "H3Characters"


def _node(graph, node_id):
    if not isinstance(graph, dict) or node_id is None:
        return {}
    return graph.get(str(node_id)) or graph.get(node_id) or {}


def _characters_chain(graph, node_id, seen):
    """
    Every H3 Characters node that ends up in `node_id`'s cast line, oldest
    first (cast_in chains append, so the upstream node's character comes first).
    """
    node = _node(graph, node_id)
    if node.get("class_type") != _CHARACTERS_CLASS or str(node_id) in seen:
        return []
    seen.add(str(node_id))
    chain = []
    upstream = (node.get("inputs") or {}).get("cast_in")
    if isinstance(upstream, list) and upstream:
        chain.extend(_characters_chain(graph, upstream[0], seen))
    chain.append(str(node_id))
    return chain


def _pick_character(graph, node_id):
    """(character, actor, franchise, cast_line, wardrobe) for one H3 Characters node, or None."""
    node = _node(graph, node_id)
    inputs = node.get("inputs") or {}
    label = inputs.get("character")
    if not isinstance(label, str):
        return None  # the picker itself is driven by a link; nothing to resolve
    try:
        from .characters import H3Characters
        seed = inputs.get("seed", 0)
        seed = int(seed) if isinstance(seed, (int, float, str)) and str(seed).lstrip("-").isdigit() else 0
        franchise_filter = inputs.get("franchise_filter")
        if not isinstance(franchise_filter, str):
            franchise_filter = "(all)"
        custom = inputs.get("custom_character")
        wardrobe = inputs.get("wardrobe")
        custom = custom if isinstance(custom, str) else ""
        wardrobe = wardrobe if isinstance(wardrobe, str) else ""
        character, actor, franchise, _path, cast, wardrobe = H3Characters().pick(
            label, franchise_filter, seed, custom, wardrobe, ""
        )
        cast_line = cast.strip().splitlines()[-1] if cast.strip() else ""
        from .characters import split_cast_line
        cast_line, _ = split_cast_line(cast_line)
        return character, actor, franchise.split(",")[0].strip(), cast_line, wardrobe.strip()
    except Exception:
        return None


def _parse_cast_line(line):
    from .characters import split_cast_line, cast_line_name
    head, wardrobe = split_cast_line(line.strip())
    m = _CAST_LINE.match(head)
    if not m:
        return cast_line_name(head), "", "", head, wardrobe
    return m.group("name").strip(), (m.group("actor") or "").strip(), m.group("show").strip(), head, wardrobe


def collect_template_vars(slots, extra_text_sockets=()):
    """
    Build the {variable} map for one node run from its **slots (the kwargs that
    carry the cast_N / context_N sockets and the hidden graph inputs). Call it
    BEFORE build_context(), which pops the graph out of the dict.

    Returns (vars, summary_lines).
    """
    graph = slots.get(HIDDEN_GRAPH)
    node_id = slots.get(HIDDEN_NODE_ID)
    me = _node(graph, node_id)
    my_inputs = me.get("inputs") or {}

    variables = {}
    summary = []

    # Socket order: cast_* first, then context_*, then anything else that is linked.
    def _rank(name):
        if name.startswith("cast_"):
            return (0, name)
        if name.startswith("context_"):
            return (1, name)
        return (2, name)

    linked = [
        (name, link) for name, link in my_inputs.items()
        if isinstance(link, list) and len(link) >= 2
    ]
    linked.sort(key=lambda item: _rank(item[0]))

    # 1. H3 Characters nodes, numbered in socket order (chains oldest-first).
    people = []
    seen = set()
    for name, link in linked:
        src_id = link[0]
        for char_id in _characters_chain(graph, src_id, seen):
            picked = _pick_character(graph, char_id)
            if picked is None:
                # fall back to the text on the socket, if it is a cast line
                value = slots.get(name)
                if isinstance(value, str) and value.strip():
                    picked = _parse_cast_line(value.strip().splitlines()[-1])
            if picked:
                people.append(picked)

    # Cast text typed/connected that came from somewhere other than H3 Characters
    # still yields characters when it is in the "Name (played by X) from Show" form.
    for name, link in linked:
        if name.startswith("cast_"):
            src = _node(graph, link[0])
            if src.get("class_type") == _CHARACTERS_CLASS:
                continue
            value = slots.get(name)
            if isinstance(value, str):
                for line in value.strip().splitlines():
                    if line.strip() and " from " in line:
                        people.append(_parse_cast_line(line))

    for k, (character, actor, franchise, cast_line, wardrobe) in enumerate(people, 1):
        variables[f"character{k}"] = character
        variables[f"actor{k}"] = actor
        variables[f"franchise{k}"] = franchise
        variables[f"show{k}"] = franchise
        variables[f"cast{k}"] = cast_line
        variables[f"wardrobe{k}"] = wardrobe
        summary.append(f"{{character{k}}}={character}" + (f" ({actor})" if actor else ""))
    if people:
        names = [p[0] for p in people]
        variables["characters"] = (
            names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        )
        variables["cast"] = "\n".join(p[3] for p in people)

    # 2. Raw socket text: {context_1} and {context1}, {cast_1} and {cast_1}...
    for name, value in slots.items():
        if name in (HIDDEN_GRAPH, HIDDEN_NODE_ID) or not isinstance(value, str) or not value.strip():
            continue
        if not (name.startswith("context_") or name.startswith("cast_") or name in extra_text_sockets):
            continue
        variables.setdefault(name, value.strip())
        variables.setdefault(name.replace("_", ""), value.strip())
        summary.append(f"{{{name}}}")

    return variables, summary


def expand_template(text, variables):
    """Replace {name} with its value for every known name; unknown names stay."""
    if not text or not variables or "{" not in text:
        return text

    def _sub(match):
        key = match.group(1)
        return variables.get(key, match.group(0)) if key in variables else match.group(0)

    return VAR_RE.sub(_sub, text)


def expand_all(variables, *texts):
    """expand_template over several strings at once; None passes through."""
    return tuple(expand_template(t, variables) if isinstance(t, str) else t for t in texts)


def used_variables(variables, *texts):
    """Which known {names} actually appear in the given texts."""
    used = []
    for text in texts:
        if not isinstance(text, str):
            continue
        for key in VAR_RE.findall(text):
            if key in variables and key not in used:
                used.append(key)
    return used


def log_template_vars(variables, summary, *texts):
    """One console line so the user can see what was available and what was used."""
    if not variables:
        return
    used = used_variables(variables, *texts)
    print(
        "🔤 H3 template vars | available: " + ", ".join(summary[:12])
        + (" …" if len(summary) > 12 else "")
        + " | used: " + (", ".join("{" + u + "}" for u in used) or "none")
    )
