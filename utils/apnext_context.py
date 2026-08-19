# APNext context sockets for the Claude Code nodes.
#
# The APNext generator nodes (APNext Time, Scene, Poses, Plots, Feelings,
# Cinematic, Science, Geography, Architecture, ...) all emit a plain STRING.
# Wired straight into an `idea` field they are just words; the writer cannot
# tell a time period from a pose. These helpers give every Claude Code node a
# row of `context_N` sockets and, using ComfyUI's hidden PROMPT / UNIQUE_ID
# inputs, look up *which* node feeds each socket. The category then selects a
# short instruction on how that kind of input should steer the scene, and the
# whole thing is handed to Claude as a labelled context block.
#
# Everything degrades gracefully: with no graph access the text is still used,
# labelled as generic context.

import re

MAX_CONTEXT = 8
CONTEXT_SOCKETS = tuple(f"context_{i}" for i in range(1, MAX_CONTEXT + 1))
HIDDEN_GRAPH = "apnext_graph"
HIDDEN_NODE_ID = "apnext_node_id"

# Widgets on the APNext generator nodes that are not category fields.
_NON_FIELD_INPUTS = {"prompt", "separator", "string", "seed", "attributes"}
_INACTIVE_VALUES = {"None", "", None}

# How each APNext category should steer a video prompt. Keys are the data/next
# folder names (== the node's `_subcategory`); a few extra keys cover our own
# nodes and the fallback.
CATEGORY_GUIDANCE = {
    "time": (
        "Period, era, decade or time of day. Make it visible: period-correct wardrobe, "
        "props, vehicles, technology, signage and architecture; the light and colour of "
        "that hour. State the time of day in every shot's opening line and keep it "
        "consistent across shots."
    ),
    "scene": (
        "Setting and environment: location type, weather, textures, plants, modifiers. "
        "Use it as the place the action happens - describe it concretely in [Shot 1] "
        "and keep its light, weather and materials consistent."
    ),
    "poses": (
        "Physical staging of the subject(s): body position, gesture, portrait pose. Turn "
        "it into what the character is physically doing at the start of the shot and how "
        "they move out of it; never leave a pose as a static hold."
    ),
    "plots": (
        "Story beats and genre. Dramatise them as on-screen events - cause and effect, "
        "an arc across the shots (and across scenes when several are written). Show, "
        "don't narrate."
    ),
    "feelings": (
        "Emotional tone. Express it through performance, delivery, body language, "
        "lighting temperature, colour, pacing and sound - not by naming the emotion."
    ),
    "cinematic": (
        "Camera and film language: shot type, genre, effects, colour grading, film stock, "
        "director/movie references. Translate into the stated visual style, framing, "
        "camera moves and grade; a director or movie reference means their observable "
        "craft (lens, palette, blocking, rhythm), never a name-drop."
    ),
    "photography": (
        "Photographic look: camera, lens, film, lighting setup, grade. Fold into the "
        "visual style and lighting description as observable qualities (depth of field, "
        "grain, key/fill, colour)."
    ),
    "science": (
        "Scientific subject matter (astronomy, elements, mathematics, medical). Use it as "
        "accurate, concrete visual content - real phenomena, instruments, environments - "
        "and get the details right."
    ),
    "geography": (
        "Country, nationality, territory. Set the place and its real look: landscape, "
        "architecture, climate, signage language, people and dress typical of it."
    ),
    "architecture": (
        "Buildings, interiors, materials, architectural styles. Use as the built "
        "environment - describe structure, materials, scale and light behaviour "
        "concretely; keep it consistent across shots."
    ),
    "art": "Artistic medium, style, palette, technique. Drives the stated visual style and how surfaces and light are rendered.",
    "artist": "Named artists / illustrators as style anchors. Translate into their observable visual traits (palette, line, light, composition); do not just name them.",
    "brands": "Brand names. Show them as products, logos on props or wardrobe, storefronts and signage (in double quotes when text is readable), only where the idea allows.",
    "character": "Character archetypes / fictional figures. Cast them as the on-screen subject(s) with a concrete physical description, wardrobe and behaviour.",
    "fashion": "Wardrobe, accessories, hairstyles, designers. Dress the subjects accordingly and restate the wardrobe anchors in every shot.",
    "human": "Occupations, hobbies, groups, festivities. Use to define who the people are, what they are doing and the social setting.",
    "interaction": "How people interact (couples, groups, crowds, individuals). Stage the physical interaction beat by beat, with clear positions.",
    "keywords": "Loose modifiers / genre / trending keywords. Treat as flavour: fold the useful ones into style, mood and detail; ignore anything that contradicts the idea.",
    "people": "Archetypes, body types, expressions, eye colour. Use as concrete physical description of the subject(s), restated per shot.",
    "stuff": "Objects and props (everyday, seasonal, sci-fi, gadgets). Put them in the scene as physical props the characters handle or that dress the set.",
    "typography": "Fonts, word art, text styles. Only for readable on-screen text (titles, signs) - write the text in double quotes and describe the lettering style.",
    "vehicle": "Vehicles. Put them in the scene as physical objects (make, era, colour, condition) that are driven, ridden or parked, with correct scale.",
    "video_game": "Games, engines, designers, actions. Use as visual/gameplay reference: the look, the world, the kind of action - not UI overlays unless asked.",
    # Our own nodes
    "cast": "Cast lines `Character (played by Actor) from Show`. These are the characters to write for; use the strings verbatim in subject_definitions and play them in character.",
    "h3_prompt": "An existing H3 prompt. Treat it as reference material for style, subjects and continuity - do not copy it back unless asked to revise it.",
    "context": "Free-form context. Use whatever in it helps make the scene concrete and consistent; the idea/direction still leads.",
}

_KNOWN_CLASSES = {
    "H3Characters": "cast",
    "H3ClaudeCodeBaseWriter": "h3_prompt",
    "H3ClaudeCodeRefWriter": "h3_prompt",
    "H3ClaudeCodeRefiner": "h3_prompt",
    "H3ClaudeCodeContinueWriter": "h3_prompt",
    "H3BasePromptWriter": "h3_prompt",
    "H3RefPromptWriter": "h3_prompt",
    "H3ScenePick": "h3_prompt",
}


def context_inputs():
    """The optional context sockets, to merge into a node's `optional` inputs."""
    return {
        name: ("STRING", {
            "forceInput": True,
            "tooltip": (
                "Steering input from another APNext node (Time, Scene, Poses, Plots, "
                "Feelings, Cinematic, Science, Geography, Architecture, Fashion, ...). "
                "The node detects which kind it is and tells Claude how to use it. "
                "Sockets grow as you connect them."
            ),
        })
        for name in CONTEXT_SOCKETS
    }


def context_hidden_inputs():
    """Hidden inputs that let the node inspect the graph it runs in."""
    return {HIDDEN_GRAPH: "PROMPT", HIDDEN_NODE_ID: "UNIQUE_ID"}


def pop_context(kwargs):
    """
    Remove context sockets and hidden graph inputs from a **kwargs dict, so
    other consumers of that dict (typed references, image slots) stay clean.
    Returns (context_values_by_socket, graph, node_id).
    """
    values = {name: kwargs.pop(name, None) for name in CONTEXT_SOCKETS}
    graph = kwargs.pop(HIDDEN_GRAPH, None)
    node_id = kwargs.pop(HIDDEN_NODE_ID, None)
    return values, graph, node_id


def _category_for(class_type):
    if not class_type:
        return "context", ""
    if class_type in _KNOWN_CLASSES:
        return _KNOWN_CLASSES[class_type], class_type
    if class_type.endswith("PromptNode"):
        stem = class_type[: -len("PromptNode")]
        category = re.sub(r"(?<!^)(?=[A-Z])", "_", stem).lower()
        return (category if category in CATEGORY_GUIDANCE else "context"), f"APNext {stem}"
    return "context", class_type


def _active_fields(inputs):
    fields = []
    for key, value in (inputs or {}).items():
        if key in _NON_FIELD_INPUTS or isinstance(value, list):
            continue
        if value in _INACTIVE_VALUES:
            continue
        fields.append(f"{key}={value}" if value not in ("Random", "Multiple Random") else f"{key} ({value})")
    return fields


def resolve_context(values, graph, node_id):
    """
    Turn the raw socket values into labelled entries:
    [{"socket", "category", "source", "fields", "text", "guidance"}, ...]
    """
    entries = []
    my_inputs = {}
    if isinstance(graph, dict) and node_id is not None:
        me = graph.get(str(node_id)) or graph.get(node_id) or {}
        my_inputs = me.get("inputs", {}) if isinstance(me, dict) else {}

    for socket in CONTEXT_SOCKETS:
        text = values.get(socket)
        if not isinstance(text, str) or not text.strip():
            continue
        category, source, fields = "context", "", []
        link = my_inputs.get(socket)
        if isinstance(link, list) and len(link) >= 1 and isinstance(graph, dict):
            src = graph.get(str(link[0])) or {}
            category, source = _category_for(src.get("class_type"))
            if source.startswith("APNext "):
                fields = _active_fields(src.get("inputs"))
        entries.append({
            "socket": socket,
            "category": category,
            "source": source,
            "fields": fields,
            "text": text.strip(),
            "guidance": CATEGORY_GUIDANCE.get(category, CATEGORY_GUIDANCE["context"]),
        })
    return entries


def context_block(entries, target="the scene"):
    """The labelled block appended to a Claude Code user prompt; "" if nothing is wired."""
    if not entries:
        return ""
    lines = [
        "CONTEXT FROM CONNECTED APNEXT NODES",
        f"Each item below is a steering input of a specific kind. Reconcile them into one "
        f"coherent version of {target}; the user's own idea/direction still leads. Where an "
        "input lists several options, pick what fits and make it concrete and visible on "
        "screen. Never paste an input's wording back verbatim as a list of tags - translate "
        "it into filmed detail.",
    ]
    for i, e in enumerate(entries, 1):
        header = f"{i}. [{e['category']}]"
        if e["source"]:
            header += f" from {e['source']}"
        if e["fields"]:
            header += f" (fields: {', '.join(e['fields'])})"
        lines.append(header)
        lines.append(f"   Input: {e['text']}")
        lines.append(f"   Use it as: {e['guidance']}")
    return "\n".join(lines)


def build_context(kwargs, target="the scene"):
    """
    One-call helper for node functions: pops the sockets from **kwargs, resolves
    them against the graph and returns (block_text, entries).
    """
    values, graph, node_id = pop_context(kwargs)
    entries = resolve_context(values, graph, node_id)
    return context_block(entries, target), entries


def with_context(user_prompt, block):
    """Append the context block to a finished user prompt (before nothing else)."""
    if not block:
        return user_prompt
    return f"{user_prompt.rstrip()}\n\n{block}\n"


def context_summary(entries):
    """Short log line: 'time, scene, poses' or 'none'."""
    return ", ".join(e["category"] for e in entries) or "none"
