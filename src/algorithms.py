"""Named beginner-method algorithms, relabelled for any front face."""

from cube import apply

ALGORITHMS = {
    "corner_insert": "R U R' U'",
    "edge_left": "U' L' U L U F U' F'",
    "edge_right": "U R U' R' U' F' U F",
    "yellow_cross": "F R U R' U' F'",
    "swap_edges": "R U R' U R U2 R' U",
    "cycle_corners": "U R U' L' U R' U' L",
    "twist_corner": "R' D' R D",
}

# The cube never rotates: an algorithm written for a green front is
# relabelled so it acts the same way with `front` facing you.
FRONT_MAPS = {
    "green": {},
    "orange": {"F": "R", "R": "B", "B": "L", "L": "F"},
    "blue": {"F": "B", "R": "L", "B": "F", "L": "R"},
    "red": {"F": "L", "R": "F", "B": "R", "L": "B"},
}


def parse(moves):
    """Split a move string into quarter turns; X2 becomes X X."""
    out = []
    for token in moves.split():
        if token.endswith("2"):
            out += [token[:-1]] * 2
        else:
            out.append(token)
    return out


def relabel(moves, front):
    if front not in FRONT_MAPS:
        raise ValueError(f"unknown front colour {front!r}; expected one of {list(FRONT_MAPS)}")
    mapping = FRONT_MAPS[front]
    return [mapping.get(m[0], m[0]) + m[1:] for m in moves]


def apply_algorithm(state, name, front="green", times=1):
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r}; expected one of {list(ALGORITHMS)}")
    moves = relabel(parse(ALGORITHMS[name]), front)
    for _ in range(times):
        for move in moves:
            state = apply(state, move)
    return state
