"""Progress through the beginner layer-by-layer method (white on D)."""

from cube import _INDEX, FACES, NORMALS, apply, is_solved


def _stickers(pos):
    """(sticker index, face index) for each sticker on the cubie at pos."""
    out = []
    for f, face in enumerate(FACES):
        normal = NORMALS[face]
        if sum(p * n for p, n in zip(pos, normal)) == 1:
            out.append((_INDEX[(pos, normal)], f))
    return out


def _centre(state, f):
    return state[f * 9 + 4]


def _cubie_correct(state, pos):
    """Every sticker on the cubie matches the centre of the face it is on."""
    return all(state[i] == _centre(state, f) for i, f in _stickers(pos))


D_EDGES = [(0, -1, 1), (1, -1, 0), (0, -1, -1), (-1, -1, 0)]
D_CORNERS = [(x, -1, z) for x in (-1, 1) for z in (-1, 1)]
MID_EDGES = [(x, 0, z) for x in (-1, 1) for z in (-1, 1)]
U_EDGES = [(0, 1, 1), (1, 1, 0), (0, 1, -1), (-1, 1, 0)]
U_CORNERS = [(x, 1, z) for x in (-1, 1) for z in (-1, 1)]
U = FACES.index("U")


def _yellow_up(state, pos):
    return all(state[i] == _centre(state, U) for i, f in _stickers(pos) if f == U)


def _corner_in_slot(state, pos):
    stickers = _stickers(pos)
    return {state[i] for i, _ in stickers} == {_centre(state, f) for _, f in stickers}


def _u_turns(state):
    """The state after 0, 1, 2 and 3 quarter turns of U."""
    out = [state]
    for _ in range(3):
        out.append(apply(out[-1], "U"))
    return out


def _top_edges_ok(state):
    return all(_cubie_correct(state, p) for p in U_EDGES)


def _top_corners_ok(state):
    return all(_corner_in_slot(state, p) for p in U_CORNERS)


# Stages 5 and 6 count as complete up to a final U turn (AUF): they pass if
# some U turn (0-3 quarter turns) would make them hold. Stage 6 needs one turn
# that satisfies both its edges and corners. Stage 7 means exactly solved.
STAGES = [
    ("cross", lambda s: all(_cubie_correct(s, p) for p in D_EDGES)),
    ("first layer", lambda s: all(_cubie_correct(s, p) for p in D_CORNERS)),
    ("second layer", lambda s: all(_cubie_correct(s, p) for p in MID_EDGES)),
    ("yellow cross", lambda s: all(_yellow_up(s, p) for p in U_EDGES)),
    ("yellow edges", lambda s: any(_top_edges_ok(t) for t in _u_turns(s))),
    ("corners positioned",
     lambda s: any(_top_edges_ok(t) and _top_corners_ok(t) for t in _u_turns(s))),
    ("solved", is_solved),
]


def stages_complete(state):
    """Number of stages complete in a row from stage 1 (0 to 7).

    Each check above only adds its new condition; counting stops at the
    first failure, so stage n implies stages 1..n-1. Stages 5 and 6 are
    judged up to a U turn; stage 7 requires the exact solved state.
    """
    count = 0
    for _, check in STAGES:
        if not check(state):
            break
        count += 1
    return count
