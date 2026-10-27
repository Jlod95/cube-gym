"""Scripted beginner-method solver for stages 2-7, following prompts/method.md.

solve(state, target) returns (actions, final_state). Actions are tool calls
in the harness format: ("turn", {"move": m}) or
("apply_algorithm", {"name": ..., "front": ..., "times": n}). Stage 1 (the
cross) is not scripted. Used as a reference to check that stage tasks are
solvable with the method's tools.
"""

from algorithms import apply_algorithm
from cube import apply, is_solved
from stages import _corner_in_slot, _cubie_correct, _stickers, stages_complete

FRONTS = ["green", "orange", "blue", "red"]
FACE_OF = {"green": 2, "orange": 1, "blue": 5, "red": 4}  # index in URFDLB
RIGHT = {"green": "orange", "orange": "blue", "blue": "red", "red": "green"}
LEFT = {v: k for k, v in RIGHT.items()}
BACK = {f: RIGHT[RIGHT[f]] for f in FRONTS}
U_EDGE_AT = {"green": (0, 1, 1), "orange": (1, 1, 0), "blue": (0, 1, -1), "red": (-1, 1, 0)}
CORNERS = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
EDGES = [p for p in [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]
         if sorted(map(abs, p)) == [0, 1, 1]]


def front_right_xz(front):
    a, b = U_EDGE_AT[front], U_EDGE_AT[RIGHT[front]]
    return (a[0] + b[0], a[2] + b[2])


def front_for_slot(x, z):
    """The front colour whose front-right slots are at (x, z)."""
    return next(f for f in FRONTS if front_right_xz(f) == (x, z))


class _Run:
    def __init__(self, state):
        self.state = state
        self.actions = []

    def turn(self, move):
        self.state = apply(self.state, move)
        self.actions.append(("turn", {"move": move}))

    def alg(self, name, front, times=1):
        self.state = apply_algorithm(self.state, name, front, times)
        self.actions.append(("apply_algorithm", {"name": name, "front": front, "times": times}))

    def centre(self, face):
        return self.state[face * 9 + 4]

    def colours(self, pos):
        return {self.state[i] for i, _ in _stickers(pos)}

    def find(self, colours, positions):
        return next(p for p in positions if self.colours(p) == colours)


def _first_layer(r):
    for _ in range(50):
        if stages_complete(r.state) >= 2:
            return
        for x, z in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            slot = (x, -1, z)
            if _cubie_correct(r.state, slot):
                continue
            want = {r.centre(3), r.centre(1 if x == 1 else 4), r.centre(2 if z == 1 else 5)}
            pos = r.find(want, CORNERS)
            if pos[1] == -1:  # stuck in the bottom layer: lift it out
                r.alg("corner_insert", front_for_slot(pos[0], pos[2]))
                pos = r.find(want, CORNERS)
            for _ in range(4):
                if (pos[0], pos[2]) == (x, z):
                    break
                r.turn("U")
                pos = r.find(want, CORNERS)
            front = front_for_slot(x, z)
            for _ in range(6):
                if _cubie_correct(r.state, slot):
                    break
                r.alg("corner_insert", front)
            break


def _second_layer(r):
    top_edges = [U_EDGE_AT[f] for f in FRONTS]
    yellow = r.centre(0)
    for _ in range(50):
        if stages_complete(r.state) >= 3:
            return
        loose = [p for p in top_edges if yellow not in r.colours(p)]
        if loose:
            colours = r.colours(loose[0])
            for _ in range(4):
                pos = r.find(colours, top_edges)
                (side_i, side_f), = [(i, f) for i, f in _stickers(pos) if f != 0]
                if r.state[side_i] == r.centre(side_f):
                    break
                r.turn("U")
            front = next(f for f in FRONTS if FACE_OF[f] == side_f)
            top = next(r.state[i] for i, f in _stickers(pos) if f == 0)
            r.alg("edge_left" if top == r.centre(FACE_OF[LEFT[front]]) else "edge_right", front)
        else:  # a middle edge is wrong and none are loose on top: lift it out
            x, _, z = next(p for p in [(1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1)]
                           if not _cubie_correct(r.state, p))
            r.alg("edge_right", front_for_slot(x, z))


def _yellow_up_set(r):
    return {f for f in FRONTS
            if all(r.state[i] == r.centre(0) for i, face in _stickers(U_EDGE_AT[f]) if face == 0)}


def _yellow_cross(r):
    for _ in range(4):
        up = _yellow_up_set(r)
        if len(up) == 4:
            return
        if not up:
            r.alg("yellow_cross", "green")
            continue
        for f in FRONTS:
            if up == {BACK[f], LEFT[f]}:
                r.alg("yellow_cross", f, 2)
                break
            if up == {LEFT[f], RIGHT[f]}:
                r.alg("yellow_cross", f, 1)
                break


def _yellow_edges(r):
    for _ in range(6):
        for _ in range(4):
            matching = {f for f in FRONTS if _cubie_correct(r.state, U_EDGE_AT[f])}
            if len(matching) >= 2:
                break
            r.turn("U")
        if len(matching) == 4:
            return
        adjacent = [f for f in FRONTS if matching == {BACK[f], RIGHT[f]}]
        r.alg("swap_edges", adjacent[0] if adjacent else "green")


def _top_corner(front):
    x, z = front_right_xz(front)
    return (x, 1, z)


def _align_top(r):
    """Turn U until the top edges match their centres (stages 5-6 count up
    to a U turn, so they may be complete but not lined up)."""
    for _ in range(4):
        if all(_cubie_correct(r.state, U_EDGE_AT[f]) for f in FRONTS):
            return
        r.turn("U")


def _corners_positioned(r):
    _align_top(r)
    for _ in range(6):
        ok = [f for f in FRONTS if _corner_in_slot(r.state, _top_corner(f))]
        if len(ok) == 4:
            return
        r.alg("cycle_corners", ok[0] if ok else "green")


def _twist_corners(r, front="green"):
    corner = _top_corner(front)
    u_sticker = next(i for i, f in _stickers(corner) if f == 0)
    for _ in range(4):
        if all(r.state[i] == r.centre(0) for i in range(9)):  # all yellow up
            break
        twists = 0
        while r.state[u_sticker] != r.centre(0) and twists < 6:
            r.alg("twist_corner", front)
            twists += 1
        r.turn("U")
    for _ in range(4):
        if is_solved(r.state):
            return
        r.turn("U")


STAGE_SOLVERS = {2: _first_layer, 3: _second_layer, 4: _yellow_cross, 5: _yellow_edges,
                 6: _corners_positioned, 7: _twist_corners}


def solve(state, target=7):
    """Run the method from the current stage up to `target` (needs stage 1)."""
    r = _Run(state)
    for stage in range(stages_complete(state) + 1, target + 1):
        STAGE_SOLVERS[stage](r)
        if stages_complete(r.state) < stage:
            break
    return r.actions, r.state
