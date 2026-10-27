"""Rubik's cube state, moves and a minimal RL environment.

State: tuple of 54 colour letters, 6 faces x 9 stickers, faces in order
U, R, F, D, L, B. Within a face, stickers are row-major as seen looking
at that face (U viewed with F at the bottom, D viewed with F at the top,
side faces viewed with U at the top).
"""

import random

FACES = "URFDLB"
FACE_COLORS = {"U": "Y", "R": "O", "F": "G", "D": "W", "L": "R", "B": "B"}
SOLVED = tuple(FACE_COLORS[f] for f in FACES for _ in range(9))

# Axes: x -> R, y -> U, z -> F.
NORMALS = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}


def _sticker_pos(face, r, c):
    """Cubie position (x, y, z) of sticker (row r, col c) on a face."""
    a, b = r - 1, c - 1  # each in -1..1
    return {
        "U": (b, 1, a),
        "R": (1, -a, -b),
        "F": (b, -a, 1),
        "D": (b, -1, -a),
        "L": (-1, -a, b),
        "B": (-b, -a, -1),
    }[face]


# (position, normal) -> sticker index
_INDEX = {}
for _f, _face in enumerate(FACES):
    for _i in range(9):
        _INDEX[(_sticker_pos(_face, _i // 3, _i % 3), NORMALS[_face])] = _f * 9 + _i


def _rot_cw(axis, v):
    """Rotate v by 90 degrees clockwise as seen looking down `axis`."""
    ax, ay, az = axis
    vx, vy, vz = v
    cross = (ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx)
    dot = ax * vx + ay * vy + az * vz
    return tuple(-cross[k] + axis[k] * dot for k in range(3))


def _face_turn_perm(face):
    """perm such that new_state[i] = state[perm[i]] for a clockwise turn."""
    axis = NORMALS[face]
    perm = list(range(54))
    for (pos, normal), src in _INDEX.items():
        if sum(p * a for p, a in zip(pos, axis)) == 1:  # sticker is in the turning layer
            dest = _INDEX[(_rot_cw(axis, pos), _rot_cw(axis, normal))]
            perm[dest] = src
    return tuple(perm)


def _invert(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


MOVES = {}
for _face in "UDLRFB":
    MOVES[_face] = _face_turn_perm(_face)
    MOVES[_face + "'"] = _invert(MOVES[_face])

MOVE_NAMES = tuple(MOVES)  # U U' D D' L L' R R' F F' B B'


def apply(state, move):
    perm = MOVES[move]
    return tuple(state[p] for p in perm)


def is_solved(state):
    return all(state[f * 9 + i] == state[f * 9 + 4] for f in range(6) for i in range(9))


def scramble(depth, seed):
    """Random move list; never turns the same face twice in a row."""
    rng = random.Random(seed)
    moves = []
    for _ in range(depth):
        choices = [m for m in MOVE_NAMES if not moves or m[0] != moves[-1][0]]
        moves.append(rng.choice(choices))
    return moves


class CubeEnv:
    def __init__(self):
        self.state = SOLVED

    def reset(self, depth, seed):
        self.state = SOLVED
        for move in scramble(depth, seed):
            self.state = apply(self.state, move)
        return self._observe()

    def step(self, move):
        self.state = apply(self.state, move)
        done = is_solved(self.state)
        return self._observe(), (1 if done else 0), done

    def snapshot(self):
        return self.state  # tuples are immutable, so no copy needed

    def restore(self, snap):
        self.state = snap

    def _observe(self):
        return observation(self.state)


FACE_POSITIONS = {"U": "top", "R": "right", "F": "front", "D": "bottom", "L": "left", "B": "back"}


def observation(state):
    """Plain-text view: each face labelled, then 3 rows of colour letters."""
    blocks = []
    for f, face in enumerate(FACES):
        rows = [" ".join(state[f * 9 + r * 3 : f * 9 + r * 3 + 3]) for r in range(3)]
        blocks.append(f"{face} ({FACE_POSITIONS[face]})\n" + "\n".join(rows))
    return "\n\n".join(blocks)
