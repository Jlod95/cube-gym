# Solving the cube

You will be shown a scrambled 3x3 Rubik's cube. Reply with a sequence of moves
that solves it.

## Orientation

The cube is held in a fixed orientation and **never rotates**:

| Face | Position | Centre colour |
|------|----------|---------------|
| U | top    | yellow (Y) |
| D | bottom | white (W)  |
| F | front  | green (G)  |
| R | right  | orange (O) |
| L | left   | red (R)    |
| B | back   | blue (B)   |

Centres never move, so a face's centre always tells you which colour belongs on
that face. Note that `R` is both the right face and the colour letter for red;
on the cube, red is the left face.

## Reading the observation

The observation lists the six faces in the order U, R, F, D, L, B. Each face has
a label line, e.g. `F (front)`, then 3 rows of 3 colour letters. The middle
letter is the centre.

Each face is drawn as you would see it looking straight at that face:

- **U (top)**: seen from above with the front at the bottom. The top row is next
  to B, the bottom row next to F, the left column next to L.
- **D (bottom)**: seen from below with the front at the top. The top row is next
  to F, the bottom row next to B, the left column next to L.
- **F, R, B, L (sides)**: seen from outside with U at the top. The top row is next
  to U, the bottom row next to D. Left columns: F next to L, R next to F, B next
  to R, L next to B.

The cube is solved when every sticker on each face matches that face's centre.

## Move notation

- `U D L R F B`: turn that face 90° clockwise, as seen looking directly at that
  face.
- `U' D' L' R' F' B'`: turn that face 90° anticlockwise.
- `U2 D2 L2 R2 F2 B2`: turn that face 180°.

## Answer format

Reply with your full solution on a single line starting with `MOVES:`, followed
by the moves separated by spaces. Use only the moves listed above. For example:

```
MOVES: R U R' U' F2 D
```

The moves will be applied in order to the cube as shown. Only the `MOVES:` line
is read.

Always give your best attempt as a MOVES line, even if you are unsure it solves
the cube. Never decline.
