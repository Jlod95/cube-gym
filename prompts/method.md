# Solving the cube: beginner layer-by-layer method

You are solving a 3x3 Rubik's cube one action at a time. After each action you
receive the new state of the cube. Work through the method's 7 stages in order.

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

Example: a solved cube shows `Y Y Y` three times under `U (top)` and `W W W`
three times under `D (bottom)`.

## Pieces

- **Edges** have 2 stickers, **corners** have 3, and **centres** have 1 and never
  move.
- A piece is **correct** when every one of its stickers matches the centre of the
  face it is on.
- A **slot** is a place for a piece, named by the faces it touches, e.g. the
  front-right bottom corner.

## The 7 stages

Each stage includes all the stages before it. Your progress counts how many
stages are complete in a row from stage 1, so breaking an earlier stage loses
the later ones too. The exception is stage 7, where the lower layers are
expected to come apart temporarily (see twist_corner).

1. **Cross**: all four bottom edges are correct: white facing down, and each
   side sticker matches its side centre.
2. **First layer**: the cross, plus all four bottom corners correct: white facing
   down, and side stickers matching the centres.
3. **Second layer**: the first layer, plus all four middle-layer edges correct.
4. **Yellow cross**: the second layer, plus all four top edges with yellow facing
   up. Their side stickers don't need to match yet.
5. **Yellow edges**: the yellow cross, plus each top edge's side sticker matching
   its side centre.
6. **Corners positioned**: the yellow edges, plus each top corner in its correct
   slot. Its three colours match the three centres around it, but it may be
   twisted.
7. **Solved**: every sticker matches its face's centre.

## Tools

### turn(move)

Turns one face a quarter turn. Moves: `U U' D D' L L' R R' F F' B B'`.

- A letter alone turns that face 90° clockwise, as seen looking directly at
  that face.
- A letter with `'` turns it 90° anticlockwise.
- For a half turn, call turn twice.

### apply_algorithm(name, front, times)

Performs a named algorithm `times` times in a row. `times` is a whole number
from 1 to 6 (default 1); any other value is rejected as an invalid action.

- **front** is one of `green`, `orange`, `blue`, `red`, the colour of a side
  centre. The algorithm is performed as if that colour faces you. "Front",
  "right", "left" and "back" in the descriptions below are relative to that
  colour, and U and D are always top and bottom.
- Front and right for each choice:
  - `green`: right is orange
  - `orange`: right is blue
  - `blue`: right is red
  - `red`: right is green
- The cube itself never moves; only the faces turn.

| name | moves (front = green) | stage | what it does |
|------|------------------------|-------|--------------|
| corner_insert | R U R' U' | 2 | Works on the front-right corners. Repeating it brings a white corner from the top front-right slot into the bottom front-right slot, one twist at a time. |
| edge_left | U' L' U L U F U' F' | 3 | Moves the top front edge into the front-left middle slot. |
| edge_right | U R U' R' U' F' U F | 3 | Moves the top front edge into the front-right middle slot. |
| yellow_cross | F R U R' U' F' | 4 | Changes which top edges have yellow facing up. Keeps the first two layers intact. |
| swap_edges | R U R' U R U2 R' U | 5 | Swaps the top front and top left edges. The top back and top right edges stay in place. Keeps yellow edges facing up. |
| cycle_corners | U R U' L' U R' U' L | 6 | Keeps the top front-right corner in place and cycles the other three top corners. Keeps the top edges intact. |
| twist_corner | R' D' R D | 7 | Twists the top front-right corner in place. It scrambles the lower layers along the way. |

## Rules for each turn

- **One tool call per turn.** Only your first tool call is carried out; any
  others in the same reply are rejected and not run.
- **Replying without a tool call ends the episode.** Don't stop to explain or
  ask questions; keep calling tools until the cube is solved. Only reply with
  plain text if you are giving up, and don't claim the cube is solved unless it
  is.
- **Limits.** You have at most 60 actions (tool calls) and 300 quarter turns in
  total. An algorithm counts one quarter turn per move, times `times`. The
  episode ends when either limit is reached.
- Invalid tool calls (unknown move, algorithm, front or `times`) still use up an
  action. They return an error message and leave the cube unchanged.

## How to do each stage

### Stage 1: cross (no algorithm)

Plan the cross yourself with single turns.

- For each white edge, find where it is and bring it into its bottom slot with
  white facing down, so its side sticker matches the side centre.
- A common approach: bring the edge to the top layer, turn U until its side
  sticker is above the matching centre, then turn that face twice so it drops
  into the bottom layer.
- Check that each new turn doesn't knock out an edge you have already placed.
  If it does, restore it afterwards.

### Stage 2: first layer, with corner_insert

For each white corner:

1. If it is in the top layer, turn U until it sits directly above its target
   slot. The target slot is the bottom corner between the centres matching its
   two non-white colours.
2. Choose `front` so that slot is the front-right slot.
3. Apply corner_insert repeatedly until the corner is correct. That takes 1, 3
   or 5 times, depending on how it is twisted. Use `times` to do it in one call.

If a white corner is in a bottom slot but wrong (the wrong slot, or twisted),
make that slot the front-right one and apply corner_insert once to lift it into
the top layer. Then continue as above.

### Stage 3: second layer, with edge_left / edge_right

1. Find a top-layer edge with no yellow on it.
2. Turn U until its side sticker matches the centre below it. That centre's
   colour is your `front`.
3. Look at its top sticker:
   - if it matches the centre to the left of front, apply edge_left
   - if it matches the centre to the right, apply edge_right

If a middle edge is in the wrong slot or flipped, and no yellow-free edges are
left on top, apply edge_left or edge_right with that slot as the front-left or
front-right slot. This lifts the edge into the top layer. Then insert it as
above.

### Stage 4: yellow cross, with yellow_cross

Look only at the four top edges, ignoring corners, and count those with yellow
facing up:

- **Dot** (none up): apply yellow_cross once with any front. This gives an L
  shape; handle it as below. That makes 3 applications in all.
- **L shape** (two adjacent edges up): choose `front` so the two yellow edges are
  at the back and the left. Apply yellow_cross 2 times.
- **Line** (two opposite edges up): choose `front` so the line runs left to
  right. Apply yellow_cross once.

### Stage 5: yellow edges, with swap_edges

1. Turn U until at least two top edges match their side centres. If all four
   match, the stage is complete.
2. If the two matching edges are **adjacent**, choose `front` so they are at the
   back and the right. Apply swap_edges once.
3. If they are **opposite**, apply swap_edges once with any front. Then go back
   to step 1.

### Stage 6: corners positioned, with cycle_corners

A top corner is positioned when its three colours match the three centres around
it, in any twist.

- If one corner is positioned, choose `front` so it is the top front-right
  corner. Apply cycle_corners once, then check. If the corners are not all
  positioned yet, apply it once more with the same front.
- If no corner is positioned, apply cycle_corners once with any front. Then one
  will be; continue as above.

### Stage 7: solved, with twist_corner

1. Choose a `front` and keep it for the whole stage. Never change it partway
   through.
2. If the top front-right corner shows something other than yellow on top,
   apply twist_corner until yellow is on top. That is always 2 or 4 times.
3. Turn **U only** to bring the next twisted corner to the top front-right
   position. Don't use any other turn or algorithm between corners.
4. Repeat until every top corner has yellow facing up.
5. Turn U until the top layer lines up with the centres.

While you are partway through, the lower layers will look scrambled. **This is
expected; do not try to fix them.** They come back by themselves once every top
corner has been twisted correctly.

## General advice

- Before each action, identify which stage you are on: the first one that isn't
  complete.
- Prefer apply_algorithm over single turns from stage 2 onward.
- Single turns between algorithms should only be U turns (to position pieces).
  The exception is stage 1, and lifting stuck pieces out as described above.
