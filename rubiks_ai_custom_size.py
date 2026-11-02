#!/usr/bin/env python3
"""
rubiks_ai.py - teach a neural network to solve the NxNxN Rubik's Cube (2x2x2, 3x3x3, 4x4x4 ... 8x8x8 ...).
               The cube size is chosen with --size.

THE METHOD (the "learn backwards" trick)
  1. Start from a SOLVED cube and apply k random quarter turns. The result is a cube that is at
     most k moves from solved, and we know the way back. Doing this millions of times gives an
     unlimited, free training set - no solver and no human knowledge needed.
  2. Train one network with two heads on those cubes:
        value head  - "how many moves away is this cube?"        (regression on k)
        policy head - "which move should I play next?"            (classification: undo the
                       last scramble move)
  3. To solve a cube, use the value head like a GPS (beam search): try all moves from the best
     candidate cubes, let the network score every resulting cube, keep the `beam` best-scoring
     ones, repeat until one of them is solved. beam=1 is the pure greedy "always take the move
     that lowers the distance fastest" version; a wider beam is far more reliable.

USAGE
  python rubiks_ai_custom_size.py selftest                       # checks the cube simulator, sizes 2 3 4 5 8 (no PyTorch needed)
  python rubiks_ai_custom_size.py selftest --size 6              # ...or one size only
  python rubiks_ai_custom_size.py train --size 3 --iters 20000   # trains and saves cube_model.pt   (3 is the default size)
  python rubiks_ai_custom_size.py train --size 4 --iters 20000 --save cube4.pt
  python rubiks_ai_custom_size.py eval  --model cube_model.pt    # solve rate per scramble depth
  python rubiks_ai_custom_size.py solve --model cube_model.pt --random 12
  python rubiks_ai_custom_size.py solve --model cube_model.pt --scramble "R U R' U' F2 D L'"
  python rubiks_ai_custom_size.py solve --model cube_model.pt --cube "<6*N*N facelet letters>"
  A model file remembers the cube size it was trained for, so eval / solve need no --size
  (if you give one anyway, it is checked against the model).

REQUIREMENTS: numpy, and PyTorch for train / eval / solve  (pip install numpy torch).

CUBE CONVENTIONS
  A cube of size N has 6*N*N stickers numbered in the standard "Kociemba" facelet order: the faces
  U R F D L B one after the other, each face read left-to-right, top-to-bottom as seen from outside
  (N rows of N stickers). Drawn for N = 3 (U1..U9 R1..R9 F1..F9 D1..D9 L1..L9 B1..B9):

              U1 U2 U3
              U4 U5 U6
              U7 U8 U9
     L1 L2 L3 F1 F2 F3 R1 R2 R3 B1 B2 B3
     L4 L5 L6 F4 F5 F6 R4 R5 R6 B4 B5 B6
     L7 L8 L9 F7 F8 F9 R7 R8 R9 B7 B8 B9
              D1 D2 D3
              D4 D5 D6
              D7 D8 D9

MOVES
  A move is a quarter turn of ONE layer: X = 90 degrees clockwise seen from face X, X' = counter-clockwise.
  Notation understood by parse_moves():
      R  R'  R2       the outer layer of face R (R2 = two quarter turns)
      2R  3R'         the 2nd / 3rd layer counted from face R (a single slice)
      Rw  3Rw'        the 2 / 3 outermost layers of R turned together
  The search itself uses one clockwise and one counter-clockwise quarter turn of every layer EXCEPT ONE PER
  AXIS, i.e. 6*(N-1) moves: 12 for 3x3x3 (exactly the old U U' R R' F F' D D' L L' B B'), 18 for 4x4x4,
  42 for 8x8x8. Turning all N layers of an axis only rotates the whole cube, so one layer per axis is
  redundant, and leaving it out means the cube is never rotated: every cube has exactly one solved state.
  The left-out layer is the middle one for odd N (the face centres then stay where they are) and the outer
  D, L and B layer for even N (so the DLB corner stays where it is). parse_moves() accepts every layer
  anyway: a turn of a left-out layer is rewritten as the opposite turn of all the other layers of that
  axis, which is the same cube just held differently (so "D" on a 4x4x4 becomes 3 engine moves).

USING THE ENGINE FROM OTHER CODE (e.g. a GUI)
  cube  = CubeEngine(4)                                  # any size >= 2
  state = cube.solved.copy()                             # numpy array: 6*N*N colour ids, 0..5 = U R F D L B
  state = cube.apply_moves(state, cube.parse_moves("R U 2R' Fw"))
  cube.render(state)   cube.facelets(state)   cube.is_solved(state)   cube.move_names   cube.n_moves
  cube.layer_moves("R", 1, 1)                            # engine moves for "turn layer 2 below R clockwise once"
  device = pick_device("auto")
  model, cube = load_model("cube4.pt", device)
  path = beam_search(cube, make_predictor(model, device), state, beam_width=2048)   # None if not solved
  print(cube.format_moves(path))
"""
import argparse
import contextlib
import math
import queue
import re
import sys
import threading
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # the cube simulator and the search work without PyTorch
    torch = None

# =============================================================================================
# 1. CUBE SIMULATOR (numpy). A cube is an array of 6*N*N colour ids; colour id = index in "URFDLB".
# =============================================================================================
FACES = "URFDLB"

# Outward normal of every face, and the directions in which the facelet numbering runs
# (column direction, row direction) when the face is viewed from outside.
_NORMAL = {"U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1), "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1)}
_READING = {
    "U": ((1, 0, 0), (0, 0, 1)),
    "R": ((0, 0, -1), (0, -1, 0)),
    "F": ((1, 0, 0), (0, -1, 0)),
    "D": ((1, 0, 0), (0, 0, -1)),
    "L": ((0, 0, 1), (0, -1, 0)),
    "B": ((-1, 0, 0), (0, -1, 0)),
}

_TOKEN = re.compile(r"(\d*)([URFDLB])(w?)(.*)")        # [layer number] face [w] [suffix], e.g. 3Rw'
_TURNS = {"": 1, "'": 3, "\u2019": 3, "2": 2, "2'": 2, "2\u2019": 2}   # suffix -> clockwise quarter turns


class CubeEngine:
    """Simulator of the NxNxN cube - everything that depends on the cube size lives in here.

    A cube is a numpy array of 6*N*N colour ids (colour id = index in "URFDLB") in facelet order.
      size, n_stickers, n_moves, move_names   the basics (n_moves = 6*(N-1))
      solved                                  the solved cube
      perms[m]                                new_state = state[perms[m]] applies move m
      inv[m]                                  the inverse move (m ^ 1); inv[n_moves] = n_moves means "no move"
    CubeEngine(3) is exactly the 3x3x3-only engine of the previous version of this script.
    """

    def __init__(self, size=3):
        try:
            n = int(size)
        except (TypeError, ValueError):
            raise ValueError("the cube size must be an integer of at least 2") from None
        if n < 2 or n != size:
            raise ValueError("the cube size must be an integer of at least 2")
        self.size = n
        # Public immutable-by-convention geometry for renderers and other clients.
        self.face_normals = {f: np.asarray(_NORMAL[f], dtype=float) for f in FACES}
        self.face_reading = _READING
        self.n_stickers = 6 * n * n
        # A move pair (clockwise + counter-clockwise) for every layer except one per axis. A layer is named
        # (face, depth): `depth` layers below that face, 0 = the outer layer of the face. Left out: the middle
        # layer when N is odd, and the outer D, L and B layers when N is even (see MOVES at the top).
        self.layers = [(f, d) for f in range(6) for d in range(1 if n % 2 == 0 and f >= 3 else 0, n // 2)]
        self.n_moves = 2 * len(self.layers)                                 # 6*(N-1)
        self.layer_names = [("" if d == 0 else str(d + 1)) + FACES[f] for f, d in self.layers]
        self.move_names = [name + s for name in self.layer_names for s in ("", "'")]   # U U' R R' F F' D D' ...
        self._pair = {layer: p for p, layer in enumerate(self.layers)}
        self._build_geometry()
        self.perms = self._build_move_permutations()
        self.inv = np.array([m ^ 1 for m in range(self.n_moves)] + [self.n_moves], dtype=np.int64)  # inverse move; n_moves = "none"
        self.solved = np.repeat(np.arange(6), n * n).astype(np.uint8)
        self._hash = np.random.default_rng(12345).integers(1, 2 ** 62, size=self.n_stickers, dtype=np.int64) | 1

    # ---- geometry -----------------------------------------------------------------------------
    def _build_geometry(self):
        n = self.size
        pos, nor = [], []        # cubie centre (doubled coordinates, so they stay integers) and outward normal of every sticker
        for f in FACES:
            nv = np.array(_NORMAL[f])
            col, row = (np.array(v) for v in _READING[f])
            for r in range(n):
                for c in range(n):
                    pos.append(nv * (n - 1) + (2 * c - (n - 1)) * col + (2 * r - (n - 1)) * row)
                    nor.append(nv)
        self._pos, self._nor = np.array(pos), np.array(nor)
        self.sticker_normals = self._nor.astype(float, copy=True)
        self._index = {(tuple(p), tuple(q)): i
                       for i, (p, q) in enumerate(zip(self._pos.tolist(), self._nor.tolist()))}
        m = n - 1
        # the 8 corner pieces (3 sticker numbers each), and the D, L, B stickers of the DLB corner
        self._corners = [[self._index[((sx * m, sy * m, sz * m), nv)] for nv in ((sx, 0, 0), (0, sy, 0), (0, 0, sz))]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        self._ref = [self._index[((-m, -m, -m), nv)] for nv in ((0, -1, 0), (-1, 0, 0), (0, 0, -1))]

    def _layer_perm(self, f, d):
        """Sticker permutation of turning the layer `d` layers below face number `f` clockwise (seen from that face)."""
        axis = np.array(_NORMAL[FACES[f]])

        def rotate(v):                              # -90 degrees about the outward normal = clockwise
            return -np.cross(axis, v) + axis * (v @ axis)[:, None]

        moving = np.flatnonzero(self._pos @ axis == (self.size - 1) - 2 * d)   # the stickers of the layer being turned
        perm = np.arange(self.n_stickers)
        for i, pos, nor in zip(moving, rotate(self._pos[moving]).tolist(), rotate(self._nor[moving]).tolist()):
            perm[self._index[(tuple(pos), tuple(nor))]] = i
        return perm

    def _build_move_permutations(self):
        """Derive the sticker permutations from real 3D geometry (so no hand-typed tables to get wrong).

        new_state = state[perms[m]] applies move m.
        """
        perms = np.zeros((self.n_moves, self.n_stickers), dtype=np.int64)
        for p, (f, d) in enumerate(self.layers):
            perm = self._layer_perm(f, d)
            perms[2 * p] = perm                     # clockwise
            perms[2 * p + 1] = np.argsort(perm)     # counter-clockwise = inverse permutation
        return perms

    # ---- states and moves ---------------------------------------------------------------------
    def hash_states(self, states):
        """64-bit fingerprint of each cube (used to drop duplicate / already visited cubes)."""
        return (states.astype(np.int64) * self._hash).sum(axis=1)

    def apply_moves(self, state, moves):
        for m in moves:
            state = state[self.perms[m]]
        return state

    def is_solved(self, state):
        return bool(np.array_equal(state, self.solved))

    def layer_moves(self, face, depth, turns=1):
        """Engine moves that turn the layer `depth` layers below `face` (0 = the face's outer layer) by `turns`
        clockwise quarter turns seen from that face (negative = counter-clockwise). `face` is a letter or 0..5."""
        if isinstance(face, str):
            face = FACES.index(face)
        if not 0 <= depth < self.size:
            raise ValueError(f"layer {depth + 1} does not exist on a {self.size}x{self.size}x{self.size} cube")
        t = turns % 4
        if t == 0:
            return []
        for f, d, sign in ((face, depth, 1), ((face + 3) % 6, self.size - 1 - depth, -1)):   # the same layer seen from the other side
            p = self._pair.get((f, d))
            if p is not None:
                t2 = t if sign == 1 else (-t) % 4
                return [2 * p] * t2 if t2 < 3 else [2 * p + 1]
        # This layer has no moves of its own. Turning all N layers of an axis is only a rotation of the whole
        # cube, so turning this layer equals turning all the other layers of the axis the opposite way.
        return [m for d in range(self.size) if d != depth for m in self.layer_moves(face, d, -turns)]

    def parse_moves(self, text):
        """"R U R' U2" -> list of move ids. Also understands 2R (a slice), Rw / 3Rw (wide turns); see MOVES at the top."""
        n, moves = self.size, []
        for tok in text.replace(",", " ").split():
            m = _TOKEN.fullmatch(tok)
            if m is None or m.group(4) not in _TURNS:
                raise ValueError(f"unknown move '{tok}' (use R, R', R2, 3R, Rw, or 3Rw; put layer numbers before the face)")
            num, face, wide, suffix = m.groups()
            if wide:                                 # 3Rw: the 3 outermost layers of R together
                width = int(num) if num else 2
                depths = range(width) if 1 <= width < n else None
            else:                                    # 3R: only the 3rd layer from R
                layer = int(num) if num else 1
                depths = [layer - 1] if 1 <= layer <= n else None
            if depths is None:
                raise ValueError(f"move '{tok}' does not exist on a {n}x{n}x{n} cube")
            for d in depths:
                moves += self.layer_moves(face, d, _TURNS[suffix])
        return moves

    def format_moves(self, moves):
        """Pretty-print a move list, merging repeats: R R -> R2, R R R -> R'."""
        out, i = [], 0
        while i < len(moves):
            j = i
            while j < len(moves) and moves[j] == moves[i]:
                j += 1
            name, ccw = self.layer_names[moves[i] // 2], moves[i] % 2 == 1
            turns = (j - i) % 4 if not ccw else (-(j - i)) % 4          # net clockwise quarter turns
            if turns == 1:
                out.append(name)
            elif turns == 2:
                out.append(name + "2")
            elif turns == 3:
                out.append(name + "'")
            i = j
        return " ".join(out)

    def state_from_facelets(self, text):
        """Build a cube from 6*N*N letters in the facelet order described at the top of this file.

        Any six letters can be used for the colours (e.g. "WRGYOB"): colours are identified by pieces that never
        move - the face centres (odd N; for N = 3 the stickers 5, 14, 23, 32, 41, 50), or for even N, which has no
        centres, the corner pieces around the DLB corner.
        """
        n = self.size
        s = "".join(text.split())
        if len(s) != self.n_stickers:
            raise ValueError(f"expected {self.n_stickers} stickers for a {n}x{n}x{n} cube, got {len(s)}")
        if n % 2:
            colours = [s[f * n * n + (n // 2) * n + n // 2] for f in range(6)]    # the centre sticker of every face
        else:
            cd, cl, cb = (s[i] for i in self._ref)         # the DLB corner never moves: it shows the colours of D, L, B
            pieces = [{s[i] for i in c} for c in self._corners]

            def third(a, b, other):       # the 3rd colour of the corner piece that has colours a and b but not `other`
                for p in pieces:
                    if len(p) == 3 and a in p and b in p and other not in p:
                        return next(iter(p - {a, b}))
                raise ValueError("cannot tell the six colours apart: the corner pieces do not belong to one cube")

            colours = [third(cl, cb, cd), third(cd, cb, cl), third(cd, cl, cb), cd, cl, cb]     # U R F D L B
        if len(set(colours)) != 6:
            raise ValueError("the six face colours (centres, or corner pieces for even sizes) must be six different colours")
        colour_of = {ch: i for i, ch in enumerate(colours)}
        try:
            state = np.array([colour_of[ch] for ch in s], dtype=np.uint8)
        except KeyError as e:
            raise ValueError(f"sticker colour {e} does not match any face colour") from None
        if not (np.bincount(state, minlength=6) == n * n).all():
            raise ValueError(f"every colour must appear exactly {n * n} times")
        return state

    def facelets(self, state, letters=FACES):
        """The 6*N*N colour letters of a cube (the inverse of state_from_facelets)."""
        return "".join(letters[c] for c in state)

    def render(self, state):
        """Text picture of the cube as an unfolded net."""
        n = self.size

        def row(face, r):
            return " ".join(FACES[state[face * n * n + r * n + c]] for c in range(n))
        pad = " " * (2 * n + 1)                                                 # as wide as one face plus the gap
        lines = [pad + row(0, r) for r in range(n)]
        lines += ["  ".join(row(face, r) for face in (4, 2, 1, 5)) for r in range(n)]   # L F R B
        lines += [pad + row(3, r) for r in range(n)]
        return "\n".join(lines)

    # ---- training data: random walks away from the solved cube --------------------------------
    def random_walks(self, n, depths, rng):
        """Scramble n solved cubes with random quarter turns.

        depths: an int, or an array with the scramble length of every cube.
        Returns (states (n, 6*N*N) uint8, depth (n,), last_move (n,)); last_move is n_moves where depth == 0.

        The walk never undoes the previous move and never repeats the same move 3 times in a row
        (X X X = X'), so a cube scrambled with k moves is usually really about k moves from solved.
        """
        nm, inv = self.n_moves, self.inv
        depths = np.broadcast_to(np.asarray(depths), (n,)).astype(np.int64)
        states = np.tile(self.solved, (n, 1))
        prev1 = np.full(n, nm)
        prev2 = np.full(n, nm)
        for t in range(int(depths.max()) if n else 0):
            active = depths > t
            a = rng.integers(0, nm, size=n)
            bad = active & ((a == inv[prev1]) | ((a == prev1) & (prev1 == prev2)))
            while bad.any():
                a[bad] = rng.integers(0, nm, size=int(bad.sum()))
                bad = active & ((a == inv[prev1]) | ((a == prev1) & (prev1 == prev2)))
            moved = np.take_along_axis(states, self.perms[a], axis=1)
            states = np.where(active[:, None], moved, states)
            prev2 = np.where(active, prev1, prev2)
            prev1 = np.where(active, a, prev1)
        return states, depths, prev1


def _make_cube(size):
    """Construct a cube engine; validation errors are raised to the caller."""
    return CubeEngine(size)


# =============================================================================================
# 2. TRAINING DATA: random walks away from the solved cube (CubeEngine.random_walks)
# =============================================================================================
class BatchGenerator:
    """Background batch producer whose failures are always propagated to the trainer."""

    def __init__(self, cube, batch, max_depth, seed=0, queue_size=8):
        if batch < 1 or max_depth < 0:
            raise ValueError("batch must be positive and max_depth must be non-negative")
        self.cube, self.batch, self.max_depth = cube, batch, max_depth
        self.rng = np.random.default_rng(seed)
        self.q = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._work, name="cube-batch-generator", daemon=True)
        self.thread.start()

    def _work(self):
        try:
            while not self.stop_event.is_set():
                depths = self.rng.integers(0, self.max_depth + 1, size=self.batch)
                batch = self.cube.random_walks(self.batch, depths, self.rng)
                while not self.stop_event.is_set():
                    try:
                        self.q.put(batch, timeout=0.25)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()

    def get(self, timeout=1.0):
        while True:
            if self.error is not None:
                raise RuntimeError("training batch generator failed") from self.error
            try:
                return self.q.get(timeout=timeout)
            except queue.Empty:
                if not self.thread.is_alive():
                    if self.error is not None:
                        raise RuntimeError("training batch generator failed") from self.error
                    raise RuntimeError("training batch generator stopped unexpectedly")

    def close(self):
        self.stop_event.set()



# =============================================================================================
# 3. SEARCH: beam search guided by the network ("GPS" mode)
# =============================================================================================
def default_search_steps(cube):
    """A size-aware search horizon; checkpoints may carry their training scramble depth."""
    trained_depth = int(getattr(cube, "training_max_depth", 0) or 0)
    return max(50, 4 * cube.size * cube.size, 2 * trained_depth)


def _beam_search_torch(cube, predict, start, beam_width, max_steps, policy_weight):
    """Tensor-native beam expansion and duplicate filtering for CUDA predictors."""
    device = predict.device
    torch_predict = predict.torch_predict
    nm = cube.n_moves
    perms = torch.as_tensor(cube.perms, dtype=torch.long, device=device)
    hash_keys = torch.as_tensor(cube._hash, dtype=torch.int64, device=device)
    solved = torch.as_tensor(cube.solved, dtype=torch.uint8, device=device)

    def hashes(states):
        return (states.to(torch.int64) * hash_keys).sum(dim=1)

    beam = torch.as_tensor(np.asarray(start, dtype=np.uint8), device=device).unsqueeze(0)
    beam_logp = torch_predict(beam)[1] if policy_weight else None
    seen = hashes(beam)
    history = []

    for _ in range(max_steps):
        b = beam.shape[0]
        children = beam[:, perms].reshape(b * nm, cube.n_stickers)
        parent = torch.arange(b, device=device).repeat_interleave(nm)
        move = torch.arange(nm, device=device).repeat(b)

        hit = torch.nonzero((children == solved).all(dim=1), as_tuple=False).flatten()
        if hit.numel():
            c = int(hit[0].item())
            path, i = [int(move[c].item())], int(parent[c].item())
            for par, mv in reversed(history):
                path.append(int(mv[i].item()))
                i = int(par[i].item())
            return path[::-1]

        h = hashes(children)
        fresh = ~torch.isin(h, seen)
        keep = torch.nonzero(fresh, as_tuple=False).flatten()
        if not keep.numel():
            return None
        children, parent, move, h = children[keep], parent[keep], move[keep], h[keep]

        # Stable, first-occurrence deduplication without copying sticker states back to the CPU.
        order = torch.argsort(h, stable=True)
        sorted_hashes = h[order]
        first_mask = torch.ones_like(sorted_hashes, dtype=torch.bool)
        first_mask[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
        first = torch.sort(order[first_mask]).values
        children, parent, move, h = children[first], parent[first], move[first], h[first]

        value, logp = torch_predict(children)
        score = value - policy_weight * beam_logp[parent, move] if policy_weight else value
        take = min(beam_width, score.numel())
        selected = torch.topk(score, k=take, largest=False).indices
        beam = children[selected]
        history.append((parent[selected], move[selected]))
        seen = torch.cat((seen, h[selected]))
        beam_logp = logp[selected]
    return None


def beam_search(cube, predict, start, beam_width=1024, max_steps=None, policy_weight=0.0):
    """Find a move sequence that solves `start`, or return None.

    When max_steps is omitted, use a size-aware horizon (and checkpoint training depth when known).
    A finite beam search is still heuristic: a larger horizon does not guarantee a solution for an
    arbitrary legal state, but it avoids imposing the same 50-step ceiling on every cube size.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if max_steps is None:
        max_steps = default_search_steps(cube)
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    nm, perms, solved = cube.n_moves, cube.perms, cube.solved
    start = np.asarray(start, dtype=np.uint8)
    if start.shape != (cube.n_stickers,):
        raise ValueError(f"start must contain exactly {cube.n_stickers} stickers")
    if cube.is_solved(start):
        return []
    if (torch is not None and getattr(predict, "torch_predict", None) is not None
            and getattr(getattr(predict, "device", None), "type", None) == "cuda"):
        return _beam_search_torch(cube, predict, start, beam_width, max_steps, policy_weight)
    beam = start[None, :]
    beam_logp = predict(beam)[1] if policy_weight else None
    seen = cube.hash_states(beam)
    history = []

    for _ in range(max_steps):
        b = len(beam)
        children = beam[:, perms].reshape(b * nm, cube.n_stickers)
        parent = np.repeat(np.arange(b), nm)
        move = np.tile(np.arange(nm), b)

        hit = np.flatnonzero((children == solved).all(axis=1))
        if hit.size:
            c = int(hit[0])
            path, i = [int(move[c])], int(parent[c])
            for par, mv in reversed(history):
                path.append(int(mv[i]))
                i = int(par[i])
            return path[::-1]

        hashes = cube.hash_states(children)
        keep = ~np.isin(hashes, seen)
        children, parent, move, hashes = children[keep], parent[keep], move[keep], hashes[keep]
        if not len(children):
            return None
        _, first = np.unique(hashes, return_index=True)
        children, parent, move, hashes = children[first], parent[first], move[first], hashes[first]
        if not len(children):
            return None

        value, logp = predict(children)
        score = value - policy_weight * beam_logp[parent, move] if policy_weight else value
        if len(children) > beam_width:
            sel = np.argpartition(score, beam_width - 1)[:beam_width]
        else:
            sel = np.arange(len(children))
        beam = children[sel]
        history.append((parent[sel], move[sel]))
        seen = np.concatenate([seen, hashes[sel]])
        beam_logp = logp[sel]
    return None


def evaluate(cube, predict, depths, n, beam_width, rng, max_steps=None, policy_weight=0.0):
    """Solve n random scrambles of each depth, print the solve rate. Returns overall solve rate."""
    if n < 1:
        raise ValueError("evaluation count n must be positive")
    total = solved_total = 0
    for d in depths:
        starts, _, _ = cube.random_walks(n, d, rng)
        ok, lengths, t0 = 0, [], time.time()
        for s in starts:
            path = beam_search(cube, predict, s, beam_width, max_steps, policy_weight)
            if path is not None and cube.is_solved(cube.apply_moves(s, path)):
                ok += 1
                lengths.append(len(path))
        avg = f"{np.mean(lengths):.1f}" if lengths else "-"
        print(f"  scramble depth {d:2d}: solved {ok}/{n}   avg solution length {avg:>5}   "
              f"({(time.time() - t0) / n:.2f} s per cube)")
        total += n
        solved_total += ok
    return solved_total / max(total, 1)


# =============================================================================================
# 4. THE NETWORK (PyTorch): shared body, value head + policy head
# =============================================================================================
VALUE_SCALE = 10.0      # the value head predicts (moves to go) / VALUE_SCALE, which trains more smoothly

if torch is not None:

    class ResBlock(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
                                      nn.Linear(h, h), nn.BatchNorm1d(h))

        def forward(self, x):
            return F.relu(x + self.body(x))

    class CubeNet(nn.Module):
        """6*N*N stickers x 6 colours (one-hot) -> fully connected body with residual blocks -> 2 heads."""

        def __init__(self, cube, hidden=1024, blocks=3):
            super().__init__()
            self.n_stickers = cube.n_stickers
            self.stem = nn.Sequential(nn.Linear(cube.n_stickers * 6, 2 * hidden), nn.BatchNorm1d(2 * hidden), nn.ReLU(),
                                      nn.Linear(2 * hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
            self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(blocks)])
            self.value = nn.Linear(hidden, 1)               # "Distance Guesser"
            self.policy = nn.Linear(hidden, cube.n_moves)   # "Move Advisor"

        def forward(self, states):
            x = F.one_hot(states.long(), 6).flatten(1).float()
            h = self.blocks(self.stem(x))
            return self.value(h).squeeze(1), self.policy(h)

    def pick_device(name):
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def make_predictor(model, device, chunk=None):
        """Create numpy and tensor predictors; CUDA search can keep expansions on-device."""
        model.eval()
        use_amp = device.type == "cuda"
        chunk = chunk or max(1024, (1 << 22) // model.n_stickers)

        @torch.no_grad()
        def predict_torch(states):
            values, logps = [], []
            for i in range(0, len(states), chunk):
                x = states[i:i + chunk].to(device)
                ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
                with ctx:
                    v, logits = model(x)
                values.append(v.float() * VALUE_SCALE)
                logps.append(F.log_softmax(logits.float(), dim=1))
            return torch.cat(values), torch.cat(logps)

        @torch.no_grad()
        def predict(states):
            tensor = torch.from_numpy(np.ascontiguousarray(states)).to(device)
            value, logp = predict_torch(tensor)
            return value.cpu().numpy(), logp.cpu().numpy()

        predict.torch_predict = predict_torch
        predict.device = device
        return predict


    def save_model(path, model, cube, hidden, blocks, max_depth=None):
        torch.save({
            "state_dict": model.state_dict(),
            "size": cube.size,
            "hidden": hidden,
            "blocks": blocks,
            "max_depth": int(max_depth if max_depth is not None else getattr(cube, "training_max_depth", 26)),
        }, path)

    def load_model(path, device):
        """Return (model, cube). Legacy checkpoints without size are treated as 3x3x3."""
        ck = torch.load(path, map_location=device)
        cube = CubeEngine(ck.get("size", 3))
        cube.training_max_depth = int(ck.get("max_depth", max(26, 8 * cube.size)))
        model = CubeNet(cube, ck["hidden"], ck["blocks"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model, cube


    def default_training_depth(size):
        """Increase training scramble depth with cube size instead of reusing only 3x3 defaults."""
        return max(26, 8 * int(size))


    def default_hidden_size(size):
        """Scale the fully-connected network down as sticker input size grows."""
        return max(256, min(1024, int(round((1024 * 3.0 / int(size)) / 64.0) * 64)))


    def default_eval_depths(max_depth):
        return sorted({max(1, int(round(max_depth * fraction))) for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)})

    # -----------------------------------------------------------------------------------------
    # 5. TRAINING
    # -----------------------------------------------------------------------------------------
    def train(args):
        if args.iters < 1 or args.batch < 1 or args.blocks < 0:
            raise ValueError("iters and batch must be positive; blocks must be non-negative")
        if args.log_every < 1 or args.eval_every < 1 or args.eval_beam < 1:
            raise ValueError("log_every, eval_every, and eval_beam must be positive")
        torch.manual_seed(args.seed)
        device = pick_device(args.device)
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if args.resume:
            model, cube = load_model(args.resume, device)
            args.hidden = model.stem[0].out_features // 2
            args.blocks = len(model.blocks)
            if args.size is not None and args.size != cube.size:
                raise ValueError(f"{args.resume} is a {cube.size}x{cube.size}x{cube.size} model but --size {args.size} was given")
            if args.max_depth is None:
                args.max_depth = cube.training_max_depth
        else:
            size = 3 if args.size is None else args.size
            cube = _make_cube(size)
            args.hidden = args.hidden if args.hidden is not None else default_hidden_size(size)
            args.max_depth = args.max_depth if args.max_depth is not None else default_training_depth(size)
            if args.hidden < 1:
                raise ValueError("hidden must be positive")
            model = CubeNet(cube, args.hidden, args.blocks).to(device)
        if args.max_depth is None:
            args.max_depth = default_training_depth(cube.size)
        if args.max_depth < 1:
            raise ValueError("max_depth must be positive")
        cube.training_max_depth = int(args.max_depth)

        params = sum(p.numel() for p in model.parameters())
        print(f"device {device}, {cube.size}x{cube.size}x{cube.size} cube ({cube.n_moves} moves), "
              f"{params / 1e6:.1f}M parameters, batch {args.batch}, scramble depth 0..{args.max_depth}")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=args.lr * 0.01)
        batches = BatchGenerator(cube, args.batch, args.max_depth, seed=args.seed)
        eval_rng = np.random.default_rng(args.seed + 1)
        depths_to_eval = default_eval_depths(args.max_depth)

        t0 = time.time()
        sum_v = sum_p = sum_acc = None
        try:
            for it in range(1, args.iters + 1):
                model.train()
                states, depth, last = batches.get()
                x = torch.from_numpy(states).to(device)
                target_v = torch.from_numpy(depth).float().to(device) / VALUE_SCALE
                policy_mask = depth > 0
                mask = torch.from_numpy(policy_mask).to(device)
                target_p = torch.from_numpy(cube.inv[last]).to(device)

                value, logits = model(x)
                loss_v = F.mse_loss(value, target_v)
                if policy_mask.any():
                    loss_p = F.cross_entropy(logits[mask], target_p[mask])
                    acc = (logits[mask].argmax(1) == target_p[mask]).float().mean()
                else:
                    loss_p = logits.sum() * 0.0
                    acc = logits.new_zeros(())
                loss = loss_v + args.policy_weight * loss_p

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()

                detached = (loss_v.detach(), loss_p.detach(), acc.detach())
                if sum_v is None:
                    sum_v, sum_p, sum_acc = detached
                else:
                    sum_v = sum_v + detached[0]
                    sum_p = sum_p + detached[1]
                    sum_acc = sum_acc + detached[2]

                if it % args.log_every == 0 or it == args.iters:
                    k = it % args.log_every or args.log_every
                    v_avg, p_avg, acc_avg = torch.stack((sum_v, sum_p, sum_acc)).cpu().tolist()
                    print(f"iter {it:6d}/{args.iters}  distance error {np.sqrt(v_avg / k) * VALUE_SCALE:5.2f} moves (rms)  "
                          f"policy loss {p_avg / k:.3f}  policy acc {acc_avg / k:.2f}  "
                          f"{it / (time.time() - t0):.1f} it/s")
                    sum_v = sum_p = sum_acc = None

                if it % args.eval_every == 0 or it == args.iters:
                    save_model(args.save, model, cube, args.hidden, args.blocks, args.max_depth)
                    print(f"-- saved {args.save}; test solves with beam {args.eval_beam}: --")
                    evaluate(cube, make_predictor(model, device), depths_to_eval, 10, args.eval_beam,
                             eval_rng, default_search_steps(cube))
        finally:
            batches.close()
        print(f"done in {(time.time() - t0) / 60:.1f} min, model saved to {args.save}")


# =============================================================================================
# 6. SELF TEST (no PyTorch needed)
# =============================================================================================
def selftest(sizes):
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    # Number of different cubes reachable in exactly k quarter turns: well known sequences for 2x2x2 and 3x3x3
    known = {2: [1, 6, 27, 120, 534, 2256], 3: [1, 12, 114, 1068, 10011, 93840]}

    for n in sizes:
        cube = CubeEngine(n)
        solved, apply, parse, ident = cube.solved, cube.apply_moves, cube.parse_moves, np.arange(cube.n_stickers)

        def rows(state, face):                      # the stickers of one face as a string of colour letters
            return "".join(FACES[c] for c in state[face * n * n:(face + 1) * n * n])

        print(f"{n}x{n}x{n} cube simulator, {cube.n_moves} moves (clockwise ones: {' '.join(cube.move_names[::2])}):")
        good = True
        for m in range(cube.n_moves):
            p, q = cube.perms[m], cube.perms[cube.inv[m]]
            good &= (np.array_equal(np.sort(p), ident) and not np.array_equal(p[p], ident)
                     and np.array_equal(p[p][p[p]], ident) and np.array_equal(p[q], ident))
        check("every move is a permutation of order 4 and its inverse undoes it", good)
        check("a layer turn moves the 4N stickers around the layer, plus the face itself for an outer layer",
              all(int((cube._layer_perm(f, d) != ident).sum()) == 4 * n + (n * n - n % 2 if d in (0, n - 1) else 0)
                  for f in range(6) for d in range(n)))
        good = True
        for f in range(3):                          # all N layers of an axis, turned the same way = the whole cube rotated
            st = solved
            for d in range(n):
                st = st[cube._layer_perm(f, d)]
            good &= all(len(set(rows(st, face))) == 1 for face in range(6))
        check("turning all N layers of an axis rotates the whole cube (every face stays one colour)", good)

        # The sticker moves are written out by hand from how a real cube behaves.
        st_u = apply(solved, parse("U"))            # U: F's top row goes to L, L's to B, B's to R, R's to F
        check("U turns clockwise (F top row -> L, L -> B, B -> R, R -> F)",
              rows(st_u, 1)[:n] == "B" * n and rows(st_u, 2)[:n] == "R" * n
              and rows(st_u, 4)[:n] == "F" * n and rows(st_u, 5)[:n] == "L" * n)
        st_f = apply(solved, parse("F"))            # F: U bottom row -> R left col -> D top row -> L right col -> U
        check("F turns clockwise (L -> U bottom row, U -> R left column, R -> D top row, D -> L right column)",
              rows(st_f, 0)[-n:] == "L" * n and rows(st_f, 1)[0::n] == "U" * n
              and rows(st_f, 3)[:n] == "R" * n and rows(st_f, 4)[n - 1::n] == "D" * n)
        st_r = apply(solved, parse("R"))            # R: F right col -> U right col -> B left col -> D right col -> F
        check("R turns clockwise (F -> U right column, U -> B left column, B -> D right column, D -> F right column)",
              rows(st_r, 0)[n - 1::n] == "F" * n and rows(st_r, 5)[0::n] == "U" * n
              and rows(st_r, 3)[n - 1::n] == "B" * n and rows(st_r, 2)[n - 1::n] == "D" * n)
        if n >= 4:
            st = apply(solved, parse("2U"))         # a slice: only the 2nd row of the four side faces moves
            rest = n * n - 2 * n
            check("2U turns only the 2nd row of the side faces, clockwise seen from U",
                  rows(st, 0) == "U" * n * n and rows(st, 3) == "D" * n * n
                  and rows(st, 2) == "F" * n + "R" * n + "F" * rest and rows(st, 1) == "R" * n + "B" * n + "R" * rest
                  and rows(st, 5) == "B" * n + "L" * n + "B" * rest and rows(st, 4) == "L" * n + "F" * n + "L" * rest)
        st, k = apply(solved, parse("D")), n * (n - 1)
        if n % 2:                                   # D is a move of its own: F's bottom row goes to R, R's to B, B's to L, L's to F
            check("D turns clockwise (F bottom row -> R, R -> B, B -> L, L -> F)",
                  rows(st, 2)[-n:] == "L" * n and rows(st, 1)[-n:] == "F" * n
                  and rows(st, 5)[-n:] == "R" * n and rows(st, 4)[-n:] == "B" * n)
        else:                                       # D is no move: all the other layers turn the other way instead
            check("D (not a move of its own) = all other layers turned the opposite way",
                  rows(st, 2) == "R" * k + "F" * n and rows(st, 1) == "B" * k + "R" * n
                  and rows(st, 5) == "L" * k + "B" * n and rows(st, 4) == "F" * k + "L" * n
                  and rows(st, 0) == "U" * n * n and rows(st, 3) == "D" * n * n)
        if n >= 3:
            other = " ".join(("" if j == 1 else str(j)) + "L" for j in range(1, n - 1))
            check(f"Rw = '{other}' (the remaining layers turned the other way)",
                  np.array_equal(apply(solved, parse("Rw")), apply(solved, parse(other))))
        check("every face turn (U D R L F B, with ' and 2) is understood and undone by its inverse",
              all(cube.is_solved(apply(solved, parse(f"{x} {x}'"))) and cube.is_solved(apply(solved, parse(f"{x}2 {x}2")))
                  for x in FACES))
        invalid_suffixes_rejected = True
        for bad_move in ("R1", "R3", "2R3", "3R1"):
            try:
                parse(bad_move)
                invalid_suffixes_rejected = False
            except ValueError:
                pass
        check("layer numbers belong before the face; unsupported R1/R3 suffixes are rejected",
              invalid_suffixes_rejected)
        if n == 3:
            check("moves are numbered as in the 3x3x3-only script (R U R' U2 = 2 0 3 0 0)",
                  parse("R U R' U2") == [2, 0, 3, 0, 0] and cube.format_moves(parse("R U R' U2 F'")) == "R U R' U2 F'")
        sexy = parse("R U R' U'")
        check("(R U R' U') has order 6",
              cube.is_solved(apply(solved, sexy * 6)) and not any(cube.is_solved(apply(solved, sexy * j)) for j in (1, 2, 3)))
        scr = cube.random_walks(5, 20, np.random.default_rng(1))[0]
        check("a cube written with any six colour letters (here WRGYOB) is recognised, solved or scrambled",
              all(np.array_equal(cube.state_from_facelets(cube.facelets(s, "WRGYOB")), s) for s in [solved, *scr]))
        st1, _, last1 = cube.random_walks(30, 1, np.random.default_rng(2))
        check("a 1-move random scramble is exactly the move it records",
              all(np.array_equal(st1[i], apply(solved, [last1[i]])) for i in range(30)))

        if n in known:
            expected = known[n]
        else:                                       # 6a moves; same-axis moves commute, X X = X' X'
            a = n - 1
            expected = [1, cube.n_moves,
                        cube.n_moves * (cube.n_moves - 1) - 3 * (math.comb(2 * a, 2) - a) - 3 * a]
        frontier, seen, got = solved[None, :], set(cube.hash_states(solved[None, :]).tolist()), [1]
        for _ in range(len(expected) - 1):
            kids = frontier[:, cube.perms].reshape(-1, cube.n_stickers)
            h = cube.hash_states(kids)
            _, idx = np.unique(h, return_index=True)
            kids, h = kids[idx], h[idx]
            fresh = ~np.isin(h, np.fromiter(seen, dtype=np.int64, count=len(seen)))
            frontier = kids[fresh]
            seen.update(h[fresh].tolist())
            got.append(len(frontier))
        check(f"cubes at distance 0..{len(expected) - 1} (quarter-turn metric) = {got} (expected {expected})", got == expected)

        # search (no network): beam search guided by 'how many stickers are in the wrong place'
        def misplaced(states):
            return (states != solved).sum(axis=1).astype(np.float32), np.zeros((len(states), cube.n_moves), np.float32)

        depth = 4 if n <= 3 else 3
        starts, _, _ = cube.random_walks(10, depth, np.random.default_rng(0))
        found = 0
        for s in starts:
            path = beam_search(cube, misplaced, s, beam_width=4000, max_steps=12)
            found += path is not None and cube.is_solved(apply(s, path))
        check(f"search without a network: solved {found}/10 {depth}-move scrambles with a crude hand-made heuristic", found >= 9)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


# =============================================================================================
# 7. COMMAND LINE
# =============================================================================================
def _need_torch():
    if torch is None:
        sys.exit("PyTorch is required for this command:  pip install torch")


def cube_size_arg(value):
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("cube size must be an integer >= 2") from None
    if size < 2:
        raise argparse.ArgumentTypeError("cube size must be at least 2")
    return size


def nonnegative_int_arg(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from None
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return number


def _start_state(args, cube):
    rng = np.random.default_rng(args.seed)
    if args.cube:
        return cube.state_from_facelets(args.cube)
    if args.scramble:
        return cube.apply_moves(cube.solved, cube.parse_moves(args.scramble))
    count = args.random
    if count < 0:
        raise ValueError("--random must be non-negative")
    # Generate the explicit random sequence as well as the state so the CLI can display it.
    moves, prev1, prev2 = [], cube.n_moves, cube.n_moves
    for _ in range(count):
        candidates = np.arange(cube.n_moves)
        allowed = candidates[(candidates != cube.inv[prev1]) & ~((candidates == prev1) & (prev1 == prev2))]
        move = int(rng.choice(allowed))
        moves.append(move)
        prev2, prev1 = prev1, move
    return cube.apply_moves(cube.solved, moves), cube.format_moves(moves)


def main():
    ap = argparse.ArgumentParser(description="Neural network NxNxN Rubik's Cube solver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    size_help = "cube size N: 3 = 3x3x3, 4 = 4x4x4, 8 = 8x8x8, ..."

    t = sub.add_parser("train", help="train a new network (or continue with --resume)")
    t.add_argument("--size", type=cube_size_arg, default=None, help=size_help + " (default 3, or the size of the --resume model)")
    t.add_argument("--iters", type=int, default=20000, help="training steps")
    t.add_argument("--batch", type=int, default=4096, help="cubes per step")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--max_depth", type=int, default=None, help="longest random scramble; defaults scale with cube size")
    t.add_argument("--hidden", type=int, default=None, help="network width; defaults scale down for larger cubes")
    t.add_argument("--blocks", type=int, default=3, help="residual blocks (use 1-2 on a CPU)")
    t.add_argument("--policy_weight", type=float, default=0.1)
    t.add_argument("--save", default="cube_model.pt")
    t.add_argument("--resume", default=None, help="checkpoint to continue training from")
    t.add_argument("--device", default="auto")
    t.add_argument("--eval_every", type=int, default=2000)
    t.add_argument("--eval_beam", type=int, default=256)
    t.add_argument("--log_every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("solve", help="solve one cube with a trained network")
    s.add_argument("--model", required=True)
    s.add_argument("--size", type=cube_size_arg, default=None, help="cube size; taken from the model, so only a safety check here")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--scramble", help="moves to apply to a solved cube, e.g. \"R U R' U' F2\" (also 2R, Rw, 3Rw' ...)")
    g.add_argument("--cube", help="6*N*N sticker letters in U R F D L B order (see the top of this file)")
    g.add_argument("--random", type=nonnegative_int_arg, metavar="N", help="scramble with N random moves")
    s.add_argument("--beam", type=int, default=2048, help="beam width: wider = slower but solves more")
    s.add_argument("--max_steps", type=int, default=None, help="search horizon; defaults scale with cube size/checkpoint")
    s.add_argument("--policy_weight", type=float, default=0.0)
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=None)

    e = sub.add_parser("eval", help="solve rate for several scramble depths")
    e.add_argument("--model", required=True)
    e.add_argument("--size", type=cube_size_arg, default=None, help="cube size; taken from the model, so only a safety check here")
    e.add_argument("--depths", default=None, help="comma-separated scramble depths; defaults scale with checkpoint")
    e.add_argument("--n", type=int, default=20, help="cubes per depth")
    e.add_argument("--beam", type=int, default=1024)
    e.add_argument("--max_steps", type=int, default=None, help="search horizon; defaults scale with cube size/checkpoint")
    e.add_argument("--policy_weight", type=float, default=0.0)
    e.add_argument("--device", default="auto")
    e.add_argument("--seed", type=int, default=0)

    st = sub.add_parser("selftest", help="check the cube simulator and the search (no PyTorch needed)")
    st.add_argument("--size", type=cube_size_arg, default=None, help="test only this size (default: 2 3 4 5 8)")
    args = ap.parse_args()

    if args.cmd == "selftest":
        sizes = [_make_cube(args.size).size] if args.size is not None else [2, 3, 4, 5, 8]
        sys.exit(0 if selftest(sizes) else 1)
    _need_torch()
    if args.cmd == "train":
        train(args)
        return

    device = pick_device(args.device)
    model, cube = load_model(args.model, device)
    if args.size not in (None, cube.size):
        sys.exit(f"{args.model} is a {cube.size}x{cube.size}x{cube.size} model but --size {args.size} was given")
    predict = make_predictor(model, device)

    if args.cmd == "eval":
        depths = ([int(x) for x in args.depths.split(",")] if args.depths
                  else default_eval_depths(getattr(cube, "training_max_depth", default_training_depth(cube.size))))
        evaluate(cube, predict, depths, args.n, args.beam, np.random.default_rng(args.seed),
                 args.max_steps, args.policy_weight)
        return

    if args.random is not None:
        start, scramble_text = _start_state(args, cube)
        print(f"random scramble: {scramble_text or '(already solved)'}")
    else:
        start = _start_state(args, cube)
    print(f"cube to solve ({cube.size}x{cube.size}x{cube.size}):\n" + cube.render(start))
    est = predict(start[None, :])[0][0]
    print(f"\nnetwork's guess: about {est:.1f} moves from solved")
    t0 = time.time()
    path = beam_search(cube, predict, start, args.beam, args.max_steps, args.policy_weight)
    dt = time.time() - t0
    if path is None:
        print(f"no solution found in {dt:.1f} s - try a wider --beam, more --max_steps, or train longer. "
              f"(Also check that the cube is a legal one: a cube with a twisted corner or a swapped "
              f"pair of pieces cannot be solved.)")
    else:
        assert cube.is_solved(cube.apply_moves(start, path))
        print(f"solved in {len(path)} quarter turns ({dt:.1f} s, beam {args.beam}):")
        print(cube.format_moves(path) if path else "(already solved)")


if __name__ == "__main__":
    main()
