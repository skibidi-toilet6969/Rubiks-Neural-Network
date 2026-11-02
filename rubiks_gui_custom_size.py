#!/usr/bin/env python3
"""
Dynamic-size Rubik's Cube GUI for rubiks_ai_custom_size.py.

The selected .pt checkpoint determines the cube size. Checkpoints produced by
rubiks_ai_custom_size.py store that size in checkpoint["size"]; legacy
checkpoints without the field are treated as 3x3x3.

Run:
    python rubiks_gui_custom_size.py
    python rubiks_gui_custom_size.py --model cube5.pt
"""

import argparse
import itertools
import os
import re
import socket
import threading
import queue
from collections import deque
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import rubiks_ai_custom_size as rubiks_ai


COLOR_MAP = {
    0: (240, 240, 240),  # U
    1: (200, 30, 30),    # R
    2: (30, 180, 30),    # F
    3: (240, 200, 30),   # D
    4: (30, 90, 220),    # L
    5: (255, 130, 20),   # B
}

BODY_COLOR = (10, 10, 10)
BACKGROUND = (25, 25, 25)


def rgb_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


CAMERA_DISTANCE_FACTOR = 3.0
QUARTER_TURN_SECONDS = 0.30
HALF_TURN_FACTOR = 1.5
FAST_FORWARD_BACKLOG = 4
STICKER_HALF = 0.44
STICKER_OFFSET = 0.525


def rotation_matrix(axis, angle_rad):
    ax, ay, az = axis

    K = np.array([
        [0.0, -az, ay],
        [az, 0.0, -ax],
        [-ay, ax, 0.0]
    ])

    return (
        np.eye(3)
        + np.sin(angle_rad) * K
        + (1.0 - np.cos(angle_rad)) * (K @ K)
    )


def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)


class CubeGeometry:
    """Build render geometry for any NxNxN cube."""

    def __init__(self, cube):
        n = cube.size
        half = (n - 1) / 2.0

        signs = [
            (-1, -1),
            (1, -1),
            (1, 1),
            (-1, 1)
        ]

        body_corners = []
        body_centers = []
        body_normals = []
        body_cubies = []
        body_neighbors = []
        body_has_neighbor = []

        # Build the six body faces for every cubie.
        for ix, iy, iz in itertools.product(range(n), repeat=3):

            coord = np.array(
                [ix - half, iy - half, iz - half],
                dtype=float
            )

            for axis_idx in range(3):

                a, b = [
                    v for v in range(3)
                    if v != axis_idx
                ]

                for sign in (-1.0, 1.0):

                    normal = np.zeros(3)
                    normal[axis_idx] = sign

                    center = coord + normal * 0.5

                    ea = np.zeros(3)
                    eb = np.zeros(3)

                    ea[a] = 1.0
                    eb[b] = 1.0

                    quad = np.array([
                        center
                        + 0.5 * sa * ea
                        + 0.5 * sb * eb
                        for sa, sb in signs
                    ])

                    neighbor = coord + normal

                    body_corners.append(quad)
                    body_centers.append(center)
                    body_normals.append(normal)
                    body_cubies.append(coord)
                    body_neighbors.append(neighbor)

                    body_has_neighbor.append(
                        bool(np.all(np.abs(neighbor) <= half))
                    )

        # Stickers follow the AI engine's facelet order.
        sticker_corners = []
        sticker_centers = []
        sticker_cubies = []

        for face_idx, face_name in enumerate(rubiks_ai.FACES):

            normal = np.array(
                rubiks_ai._NORMAL[face_name],
                dtype=float
            )

            col_dir, row_dir = (
                np.array(v, dtype=float)
                for v in rubiks_ai._READING[face_name]
            )

            for row in range(n):
                for col in range(n):

                    # Place stickers just outside the cubie surface.
                    sticker_center = (
                        normal * (half + STICKER_OFFSET)
                        + (col - half) * col_dir
                        + (row - half) * row_dir
                    )

                    quad = np.array([
                        sticker_center
                        + dc * STICKER_HALF * col_dir
                        + dr * STICKER_HALF * row_dir
                        for dr, dc in signs
                    ])

                    sticker_corners.append(quad)
                    sticker_centers.append(sticker_center)

                    # Store the cubie coordinate for animation selection.
                    sticker_cubies.append(
                        sticker_center - normal * STICKER_OFFSET
                    )

        self.body_corners = np.asarray(body_corners)
        self.body_centers = np.asarray(body_centers)
        self.body_normals = np.asarray(body_normals)
        self.body_cubies = np.asarray(body_cubies)
        self.body_neighbors = np.asarray(body_neighbors)
        self.body_has_neighbor = np.asarray(body_has_neighbor)

        self.sticker_corners = np.asarray(sticker_corners)
        self.sticker_centers = np.asarray(sticker_centers)
        self.sticker_cubies = np.asarray(sticker_cubies)
        self.sticker_normals = cube.sticker_normals.astype(float, copy=True)


class Rubiks3DApp:

    def __init__(self, model_path=None, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.lock = threading.RLock()
        self.model_lock = threading.Lock()

        self.rot_x = 25.0
        self.rot_y = -35.0
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)

        self.anim_queue = deque()
        self.current_anim = None
        self.loaded_model = None
        self.predict = None
        self.device = None
        self.loaded_model_path = None
        self._model_load_busy = False
        self._solve_busy = False
        self._closing = False
        self._render_dirty = True
        self._polygon_items = []
        self._label_item = None

        self._ui_queue = queue.Queue()
        self._socket_stop = threading.Event()
        self._socket_server = None
        self._server_ready = False

        self.root = tk.Tk()
        self._setup_tkinter_ui()
        self._last_frame_time = time.perf_counter()

        self.cube = rubiks_ai.CubeEngine(3)
        self._rebuild_cube(self.cube)

        if model_path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, model_path)

        self.server_thread = threading.Thread(
            target=self._start_socket_server, name="cube-tcp-listener", daemon=True
        )
        self.server_thread.start()
        self.root.after(20, self._process_ui_queue)

        if model_path:
            self.root.after(150, self.load_model_ui)

    def _rebuild_cube(self, cube):
        """Switch engine, state, and 3D geometry together."""
        with self.lock:
            self.cube = cube
            self.cube_state = cube.solved.copy()
            self.display_state = cube.solved.copy()
            self.anim_queue.clear()
            self.current_anim = None
            self.geo = CubeGeometry(cube)
            self.face_axis = {f: cube.face_normals[f] for f in rubiks_ai.FACES}
            # CubeEngine's clockwise permutation is a -90 degree rotation about its outward normal.
            self.face_sign = {f: -1.0 for f in rubiks_ai.FACES}
            self._render_dirty = True

        if hasattr(self, "size_status"):
            self._ui_status(
                self.size_status,
                f"Detected model size: {cube.size}×{cube.size}×{cube.size}",
                "#ffc107",
            )

    def _setup_tkinter_ui(self):

        self.root.title(
            "Rubik's AI Control Panel — dynamic cube size"
        )

        self.root.geometry("1180x760")
        self.root.minsize(900, 620)
        self.root.configure(bg=rgb_hex(BACKGROUND))

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close
        )

        # Left control panel.
        control_panel = tk.Frame(
            self.root,
            bg="#1e1e1e",
            width=470
        )

        control_panel.grid(
            row=0,
            column=0,
            sticky="nsw"
        )

        control_panel.grid_propagate(False)
        control_panel.columnconfigure(0, weight=1)

        # Right cube viewport.
        viewport = tk.Frame(
            self.root,
            bg=rgb_hex(BACKGROUND)
        )

        viewport.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(0, 10),
            pady=10
        )

        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            viewport,
            bg=rgb_hex(BACKGROUND),
            highlightthickness=0,
            borderwidth=0
        )

        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.canvas.bind(
            "<ButtonPress-1>",
            self._on_canvas_press
        )

        self.canvas.bind(
            "<ButtonRelease-1>",
            self._on_canvas_release
        )

        self.canvas.bind(
            "<B1-Motion>",
            self._on_canvas_drag
        )
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Scramble controls.
        frame_top = tk.LabelFrame(
            control_panel,
            text=" Scramble Controls ",
            fg="white",
            bg="#2d2d2d",
            font=("Arial", 11, "bold")
        )

        frame_top.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=10,
            pady=8
        )

        frame_top.columnconfigure(0, weight=1)

        tk.Label(
            frame_top,
            text="Enter scramble moves (e.g. R U R' U'):",
            fg="#ccc",
            bg="#2d2d2d"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=5
        )

        self.scramble_entry = tk.Entry(
            frame_top,
            bg="#1e1e1e",
            fg="white",
            insertbackground="white",
            font=("Consolas", 11)
        )

        self.scramble_entry.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=10,
            pady=5
        )

        self.scramble_entry.insert(
            0,
            "R U R' U' F2 D L'"
        )

        tk.Button(
            frame_top,
            text="Apply Scramble",
            bg="#007acc",
            fg="white",
            command=self.apply_scramble_ui
        ).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=10,
            pady=5
        )

        self.scramble_status = tk.Label(
            frame_top,
            text="Status: Ready",
            fg="#aaa",
            bg="#2d2d2d",
            anchor="w",
            justify="left"
        )

        self.scramble_status.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=10,
            pady=5
        )

        # AI model controls.
        frame_ai = tk.LabelFrame(
            control_panel,
            text=" AI Solver / Model ",
            fg="white",
            bg="#2d2d2d",
            font=("Arial", 11, "bold")
        )

        frame_ai.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=10,
            pady=8
        )

        frame_ai.columnconfigure(0, weight=1)

        tk.Label(
            frame_ai,
            text="PyTorch checkpoint (.pt):",
            fg="#ccc",
            bg="#2d2d2d"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=10,
            pady=(6, 2)
        )

        path_row = tk.Frame(
            frame_ai,
            bg="#2d2d2d"
        )

        path_row.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=10
        )

        path_row.columnconfigure(0, weight=1)

        self.model_entry = tk.Entry(
            path_row,
            bg="#1e1e1e",
            fg="white",
            insertbackground="white",
            font=("Consolas", 10)
        )

        self.model_entry.grid(
            row=0,
            column=0,
            sticky="ew"
        )

        tk.Button(
            path_row,
            text="Browse…",
            command=self.browse_model
        ).grid(
            row=0,
            column=1,
            padx=(5, 0)
        )

        tk.Button(
            frame_ai,
            text="Load Model / Detect Size",
            bg="#007acc",
            fg="white",
            command=self.load_model_ui
        ).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=10,
            pady=6
        )

        self.size_status = tk.Label(
            frame_ai,
            text="Cube size: 3×3×3 (default; no model loaded)",
            fg="#ffc107",
            bg="#2d2d2d",
            anchor="w"
        )

        self.size_status.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=10,
            pady=2
        )

        tk.Label(
            frame_ai,
            text="Beam width (higher = broader search):",
            fg="#ccc",
            bg="#2d2d2d"
        ).grid(
            row=4,
            column=0,
            sticky="w",
            padx=10,
            pady=(5, 2)
        )

        self.beam_entry = tk.Entry(
            frame_ai,
            bg="#1e1e1e",
            fg="white",
            insertbackground="white",
            font=("Consolas", 10)
        )

        self.beam_entry.grid(
            row=5,
            column=0,
            sticky="ew",
            padx=10
        )

        self.beam_entry.insert(0, "2048")

        self.solve_button = tk.Button(
            frame_ai,
            text="Run AI Solve",
            bg="#28a745",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.run_ai_solve_ui
        )
        self.solve_button.grid(
            row=6,
            column=0,
            sticky="ew",
            padx=10,
            pady=7
        )

        self.solve_status = tk.Label(
            frame_ai,
            text=f"TCP listener: starting on {self.host}:{self.port}…",
            fg="#aaa",
            bg="#2d2d2d",
            anchor="nw",
            justify="left",
            wraplength=430
        )

        self.solve_status.grid(
            row=7,
            column=0,
            sticky="ew",
            padx=10,
            pady=5
        )

    def _ui_status(self, widget, text, color="#aaa"):
        """Queue UI changes for the main thread; worker threads never call Tkinter."""
        if not self._closing:
            self._ui_queue.put(("status", widget, text, color))

    def _ui_callback(self, callback, *args):
        if not self._closing:
            self._ui_queue.put(("callback", callback, args))

    def _process_ui_queue(self):
        if self._closing:
            return
        try:
            while True:
                item = self._ui_queue.get_nowait()
                try:
                    if item[0] == "status":
                        _, widget, text, color = item
                        widget.config(text=text, fg=color)
                    else:
                        _, callback, args = item
                        callback(*args)
                except tk.TclError:
                    # The window may be closing while a worker finishes.
                    pass
        except queue.Empty:
            pass
        if not self._closing:
            self.root.after(20, self._process_ui_queue)

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Rubik's AI checkpoint",
            filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.model_entry.delete(0, tk.END)
            self.model_entry.insert(0, path)
            self.load_model_ui()

    def load_model_ui(self):
        if self._solve_busy:
            self._ui_status(self.solve_status, "Wait for the current solve to finish before loading another model.", "#ffc107")
            return
        path = self.model_entry.get().strip()
        if not path:
            messagebox.showerror("Model", "Select a .pt checkpoint first.")
            return
        if self._model_load_busy:
            return
        self._model_load_busy = True
        self._ui_status(self.solve_status, "Loading checkpoint and detecting cube size…", "#ffc107")
        threading.Thread(
            target=self._load_model_worker, args=(path,), name="cube-model-loader", daemon=True
        ).start()

    def _load_model_cached(self, path):
        """Load once per path; retain the active cube state when only the model changes."""
        full_path = os.path.abspath(os.path.expanduser(path))
        with self.model_lock:
            if self.loaded_model is not None and self.loaded_model_path == full_path:
                return self.loaded_model, self.cube, self.predict, self.device, False

            device = rubiks_ai.pick_device("auto")
            model, model_cube = rubiks_ai.load_model(full_path, device)
            predictor = rubiks_ai.make_predictor(model, device)
            with self.lock:
                resized = model_cube.size != self.cube.size
            if resized:
                self._rebuild_cube(model_cube)
            else:
                with self.lock:
                    self.cube.training_max_depth = getattr(
                        model_cube, "training_max_depth", max(26, 8 * self.cube.size)
                    )
            self.loaded_model = model
            self.predict = predictor
            self.device = device
            self.loaded_model_path = full_path
            return model, self.cube, predictor, device, resized

    def _load_model_worker(self, path):
        try:
            model, cube, _predictor, device, resized = self._load_model_cached(path)
            note = " Cube reset because the checkpoint uses a different size." if resized else " Current scramble preserved."
            self._ui_status(
                self.solve_status,
                f"Loaded {os.path.basename(path)} on {device}; detected "
                f"{cube.size}×{cube.size}×{cube.size}.{note}",
                "#28a745",
            )
        except Exception as exc:
            self._ui_status(self.solve_status, f"Model load failed: {exc}", "#dc3545")
        finally:
            self._ui_callback(self._finish_model_load)

    def _finish_model_load(self):
        self._model_load_busy = False

    def _jump_to_state(self, state):
        state = np.asarray(state, dtype=np.uint8)
        with self.lock:
            if state.shape != (self.cube.n_stickers,):
                raise ValueError(
                    f"State has {len(state)} stickers; current {self.cube.size}x{self.cube.size}x"
                    f"{self.cube.size} cube needs {self.cube.n_stickers}."
                )
            self.anim_queue.clear()
            self.current_anim = None
            self.cube_state = state.copy()
            self.display_state = state.copy()
            self._render_dirty = True

    def queue_moves_animation(self, moves):
        """Queue canonical single-engine-layer turns, including rewritten composite turns."""
        with self.lock:
            for move in moves:
                parsed = self.cube.parse_moves(move)
                self.cube_state = self.cube.apply_moves(self.cube_state, parsed)
                canonical = self.cube.format_moves(parsed)
                self.anim_queue.extend(canonical.split())
            if self.anim_queue:
                self._render_dirty = True

    def apply_scramble_ui(self):
        if self._solve_busy:
            self._ui_status(self.scramble_status, "Cannot change the cube while AI search is running.", "#ffc107")
            return
        try:

            moves = self.cube.parse_moves(
                self.scramble_entry.get()
            )

            state = self.cube.apply_moves(
                self.cube.solved,
                moves
            )

            self._jump_to_state(state)

            self.scramble_status.config(
                text=(
                    f"Status: Applied ({len(moves)} quarter-turn "
                    f"engine moves) on {self.cube.size}×"
                    f"{self.cube.size}×{self.cube.size}"
                ),
                fg="#28a745"
            )

        except Exception as exc:

            messagebox.showerror(
                "Scramble error",
                str(exc)
            )

            self.scramble_status.config(
                text="Status: Scramble failed",
                fg="#dc3545"
            )

    def run_ai_solve_ui(self):
        if self._model_load_busy:
            self._ui_status(self.solve_status, "Wait for model loading to finish before solving.", "#ffc107")
            return
        if self._solve_busy:
            return
        try:
            beam = int(self.beam_entry.get())
            if beam < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Beam width", "Beam width must be a positive integer.")
            return

        path = self.model_entry.get().strip()
        if not path and self.loaded_model is None:
            messagebox.showerror("Model", "Load a .pt checkpoint before solving.")
            return
        self._solve_busy = True
        self.solve_button.config(state=tk.DISABLED)
        if not path:
            path = self.loaded_model_path
        threading.Thread(
            target=self._solve_worker, args=(path, beam), name="cube-solver", daemon=True
        ).start()

    def _finish_solve(self):
        self._solve_busy = False
        if not self._closing:
            self.solve_button.config(state=tk.NORMAL)

    def _solve_worker(self, model_path, beam_width):
        try:
            _model, model_cube, predictor, _device, resized = self._load_model_cached(model_path)
            if resized or model_cube.size != self.cube.size:
                self._ui_status(
                    self.solve_status,
                    f"Checkpoint is {model_cube.size}×{model_cube.size}×{model_cube.size}; "
                    "cube resized and reset to solved. Apply the scramble again.",
                    "#ffc107",
                )
                return

            with self.lock:
                active_cube = self.cube
                start_state = self.cube_state.copy()

            self._ui_status(
                self.solve_status,
                f"Searching {active_cube.size}×{active_cube.size}×{active_cube.size} "
                f"(beam={beam_width}; max steps=auto)…",
                "#ffc107",
            )
            t0 = time.perf_counter()
            path = rubiks_ai.beam_search(
                active_cube, predictor, start_state, beam_width=beam_width, max_steps=None
            )
            elapsed = time.perf_counter() - t0

            if path is None:
                self._ui_status(
                    self.solve_status,
                    "No solution found within the size-aware search horizon. "
                    "Try a wider beam, more training depth, or a shorter scramble.",
                    "#dc3545",
                )
                return

            formatted = active_cube.format_moves(path)
            self.queue_moves_animation(formatted.split() if formatted else [])
            self._ui_status(
                self.solve_status,
                f"Solved in {len(path)} quarter turns ({elapsed:.2f}s): "
                f"{formatted or '(already solved)'}",
                "#28a745",
            )
        except Exception as exc:
            self._ui_status(self.solve_status, f"Solve failed: {exc}", "#dc3545")
        finally:
            self._ui_callback(self._finish_solve)

    def _send_socket_response(self, conn, message):
        try:
            conn.sendall((message.rstrip() + "\n").encode("utf-8", errors="replace"))
        except OSError:
            # A client can disconnect at any point; that must not affect the listener.
            pass

    def _handle_socket_client(self, conn, address):
        with conn:
            try:
                conn.settimeout(2.0)
                chunks = []
                total = 0
                while total < 8192:
                    try:
                        part = conn.recv(8192 - total)
                    except socket.timeout:
                        if chunks:
                            break
                        raise
                    if not part:
                        break
                    chunks.append(part)
                    total += len(part)
                    if b"\n" in part:
                        break
                    # After the first packet, briefly wait for any fragmented remainder.
                    conn.settimeout(0.25)
                raw = b"".join(chunks)
                if not raw:
                    self._send_socket_response(conn, "ERROR: empty request")
                    return
                data = raw.decode("utf-8").strip()
                if not data:
                    self._send_socket_response(conn, "ERROR: empty request")
                    return

                with self.lock:
                    if self._solve_busy:
                        self._send_socket_response(conn, "ERROR: solver is busy; retry after it finishes")
                        return
                    cube = self.cube
                    moves = cube.parse_moves(data)
                    state = cube.apply_moves(cube.solved, moves)
                    self._jump_to_state(state)
                    size = cube.size
                self._send_socket_response(conn, f"OK: scramble applied to {size}x{size}x{size}")
            except UnicodeDecodeError:
                self._send_socket_response(conn, "ERROR: request must be UTF-8")
            except socket.timeout:
                self._send_socket_response(conn, "ERROR: client timed out")
            except (ConnectionResetError, BrokenPipeError):
                return
            except OSError:
                return
            except Exception as exc:
                self._send_socket_response(conn, f"ERROR: {exc}")

    def _start_socket_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
            server.listen(16)
            server.settimeout(0.5)
            self._socket_server = server
            self._server_ready = True
            self._ui_status(
                self.solve_status, f"TCP listener: active on {self.host}:{self.port}", "#28a745"
            )
            while not self._socket_stop.is_set():
                try:
                    conn, address = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._socket_stop.is_set():
                        break
                    raise
                threading.Thread(
                    target=self._handle_socket_client, args=(conn, address),
                    name="cube-tcp-client", daemon=True,
                ).start()
        except OSError as exc:
            if not self._socket_stop.is_set():
                self._ui_status(
                    self.solve_status, f"TCP listener failed ({self.host}:{self.port}): {exc}", "#dc3545"
                )
        finally:
            self._server_ready = False
            try:
                server.close()
            except OSError:
                pass
            self._socket_server = None

    def _on_canvas_press(self, event):

        self.is_dragging = True

        self.last_mouse_pos = (
            event.x,
            event.y
        )

    def _on_canvas_release(self, _event):
        self.is_dragging = False

    def _on_canvas_drag(self, event):

        if not self.is_dragging:
            return

        dx = event.x - self.last_mouse_pos[0]
        dy = event.y - self.last_mouse_pos[1]

        self.rot_y -= dx * 0.5
        self.rot_x = float(np.clip(self.rot_x - dy * 0.5, -85.0, 85.0))
        self._render_dirty = True

        self.last_mouse_pos = (
            event.x,
            event.y
        )

    def _on_canvas_configure(self, _event):
        self._render_dirty = True

    def _on_close(self):
        self._closing = True
        self._socket_stop.set()
        server = self._socket_server
        if server is not None:
            try:
                server.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                server.close()
            except OSError:
                pass
        self.root.destroy()

    def run(self):
        """Start the Tkinter render loop."""

        self._last_frame_time = time.perf_counter()

        self.root.after(
            0,
            self._render_loop
        )

        self.root.mainloop()

    def _render_loop(self):
        """Advance animation and redraw only when the visible scene changes."""
        if self._closing:
            return
        frame_started = time.perf_counter()
        dt = min(frame_started - self._last_frame_time, 0.05)
        self._last_frame_time = frame_started
        animation_changed = self._advance_animation(dt)
        if animation_changed:
            self._render_dirty = True
        if self._render_dirty:
            self._render_3d_cube(self.canvas)
            self._render_dirty = False
        elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
        delay = max(1, int(16.0 - elapsed_ms))
        self.root.after(delay, self._render_loop)

    def _advance_animation(self, dt):
        changed = False
        with self.lock:
            if self.current_anim is None:
                if not self.anim_queue:
                    return False
                move = self.anim_queue.popleft()
                match = re.fullmatch(r"(\d*)([URFDLB])([2']?)", move)
                if not match:
                    raise RuntimeError(f"non-canonical animation move in queue: {move!r}")
                layer_text, face, suffix = match.groups()
                depth = int(layer_text) - 1 if layer_text else 0
                target = 180.0 if suffix == "2" else (-90.0 if suffix == "'" else 90.0)
                self.current_anim = {
                    "face": face, "depth": depth, "move": move, "target": target,
                    "angle": 0.0, "elapsed": 0.0,
                    "duration": QUARTER_TURN_SECONDS * (HALF_TURN_FACTOR if abs(target) > 90 else 1.0)
                    * (0.5 if len(self.anim_queue) > FAST_FORWARD_BACKLOG else 1.0),
                }
                changed = True

            anim = self.current_anim
            anim["elapsed"] += dt
            t = min(anim["elapsed"] / anim["duration"], 1.0)
            anim["angle"] = anim["target"] * ease_in_out(t)
            changed = True
            if t >= 1.0:
                self.display_state = self.cube.apply_moves(
                    self.display_state, self.cube.parse_moves(anim["move"])
                )
                self.current_anim = None
        return changed

    def _view_matrix(self):

        rx, ry = np.radians(
            self.rot_x
        ), np.radians(
            self.rot_y
        )

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)

        Ry = np.array([
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy]
        ])

        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx]
        ])

        return Rx @ Ry

    def _render_3d_cube(self, canvas):
        """Render exposed cubie faces and stickers with perspective and reusable Canvas items."""
        with self.lock:
            G = self.geo
            cube_size = self.cube.size
            state = self.display_state.copy()
            anim = dict(self.current_anim) if self.current_anim else None
            face_axis = self.face_axis
            face_sign = self.face_sign

        # Start with immutable geometry; only transform the selected layer's rows.
        body_corners = G.body_corners
        body_centers = G.body_centers
        body_normals = G.body_normals
        sticker_corners = G.sticker_corners
        sticker_centers = G.sticker_centers
        sticker_normals = G.sticker_normals
        body_visible = ~G.body_has_neighbor

        if anim:
            axis = face_axis[anim["face"]]
            layer_coordinate = (cube_size - 1) / 2.0 - anim["depth"]
            body_moving = np.isclose(G.body_cubies @ axis, layer_coordinate)
            sticker_moving = np.isclose(G.sticker_cubies @ axis, layer_coordinate)
            neighbor_moving = np.isclose(G.body_neighbors @ axis, layer_coordinate)
            # Internal faces are shown only at the seam between a rotating and fixed layer.
            body_visible = body_visible | (G.body_has_neighbor & (body_moving != neighbor_moving))

            R = rotation_matrix(axis, np.radians(anim["angle"]) * face_sign[anim["face"]])
            body_corners = body_corners.copy()
            body_centers = body_centers.copy()
            body_normals = body_normals.copy()
            sticker_corners = sticker_corners.copy()
            sticker_centers = sticker_centers.copy()
            sticker_normals = sticker_normals.copy()

            body_corners[body_moving] = body_corners[body_moving] @ R.T
            body_centers[body_moving] = body_centers[body_moving] @ R.T
            body_normals[body_moving] = body_normals[body_moving] @ R.T
            sticker_corners[sticker_moving] = sticker_corners[sticker_moving] @ R.T
            sticker_centers[sticker_moving] = sticker_centers[sticker_moving] @ R.T
            sticker_normals[sticker_moving] = sticker_normals[sticker_moving] @ R.T

        visible_body_ids = np.flatnonzero(body_visible)
        visible_body_corners = body_corners[visible_body_ids]
        visible_body_centers = body_centers[visible_body_ids]
        visible_body_normals = body_normals[visible_body_ids]
        count_body = len(visible_body_ids)

        all_corners = np.concatenate((visible_body_corners, sticker_corners), axis=0)
        all_centers = np.concatenate((visible_body_centers, sticker_centers), axis=0)
        all_normals = np.concatenate((visible_body_normals, sticker_normals), axis=0)

        V = self._view_matrix()
        camera_distance = max(8.0, CAMERA_DISTANCE_FACTOR * cube_size)
        camera = np.array([0.0, 0.0, -camera_distance])
        centers_v = all_centers @ V.T
        normals_v = all_normals @ V.T
        to_camera = camera - centers_v
        front = np.einsum("ij,ij->i", normals_v, to_camera) > 0
        indices = np.flatnonzero(front)
        if not len(indices):
            return
        distances = np.linalg.norm(to_camera[indices], axis=1)
        order = indices[np.argsort(-distances)]
        corners_v = all_corners[order] @ V.T

        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        # One projection scale: focal length is tied to camera distance so the cube fits the viewport.
        half_extent = (cube_size + 1) / 2.0
        focal = min(width, height) * 0.42 * camera_distance / half_extent
        z = np.maximum(corners_v[:, :, 2] + camera_distance, 1.0)
        factor = focal / z
        
        px = width / 2.0 + corners_v[:, :, 0] * factor
        py = height / 2.0 - corners_v[:, :, 1] * factor
        polygons = np.stack((px, py), axis=-1)

        # Reuse Canvas polygon objects; hidden items remain in the pool for later frames.
        required = len(order)
        while len(self._polygon_items) < required:
            item = canvas.create_polygon(0, 0, 0, 0, fill=rgb_hex(BODY_COLOR), outline="#020202", width=1)
            self._polygon_items.append(item)

        for draw_index, geometry_index in enumerate(order):
            coords = polygons[draw_index].reshape(-1).tolist()
            if geometry_index < count_body:
                fill, outline = rgb_hex(BODY_COLOR), "#020202"
            else:
                sticker_index = int(geometry_index - count_body)
                color = int(state[sticker_index])
                fill = "#{:02x}{:02x}{:02x}".format(*COLOR_MAP[color])
                outline = "#0a0a0a"
            item = self._polygon_items[draw_index]
            canvas.coords(item, *coords)
            canvas.itemconfigure(item, fill=fill, outline=outline, width=1, state=tk.NORMAL)
            canvas.tag_raise(item)

        for item in self._polygon_items[required:]:
            canvas.itemconfigure(item, state=tk.HIDDEN)

        label = f"{cube_size}x{cube_size}x{cube_size}  |  Drag to rotate"
        if self._label_item is None:
            self._label_item = canvas.create_text(
                15, 15, anchor="nw", text=label, fill="#dcdcdc", font=("Arial", 15)
            )
        else:
            canvas.coords(self._label_item, 15, 15)
            canvas.itemconfigure(self._label_item, text=label)
        canvas.tag_raise(self._label_item)


def main():

    parser = argparse.ArgumentParser(
        description="Dynamic-size Rubik's Cube GUI"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="optional .pt checkpoint to load at startup"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000
    )

    args = parser.parse_args()

    app = Rubiks3DApp(
        args.model,
        args.host,
        args.port
    )

    app.run()


if __name__ == "__main__":
    main()