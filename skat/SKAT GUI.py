# suspension_gui_FL.py
# Front-left double wishbone + low-mounted pull-rod:
# - Hardpoint editor (add/update/delete)
# - Load/Save JSON
# - Embedded visualizer (3D + front view y-z)
# - Motion generator (bump/rebound) with kinematic solve
#
# Coordinates: x forward, y left, z up. Units: meters.

from __future__ import annotations

import json
import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib embed
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# ----------------------------- math helpers -----------------------------

Vec3 = np.ndarray


def v3(x: float, y: float, z: float) -> Vec3:
    return np.array([float(x), float(y), float(z)], dtype=float)


def norm(a: Vec3) -> float:
    return float(np.linalg.norm(a))


def safe_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        raise ValueError(f"Invalid number: {s!r}")


def angle_deg(a: float) -> float:
    return a * 180.0 / math.pi


# ----------------------------- default geometry -----------------------------

DEFAULT_NODE_NAMES = [
    "WC",
    "LBJ",
    "UBJ",
    "LCA_F",
    "LCA_R",
    "UCA_F",
    "UCA_R",
    "PR_U",
    "PR_C",
]

REQUIRED_FOR_KINEMATICS = ["LBJ", "UBJ", "LCA_F", "LCA_R", "UCA_F", "UCA_R", "WC"]


def default_hardpoints() -> Dict[str, Vec3]:
    # Reasonable starter values (m). Front-left => y > 0.
    # x forward, y left, z up.
    pts = {
        "WC":   v3(0.35, 0.61, 0.23),
        "LBJ":  v3(0.35, 0.52, 0.20),
        "UBJ":  v3(0.33, 0.49, 0.44),
        "LCA_F": v3(0.02, 0.33, 0.14),
        "LCA_R": v3(-0.08, 0.33, 0.14),
        "UCA_F": v3(0.02, 0.36, 0.34),
        "UCA_R": v3(-0.07, 0.36, 0.34),
        "PR_U": v3(0.37, 0.49, 0.22),
        "PR_C": v3(0.10, 0.18, 0.10),
    }
    return pts


# ----------------------------- kinematics solver -----------------------------
# Unknowns: [LBJx, LBJy, LBJz, UBJx, UBJy, UBJz]
# Constraints:
# 1) |LBJ - LCA_F|^2 = L1^2
# 2) |LBJ - LCA_R|^2 = L2^2
# 3) |UBJ - UCA_F|^2 = U1^2
# 4) |UBJ - UCA_R|^2 = U2^2
# 5) |UBJ - LBJ|^2  = H^2 (upright length)
# 6) WC_z = WC0_z + dz  with WC = LBJ + (WC0 - LBJ0)  => LBJ_z + (WC0_z-LBJ0_z) = WC0_z + dz
#    => LBJ_z = LBJ0_z + dz
#
# So we prescribe bump by forcing LBJ_z directly. WC then follows rigidly by offset.

@dataclass
class KinematicsState:
    pts0: Dict[str, Vec3]
    lengths: Dict[str, float]
    wc_offset: Vec3  # WC0 - LBJ0


def build_kinematics_state(pts: Dict[str, Vec3]) -> KinematicsState:
    for k in REQUIRED_FOR_KINEMATICS:
        if k not in pts:
            raise ValueError(f"Missing required node: {k}")

    LBJ0 = pts["LBJ"].copy()
    UBJ0 = pts["UBJ"].copy()
    LCA_F = pts["LCA_F"].copy()
    LCA_R = pts["LCA_R"].copy()
    UCA_F = pts["UCA_F"].copy()
    UCA_R = pts["UCA_R"].copy()

    lengths = {
        "L1": norm(LBJ0 - LCA_F),
        "L2": norm(LBJ0 - LCA_R),
        "U1": norm(UBJ0 - UCA_F),
        "U2": norm(UBJ0 - UCA_R),
        "H":  norm(UBJ0 - LBJ0),
        "PR": norm(pts.get("PR_U", LBJ0) - pts.get("PR_C", LBJ0)),
    }
    wc_offset = pts["WC"] - LBJ0
    return KinematicsState(pts0={k: v.copy() for k, v in pts.items()}, lengths=lengths, wc_offset=wc_offset)


def residual(x: np.ndarray, state: KinematicsState, dz: float) -> np.ndarray:
    LBJ = x[0:3]
    UBJ = x[3:6]

    LCA_F = state.pts0["LCA_F"]
    LCA_R = state.pts0["LCA_R"]
    UCA_F = state.pts0["UCA_F"]
    UCA_R = state.pts0["UCA_R"]

    L1 = state.lengths["L1"]
    L2 = state.lengths["L2"]
    U1 = state.lengths["U1"]
    U2 = state.lengths["U2"]
    H  = state.lengths["H"]

    # squared-distance residuals (better scaling)
    r = np.zeros(6, dtype=float)
    r[0] = np.dot(LBJ - LCA_F, LBJ - LCA_F) - L1 * L1
    r[1] = np.dot(LBJ - LCA_R, LBJ - LCA_R) - L2 * L2
    r[2] = np.dot(UBJ - UCA_F, UBJ - UCA_F) - U1 * U1
    r[3] = np.dot(UBJ - UCA_R, UBJ - UCA_R) - U2 * U2
    r[4] = np.dot(UBJ - LBJ, UBJ - LBJ) - H * H

    LBJ0 = state.pts0["LBJ"]
    r[5] = (LBJ[2] - (LBJ0[2] + dz))
    return r


def solve_newton(state: KinematicsState, dz: float, x0: Optional[np.ndarray] = None) -> np.ndarray:
    if x0 is None:
        LBJ0 = state.pts0["LBJ"]
        UBJ0 = state.pts0["UBJ"]
        x = np.hstack([LBJ0, UBJ0]).astype(float)
    else:
        x = x0.astype(float).copy()

    # initial guess: shift LBJ/UBJ in z by dz
    x[2] = state.pts0["LBJ"][2] + dz
    x[5] = state.pts0["UBJ"][2] + dz

    max_iter = 35
    tol = 1e-10

    for _ in range(max_iter):
        r0 = residual(x, state, dz)
        if np.max(np.abs(r0)) < tol:
            return x

        # finite-diff Jacobian
        J = np.zeros((6, 6), dtype=float)
        eps = 1e-6
        for j in range(6):
            x1 = x.copy()
            x1[j] += eps
            r1 = residual(x1, state, dz)
            J[:, j] = (r1 - r0) / eps

        # solve J dx = -r
        try:
            dx = np.linalg.solve(J, -r0)
        except np.linalg.LinAlgError:
            # fallback to least squares
            dx, *_ = np.linalg.lstsq(J, -r0, rcond=None)

        # damping / line search
        alpha = 1.0
        base = np.linalg.norm(r0)
        for _ls in range(10):
            xt = x + alpha * dx
            rt = residual(xt, state, dz)
            if np.linalg.norm(rt) < base:
                x = xt
                break
            alpha *= 0.5
        else:
            # no improvement
            x = x + 0.2 * dx

    raise RuntimeError("Kinematics solver did not converge.")


def apply_solution_to_points(pts: Dict[str, Vec3], state: KinematicsState, x: np.ndarray, dz: float) -> Dict[str, Vec3]:
    out = {k: v.copy() for k, v in pts.items()}
    LBJ = x[0:3]
    UBJ = x[3:6]
    out["LBJ"] = LBJ
    out["UBJ"] = UBJ

    # Wheel center follows rigidly from LBJ (upright rigid offset)
    out["WC"] = LBJ + state.wc_offset

    # Pull-rod upright node: if present in original, follow LBJ by same offset
    if "PR_U" in out and "PR_U" in state.pts0:
        pr_offset = state.pts0["PR_U"] - state.pts0["LBJ"]
        out["PR_U"] = LBJ + pr_offset

    return out


def compute_camber_deg(pts: Dict[str, Vec3]) -> float:
    # Simple front-view camber from LBJ->UBJ in y-z:
    # camber >0 means top outward? (sign conventions vary)
    # Here: in front view, vector v = UBJ-LBJ.
    # Angle from vertical in y-z plane, sign by y component.
    v = pts["UBJ"] - pts["LBJ"]
    vy = float(v[1])
    vz = float(v[2])
    if abs(vz) < 1e-12:
        return float("nan")
    ang = math.atan2(vy, vz)  # radians
    return angle_deg(ang)


# ----------------------------- GUI -----------------------------

class SuspensionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Front-Left Suspension GUI (Hardpoints + Visualizer + Motion)")
        self.geometry("1200x750")

        self.pts: Dict[str, Vec3] = default_hardpoints()
        self.state: Optional[KinematicsState] = None
        self.last_sol: Optional[np.ndarray] = None

        self._build_ui()
        self._refresh_table()
        self._rebuild_state()
        self._redraw()

    # ---- UI layout ----

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # --- Left: file + editor + motion ---
        file_box = ttk.LabelFrame(left, text="File", padding=8)
        file_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Button(file_box, text="Load JSON...", command=self.on_load).grid(row=0, column=0, sticky="ew")
        ttk.Button(file_box, text="Save JSON...", command=self.on_save).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        file_box.columnconfigure(0, weight=1)
        file_box.columnconfigure(1, weight=1)

        editor = ttk.LabelFrame(left, text="Hardpoints", padding=8)
        editor.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        left.rowconfigure(1, weight=1)

        # table
        self.tree = ttk.Treeview(editor, columns=("name", "x", "y", "z"), show="headings", height=12)
        for col, w in [("name", 110), ("x", 90), ("y", 90), ("z", 90)]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        self.tree.grid(row=0, column=0, columnspan=4, sticky="nsew")
        editor.rowconfigure(0, weight=1)
        editor.columnconfigure(0, weight=1)

        # selection -> editor fields
        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)

        ttk.Label(editor, text="Node:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.node_name = tk.StringVar(value=DEFAULT_NODE_NAMES[0])
        self.node_menu = ttk.Combobox(editor, textvariable=self.node_name, values=DEFAULT_NODE_NAMES, state="readonly", width=18)
        self.node_menu.grid(row=1, column=1, sticky="ew", pady=(8, 0), padx=(8, 0))

        ttk.Label(editor, text="x").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Label(editor, text="y").grid(row=2, column=1, sticky="w", pady=(6, 0))
        ttk.Label(editor, text="z").grid(row=2, column=2, sticky="w", pady=(6, 0))

        self.x_entry = ttk.Entry(editor, width=10)
        self.y_entry = ttk.Entry(editor, width=10)
        self.z_entry = ttk.Entry(editor, width=10)
        self.x_entry.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        self.y_entry.grid(row=3, column=1, sticky="ew", pady=(2, 0), padx=(8, 0))
        self.z_entry.grid(row=3, column=2, sticky="ew", pady=(2, 0), padx=(8, 0))

        ttk.Button(editor, text="Add/Update", command=self.on_add_update).grid(row=3, column=3, sticky="ew", padx=(8, 0), pady=(2, 0))
        ttk.Button(editor, text="Delete", command=self.on_delete).grid(row=4, column=3, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Button(editor, text="Rebuild kinematics", command=self.on_rebuild).grid(row=5, column=3, sticky="ew", padx=(8, 0), pady=(6, 0))

        # status readout
        self.status = tk.StringVar(value="")
        ttk.Label(editor, textvariable=self.status, foreground="#444").grid(row=6, column=0, columnspan=4, sticky="w", pady=(8, 0))

        motion = ttk.LabelFrame(left, text="Motion (bump/rebound)", padding=8)
        motion.grid(row=2, column=0, sticky="ew")

        self.dz_var = tk.DoubleVar(value=0.0)
        self.dz_scale = ttk.Scale(motion, from_=-0.10, to=0.10, variable=self.dz_var, command=self.on_dz_changed)
        self.dz_scale.grid(row=0, column=0, columnspan=3, sticky="ew")
        motion.columnconfigure(0, weight=1)

        self.dz_label = ttk.Label(motion, text="dz = 0.000 m")
        self.dz_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        ttk.Button(motion, text="Home (dz=0)", command=self.on_home).grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(4, 0))
        ttk.Button(motion, text="Animate", command=self.on_animate).grid(row=1, column=2, sticky="e", padx=(8, 0), pady=(4, 0))

        # --- Right: plots ---
        plot_box = ttk.LabelFrame(right, text="Visualizer", padding=8)
        plot_box.grid(row=0, column=0, sticky="nsew")
        right.rowconfigure(0, weight=1)
        plot_box.rowconfigure(0, weight=1)
        plot_box.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(7.0, 6.0), dpi=100)
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.axfv = self.fig.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_box)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ---- Data ops ----

    def on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Load hardpoints JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Accept either {hardpoints:{...}} or raw dict of points
            if isinstance(data, dict) and "hardpoints" in data and isinstance(data["hardpoints"], dict):
                hp = data["hardpoints"]
            else:
                hp = data

            pts: Dict[str, Vec3] = {}
            for k, v in hp.items():
                if not (isinstance(v, list) and len(v) == 3):
                    continue
                pts[str(k)] = v3(v[0], v[1], v[2])

            if not pts:
                raise ValueError("No valid hardpoints found in JSON.")
            self.pts = pts
            self._refresh_table()
            self._rebuild_state()
            self._redraw()
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    def on_save(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save hardpoints JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        try:
            out = {
                "units": "m",
                "coordinate_system": {"x": "forward", "y": "left", "z": "up"},
                "hardpoints": {k: [float(p[0]), float(p[1]), float(p[2])] for k, p in self.pts.items()},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def on_select_row(self, _evt=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        name, xs, ys, zs = item["values"]
        self.node_name.set(name)
        self._set_entries(xs, ys, zs)

    def _set_entries(self, xs: str, ys: str, zs: str) -> None:
        for ent, val in [(self.x_entry, xs), (self.y_entry, ys), (self.z_entry, zs)]:
            ent.delete(0, tk.END)
            ent.insert(0, str(val))

    def on_add_update(self) -> None:
        try:
            name = self.node_name.get().strip()
            if not name:
                raise ValueError("Node name is empty.")
            x = safe_float(self.x_entry.get())
            y = safe_float(self.y_entry.get())
            z = safe_float(self.z_entry.get())
            self.pts[name] = v3(x, y, z)
            self._refresh_table()
            self._rebuild_state()
            self._redraw()
        except Exception as e:
            messagebox.showerror("Add/Update failed", str(e))

    def on_delete(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        item = self.tree.item(sel[0])
        name = item["values"][0]
        if name in self.pts:
            del self.pts[name]
        self._refresh_table()
        self._rebuild_state()
        self._redraw()

    def on_rebuild(self) -> None:
        self._rebuild_state()
        self._redraw()

    def _refresh_table(self) -> None:
        for r in self.tree.get_children():
            self.tree.delete(r)
        for name in sorted(self.pts.keys()):
            p = self.pts[name]
            self.tree.insert("", tk.END, values=(name, f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"))

    def _rebuild_state(self) -> None:
        try:
            self.state = build_kinematics_state(self.pts)
            self.last_sol = None
            self.status.set("Kinematics: OK")
        except Exception as e:
            self.state = None
            self.last_sol = None
            self.status.set(f"Kinematics: not ready ({e})")

    # ---- Motion ----

    def on_home(self) -> None:
        self.dz_var.set(0.0)
        self.on_dz_changed()

    def on_dz_changed(self, _evt=None) -> None:
        dz = float(self.dz_var.get())
        self.dz_label.configure(text=f"dz = {dz:+.3f} m")
        self._redraw()

    def on_animate(self) -> None:
        if self.state is None:
            messagebox.showwarning("Not ready", "Kinematics state not built (missing nodes).")
            return

        # simple sweep: -0.06 -> +0.06 -> -0.06
        seq = list(np.linspace(-0.06, 0.06, 40)) + list(np.linspace(0.06, -0.06, 40))
        for dz in seq:
            self.dz_var.set(float(dz))
            self.dz_label.configure(text=f"dz = {dz:+.3f} m")
            self._redraw()
            self.update_idletasks()
            self.update()

    # ---- Drawing ----

    def _get_current_points(self) -> Dict[str, Vec3]:
        # If kinematics is ready, solve for dz and return moved geometry.
        dz = float(self.dz_var.get())
        if self.state is None:
            return {k: v.copy() for k, v in self.pts.items()}

        try:
            x0 = self.last_sol
            x = solve_newton(self.state, dz, x0=x0)
            self.last_sol = x.copy()
            moved = apply_solution_to_points(self.pts, self.state, x, dz)
            self.status.set("Kinematics: OK")
            return moved
        except Exception as e:
            self.status.set(f"Kinematics: solve failed ({e})")
            return {k: v.copy() for k, v in self.pts.items()}

    def _draw_link(self, ax, pts: Dict[str, Vec3], a: str, b: str, *, label: Optional[str] = None):
        if a not in pts or b not in pts:
            return
        A, B = pts[a], pts[b]
        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], marker="o", linewidth=2, label=label)

    def _draw_front_view_link(self, ax, pts: Dict[str, Vec3], a: str, b: str, *, label: Optional[str] = None):
        if a not in pts or b not in pts:
            return
        A, B = pts[a], pts[b]
        ax.plot([A[1], B[1]], [A[2], B[2]], marker="o", linewidth=2, label=label)

    def _redraw(self) -> None:
        pts_now = self._get_current_points()

        # clear
        self.ax3d.cla()
        self.axfv.cla()

        # 3D links
        self._draw_link(self.ax3d, pts_now, "LCA_F", "LBJ", label="LCA front")
        self._draw_link(self.ax3d, pts_now, "LCA_R", "LBJ", label="LCA rear")
        self._draw_link(self.ax3d, pts_now, "UCA_F", "UBJ", label="UCA front")
        self._draw_link(self.ax3d, pts_now, "UCA_R", "UBJ", label="UCA rear")
        self._draw_link(self.ax3d, pts_now, "LBJ", "UBJ", label="Upright")
        self._draw_link(self.ax3d, pts_now, "PR_C", "PR_U", label="Pull rod")
        if "WC" in pts_now:
            W = pts_now["WC"]
            self.ax3d.scatter([W[0]], [W[1]], [W[2]], marker="x", s=60, label="Wheel center")

        # node scatter (all)
        xs, ys, zs = [], [], []
        for p in pts_now.values():
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        self.ax3d.scatter(xs, ys, zs, s=10)

        self.ax3d.set_title("3D view")
        self.ax3d.set_xlabel("x (m)")
        self.ax3d.set_ylabel("y (m)")
        self.ax3d.set_zlabel("z (m)")
        self.ax3d.grid(True)

        # reasonable bounds
        if xs:
            pad = 0.08
            self.ax3d.set_xlim(min(xs) - pad, max(xs) + pad)
            self.ax3d.set_ylim(min(ys) - pad, max(ys) + pad)
            self.ax3d.set_zlim(min(zs) - pad, max(zs) + pad)

        # front view (y-z)
        self._draw_front_view_link(self.axfv, pts_now, "LCA_F", "LBJ", label="LCA front")
        self._draw_front_view_link(self.axfv, pts_now, "LCA_R", "LBJ", label="LCA rear")
        self._draw_front_view_link(self.axfv, pts_now, "UCA_F", "UBJ", label="UCA front")
        self._draw_front_view_link(self.axfv, pts_now, "UCA_R", "UBJ", label="UCA rear")
        self._draw_front_view_link(self.axfv, pts_now, "LBJ", "UBJ", label="Upright")
        self._draw_front_view_link(self.axfv, pts_now, "PR_C", "PR_U", label="Pull rod")

        if "WC" in pts_now:
            W = pts_now["WC"]
            self.axfv.plot([W[1]], [W[2]], marker="x", markersize=9, label="Wheel center")

        camber = compute_camber_deg(pts_now) if ("LBJ" in pts_now and "UBJ" in pts_now) else float("nan")
        self.axfv.set_title(f"Front view (y-z) | camber ≈ {camber:+.2f} deg")
        self.axfv.set_xlabel("y (m)")
        self.axfv.set_ylabel("z (m)")
        self.axfv.grid(True)
        self.axfv.axis("equal")
        self.axfv.legend(loc="best", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()


# ----------------------------- entry point -----------------------------

if __name__ == "__main__":
    app = SuspensionGUI()
    app.mainloop()