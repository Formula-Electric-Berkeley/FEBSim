# geometry_generator_FL.py
# Front-left double wishbone + low-mounted pull-rod hardpoint generator
# Coordinate system: x forward, y left, z up. Front-left => y > 0.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np


Vec3 = np.ndarray


def v3(x: float, y: float, z: float) -> Vec3:
    return np.array([float(x), float(y), float(z)], dtype=float)


def deg2rad(d: float) -> float:
    return float(d) * np.pi / 180.0


def unit(a: Vec3) -> Vec3:
    n = np.linalg.norm(a)
    if n <= 0:
        raise ValueError("Zero-length vector.")
    return a / n


def to_list(p: Vec3) -> list:
    return [float(p[0]), float(p[1]), float(p[2])]


@dataclass
class FrontLeftParams:
    # ---- Global placement ----
    wheelbase_m: float = 1.55
    track_m: float = 1.22
    wheel_radius_m: float = 0.254  # ~20" OD tire => 0.254 m radius
    wheel_center_x_m: float = 0.35
    wheel_center_z_m: float = 0.23

    # ---- Upright / BJ geometry ----
    upright_height_m: float = 0.24          # LBJ -> UBJ vertical separation
    bj_inboard_m: float = 0.09              # wheel center -> BJ lateral inboard (reduces y)
    lbj_z_offset_m: float = -0.03           # relative to wheel center
    lbj_x_offset_m: float = 0.00

    kpi_deg: float = 9.0                    # kingpin inclination (top inboard) in front view
    caster_deg: float = 6.0                 # caster (top rearward) in side view

    # ---- Control arm inner pivot "box" ----
    # Inboard pivot target (midpoint between front/rear pivots)
    lca_inboard_m: float = 0.28
    uca_inboard_m: float = 0.25
    lca_mid_x_offset_m: float = -0.02       # inboard pivot midpoint relative to wheel center x
    uca_mid_x_offset_m: float = -0.03
    lca_mid_z_m: float = 0.14
    uca_mid_z_m: float = 0.34

    # Fore-aft separation of inboard pivots
    lca_pivot_sep_m: float = 0.10
    uca_pivot_sep_m: float = 0.09

    # ---- Pull-rod ----
    # Low-mounted pull-rod: upright pickup near LBJ, chassis pickup low/inboard
    pr_upright_dx_m: float = 0.02
    pr_upright_dy_m: float = -0.03
    pr_upright_dz_m: float = 0.02

    pr_chassis_x_m: float = 0.10
    pr_chassis_y_m: float = 0.18
    pr_chassis_z_m: float = 0.10

    # ---- Basic sanity bounds ----
    min_link_length_m: float = 0.08
    max_link_length_m: float = 0.80


def generate_front_left(params: FrontLeftParams) -> Dict[str, Vec3]:
    """
    Returns a dict of named hardpoints (Vec3 numpy arrays).
    """
    # Wheel center (FL)
    y_wc = params.track_m / 2.0
    WC = v3(params.wheel_center_x_m, y_wc, params.wheel_center_z_m)

    # Lower ball joint (LBJ) placed inboard and slightly below hub center
    LBJ = WC + v3(params.lbj_x_offset_m, -params.bj_inboard_m, params.lbj_z_offset_m)

    # Upper ball joint (UBJ) using simple KPI+caster model:
    # UBJ = LBJ + [dx_from_caster, dy_from_kpi, dz]
    # Positive caster_deg -> UBJ moves rearward => negative x (since x forward).
    # Positive kpi_deg -> top inboard => negative y from LBJ.
    dz = params.upright_height_m
    dx = -dz * np.tan(deg2rad(params.caster_deg))
    dy = -dz * np.tan(deg2rad(params.kpi_deg))
    UBJ = LBJ + v3(dx, dy, dz)

    # Inboard pivot midpoints (target)
    LCA_mid = v3(
        WC[0] + params.lca_mid_x_offset_m,
        WC[1] - params.lca_inboard_m,
        params.lca_mid_z_m,
    )
    UCA_mid = v3(
        WC[0] + params.uca_mid_x_offset_m,
        WC[1] - params.uca_inboard_m,
        params.uca_mid_z_m,
    )

    # Split each midpoint into front/rear pivots along x axis
    lsep = params.lca_pivot_sep_m / 2.0
    usep = params.uca_pivot_sep_m / 2.0

    LCA_F = LCA_mid + v3(+lsep, 0.0, 0.0)  # front
    LCA_R = LCA_mid + v3(-lsep, 0.0, 0.0)  # rear

    UCA_F = UCA_mid + v3(+usep, 0.0, 0.0)  # front
    UCA_R = UCA_mid + v3(-usep, 0.0, 0.0)  # rear

    # Pull-rod endpoints
    PR_U = LBJ + v3(params.pr_upright_dx_m, params.pr_upright_dy_m, params.pr_upright_dz_m)
    PR_C = v3(params.pr_chassis_x_m, params.pr_chassis_y_m, params.pr_chassis_z_m)

    pts = {
        "WC": WC,
        "LBJ": LBJ,
        "UBJ": UBJ,
        "LCA_F": LCA_F,
        "LCA_R": LCA_R,
        "UCA_F": UCA_F,
        "UCA_R": UCA_R,
        "PR_U": PR_U,
        "PR_C": PR_C,
    }

    _validate_geometry(pts, params)
    return pts


def _link_length(a: Vec3, b: Vec3) -> float:
    return float(np.linalg.norm(a - b))


def _validate_geometry(pts: Dict[str, Vec3], params: FrontLeftParams) -> None:
    # Control arm link lengths to outer joints (informational + sanity)
    links = {
        "LCA_F->LBJ": _link_length(pts["LCA_F"], pts["LBJ"]),
        "LCA_R->LBJ": _link_length(pts["LCA_R"], pts["LBJ"]),
        "UCA_F->UBJ": _link_length(pts["UCA_F"], pts["UBJ"]),
        "UCA_R->UBJ": _link_length(pts["UCA_R"], pts["UBJ"]),
        "PR_C->PR_U": _link_length(pts["PR_C"], pts["PR_U"]),
        "LBJ->UBJ": _link_length(pts["LBJ"], pts["UBJ"]),
    }

    mn = params.min_link_length_m
    mx = params.max_link_length_m
    bad = {k: L for k, L in links.items() if (L < mn or L > mx)}
    if bad:
        msg = "Geometry sanity check failed (link length out of bounds):\n"
        for k, L in bad.items():
            msg += f"  {k}: {L:.4f} m (bounds [{mn:.3f}, {mx:.3f}])\n"
        raise ValueError(msg)


def export_json(pts: Dict[str, Vec3], params: FrontLeftParams, path: str) -> None:
    out = {
        "units": "m",
        "coordinate_system": {"x": "forward", "y": "left", "z": "up"},
        "params": asdict(params),
        "hardpoints": {name: to_list(p) for name, p in pts.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _print_points(pts: Dict[str, Vec3]) -> None:
    keys = ["WC", "LBJ", "UBJ", "LCA_F", "LCA_R", "UCA_F", "UCA_R", "PR_U", "PR_C"]
    for k in keys:
        p = pts[k]
        print(f"{k:6s}:  x={p[0]: .4f}  y={p[1]: .4f}  z={p[2]: .4f}  (m)")


def _plot_front_view(pts: Dict[str, Vec3]) -> None:
    import matplotlib.pyplot as plt

    def seg(a: str, b: str) -> Tuple[np.ndarray, np.ndarray]:
        A, B = pts[a], pts[b]
        # front view: y-z
        return np.array([A[1], B[1]]), np.array([A[2], B[2]])

    plt.figure()
    for a, b, lab in [
        ("LCA_F", "LBJ", "LCA front"),
        ("LCA_R", "LBJ", "LCA rear"),
        ("UCA_F", "UBJ", "UCA front"),
        ("UCA_R", "UBJ", "UCA rear"),
        ("LBJ", "UBJ", "Upright"),
        ("PR_C", "PR_U", "Pull rod"),
    ]:
        y, z = seg(a, b)
        plt.plot(y, z, marker="o", label=lab)

    # wheel center marker
    plt.plot([pts["WC"][1]], [pts["WC"][2]], marker="x", markersize=10, label="Wheel center")

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.title("Front-left suspension (front view: y-z)")
    plt.legend()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="", help="Output JSON path (optional).")
    ap.add_argument("--plot", action="store_true", help="Plot simple front view (y-z).")
    args = ap.parse_args()

    params = FrontLeftParams()
    pts = generate_front_left(params)

    _print_points(pts)

    if args.out:
        export_json(pts, params, args.out)
        print(f"\nWrote: {args.out}")

    if args.plot:
        _plot_front_view(pts)


if __name__ == "__main__":
    main()