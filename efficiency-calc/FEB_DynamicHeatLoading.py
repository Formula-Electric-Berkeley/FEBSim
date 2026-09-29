import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

# 1. Load Efficiency Map CSV
df_map = pd.read_csv("emrax228_efficiency.csv", header=None)
if isinstance(df_map.iloc[0, 0], str):
    df_map = pd.read_csv("emrax228_efficiency.csv")

rpm_pts = df_map.iloc[:, 0].to_numpy(dtype=float)
torque_pts = df_map.iloc[:, 1].to_numpy(dtype=float)
eff_pts = df_map.iloc[:, 2].to_numpy(dtype=float)

if np.max(eff_pts) > 1.0:
    eff_pts /= 100.0  # Convert % to decimal scale

points = np.column_stack((rpm_pts, torque_pts))

# Build both Linear and Clough-Tocher 2D interpolators
interpolateCT_motor_eta = CloughTocher2DInterpolator(
    points, eff_pts, fill_value=0.85
)
interpolateL_motor_eta = LinearNDInterpolator(
    points, eff_pts, fill_value=0.85
)


def calculate_drivetrain_heat_and_power(
    speed_array, torque_array, speed_in_rad_s=True, inv_eff=0.97
):
    torque = np.abs(np.asarray(torque_array, dtype=float))

    if speed_in_rad_s:
        omega = np.asarray(speed_array, dtype=float)
        rpm = omega * (60.0 / (2.0 * np.pi))
    else:
        rpm = np.asarray(speed_array, dtype=float)
        omega = rpm * (2.0 * np.pi / 60.0)

    # 1. Mechanical Shaft Power (Watts)
    p_mech = torque * omega

    # 2. Motor Efficiency Lookup (Using Clough-Tocher)
    eta_motor = interpolateCT_motor_eta(np.column_stack((rpm, torque)))
    eta_motor = np.nan_to_num(eta_motor, nan=0.85)
    eta_motor = np.clip(eta_motor, 0.05, 1.0)

    # 3. Combined Drivetrain Efficiency (Motor * Inverter)
    eta_total = eta_motor * inv_eff

    # 4. Total Electrical Power Draw & Heat Load (Watts)
    p_battery = np.where(p_mech > 0, p_mech / eta_total, 0.0)
    q_heat = p_battery - p_mech  # Combined heat loss of motor + inverter

    return q_heat, p_battery, p_mech, eta_total, rpm


# ==============================================================================
# SAMPLE RUN AND GRAPH GENERATION
# ==============================================================================

# Simulated 10-second trajectory (replace with actual lap sim telemetry)
time_s = np.linspace(0, 10, 500)
sim_speed_rad_s = np.linspace(100, 500, 500)  # Speed in rad/s
sim_torque_nm = np.abs(np.sin(time_s * 1.5)) * 120 + 20  # Torque in Nm

# Run calculation
q_heat_w, p_bat_w, p_mech_w, eta_total, rpm_calculated = (
    calculate_drivetrain_heat_and_power(
        sim_speed_rad_s, sim_torque_nm, speed_in_rad_s=True, inv_eff=0.97
    )
)

# Meshgrid for Efficiency Surface Comparisons
rpm_grid = np.linspace(rpm_pts.min(), rpm_pts.max(), 150)
torque_grid = np.linspace(torque_pts.min(), torque_pts.max(), 150)
RPM_mesh, TORQUE_mesh = np.meshgrid(rpm_grid, torque_grid)

EFF_mesh_L = (
    np.nan_to_num(interpolateL_motor_eta(RPM_mesh, TORQUE_mesh), nan=0.85)
    * 100.0
)
EFF_mesh_CT = (
    np.nan_to_num(interpolateCT_motor_eta(RPM_mesh, TORQUE_mesh), nan=0.85)
    * 100.0
)

# ------------------------------------------------------------------------------
# PLOT 1: Top-Down 2D Contour Comparison (Linear vs Clough-Tocher)
# ------------------------------------------------------------------------------
fig_2d, (ax_l2d, ax_ct2d) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# LinearND 2D Contour
c1 = ax_l2d.contourf(
    RPM_mesh, TORQUE_mesh, EFF_mesh_L, levels=25, cmap="viridis"
)
ax_l2d.scatter(
    rpm_pts,
    torque_pts,
    c=eff_pts * 100.0,
    cmap="viridis",
    edgecolors="black",
    linewidths=0.5,
    s=15,
)
ax_l2d.set_title("LinearND Interpolator (Top-Down)", fontweight="bold")
ax_l2d.set_xlabel("Motor Speed (RPM)", fontweight="bold")
ax_l2d.set_ylabel("Torque (Nm)", fontweight="bold")
fig_2d.colorbar(c1, ax=ax_l2d, label="Efficiency (%)")

# Clough-Tocher 2D Contour
c2 = ax_ct2d.contourf(
    RPM_mesh, TORQUE_mesh, EFF_mesh_CT, levels=25, cmap="viridis"
)
ax_ct2d.scatter(
    rpm_pts,
    torque_pts,
    c=eff_pts * 100.0,
    cmap="viridis",
    edgecolors="black",
    linewidths=0.5,
    s=15,
)
ax_ct2d.set_title("Clough-Tocher Interpolator (Top-Down)", fontweight="bold")
ax_ct2d.set_xlabel("Motor Speed (RPM)", fontweight="bold")
fig_2d.colorbar(c2, ax=ax_ct2d, label="Efficiency (%)")

plt.suptitle(
    "Emrax 228 Top-Down Efficiency Comparison (Raw Points Overlaid)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("topdown_interpolator_comparison.png", dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# PLOT 2: 3D Surface Mesh Comparison (Linear vs Clough-Tocher)
# ------------------------------------------------------------------------------
fig_3d = plt.figure(figsize=(15, 6))

# LinearND 3D Surface
ax_l3d = fig_3d.add_subplot(121, projection="3d")
surf1 = ax_l3d.plot_surface(
    RPM_mesh, TORQUE_mesh, EFF_mesh_L, cmap="viridis", alpha=0.85
)
ax_l3d.set_title("LinearND Interpolator (3D Surface)", fontweight="bold")
ax_l3d.set_xlabel("RPM", fontweight="bold")
ax_l3d.set_ylabel("Torque (Nm)", fontweight="bold")
ax_l3d.set_zlabel("Efficiency (%)", fontweight="bold")

# Clough-Tocher 3D Surface
ax_ct3d = fig_3d.add_subplot(122, projection="3d")
surf2 = ax_ct3d.plot_surface(
    RPM_mesh, TORQUE_mesh, EFF_mesh_CT, cmap="viridis", alpha=0.85
)
ax_ct3d.set_title("Clough-Tocher Interpolator (3D Surface)", fontweight="bold")
ax_ct3d.set_xlabel("RPM", fontweight="bold")
ax_ct3d.set_ylabel("Torque (Nm)", fontweight="bold")
ax_ct3d.set_zlabel("Efficiency (%)", fontweight="bold")

plt.suptitle(
    "Emrax 228 3D Efficiency Surface Comparison",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("3d_interpolator_comparison.png", dpi=300)
plt.show()

# ------------------------------------------------------------------------------
# PLOT 3: Transient Heat Load Q(t) and System Efficiency over Time (Sim Plot)
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Top Plot: Heat Load Q(t) in kW
ax1.plot(
    time_s, q_heat_w / 1000.0, color="crimson", linewidth=2, label="Heat Load Q(t)"
)
ax1.set_ylabel("Heat Generation (kW)", fontweight="bold")
ax1.set_title("Transient Drivetrain Heat Load & System Efficiency")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left")

# Bottom Plot: System Efficiency (Motor * Inverter)
ax2.plot(
    time_s,
    eta_total * 100.0,
    color="forestgreen",
    linewidth=2,
    label="Combined Efficiency",
)
ax2.set_xlabel("Time (seconds)", fontweight="bold")
ax2.set_ylabel("Efficiency (%)", fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("drivetrain_heat_evolution.png", dpi=300)
plt.show()
