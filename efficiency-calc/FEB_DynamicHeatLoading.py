import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

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
interpolate_motor_eta = LinearNDInterpolator(points, eff_pts, fill_value=0.85)


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

    # 2. Motor Efficiency Lookup
    eta_motor = interpolate_motor_eta(np.column_stack((rpm, torque)))
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

# Plot Heat Load Q(t) and Total Efficiency over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Top Plot: Heat Load Q(t) in kW
ax1.plot(time_s, q_heat_w / 1000.0, color="crimson", linewidth=2, label="Heat Load Q(t)")
ax1.set_ylabel("Heat Generation (kW)", fontweight="bold")
ax1.set_title("Transient Drivetrain Heat Load & System Efficiency")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left")

# Bottom Plot: System Efficiency (Motor * Inverter)
ax2.plot(time_s, eta_total * 100.0, color="forestgreen", linewidth=2, label="Combined Efficiency")
ax2.set_xlabel("Time (seconds)", fontweight="bold")
ax2.set_ylabel("Efficiency (%)", fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("drivetrain_heat_evolution.png", dpi=300)
plt.show()