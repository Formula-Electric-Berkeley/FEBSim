import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants from Table II
l1 = 0.420
l2 = 0.162
l3 = 0.260
l5 = 0.519
o1 = np.array([0.0, 0.0, 0.0])
o2 = np.array([0.130, 0.160, 0.0])
p5 = np.array([0.010, 0.020, -0.400])
p8 = np.array([0.480, -0.050, 0.0])
l6 = np.array([-0.209, 0.007, -0.030])
l4 = np.array([0.0, 0.103, 0.0])
r = 0.473
rho = 0.312

# Input (steering + road profile)
s = 0.1
y = 0.05

# Rodrigues rotation matrix
def rodrigues_rotation_matrix(c):
    c1, c2, c3 = c
    c_delta = 1 + c1**2 + c2**2 + c3**2
    C = np.array([
        [(1 + c1**2 - c2**2 - c3**2), 2*(c1*c2 - c3),        2*(c1*c3 + c2)],
        [2*(c1*c2 + c3),            (1 - c1**2 + c2**2 - c3**2), 2*(c2*c3 - c1)],
        [2*(c1*c3 - c2),            2*(c2*c3 + c1),        (1 - c1**2 - c2**2 + c3**2)]
    ])
    return C / c_delta

# Eta residuals
def compute_residuals(vars):
    c1, c2, c3, theta, psi, x, z = vars
    c = np.array([c1, c2, c3])
    c_delta = 1 + c1**2 + c2**2 + c3**2
    R = rodrigues_rotation_matrix(c)

    # Compute eta1 from equation (5)
    Rz_theta = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    Rz_psi = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0, 0, 1]
    ])
    upper_arm_rot = Rz_psi @ np.array([l3, 0, 0])
    p1 = o1 + Rz_theta @ np.array([l1, 0, 0])
    p2 = o2 + Rz_psi @ np.array([l3, 0, 0])
    eta1 = o2 - o1 + upper_arm_rot - Rz_theta @ np.array([l1, 0, 0]) - R @ np.array([l2, 0, 0])

    # Compute eta2 from equation (12): length constraint on tie rod
    p4 = p1 + R @ (r * np.array([l2, 0, 0])) + R @ l4
    s_vec = np.array([s, 0, 0])  # simplified for now (could rotate)
    p5_s = p5 + s_vec
    eta2 = np.linalg.norm(p5_s - p4)**2 - l5**2

    # Compute eta3 from equation (13)
    p8_s = p8 + np.array([x, y, z])
    eta3 = p8_s - R @ l6 - R @ (rho * np.array([l2, 0, 0])) - Rz_theta @ np.array([l1, 0, 0]) - o1

    return np.concatenate((eta1, [eta2], eta3))

# Cost function to minimize
def cost(vars):
    residuals = compute_residuals(vars)
    return np.sum(residuals**2)


def visualize_suspension(c1, c2, c3, theta, psi, x, z):
    c = np.array([c1, c2, c3])
    R = rodrigues_rotation_matrix(c)

    # Points of interest
    p1 = o1 + np.array([l1 * np.cos(theta), l1 * np.sin(theta), 0])  # Lower A-arm outer
    p2 = o2 + np.array([l3 * np.cos(psi), l3 * np.sin(psi), 0])      # Upper A-arm outer
    p3 = p1 + R @ (r * np.array([l2, 0, 0]))                         # Midpoint knuckle
    p4 = p3 + R @ l4                                                # Tie rod attachment
    p5_s = p5 + np.array([s, 0, 0])                                  # Steering input
    p8_s = p8 + np.array([x, y, z])                                  # Wheel contact

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw A-arms
    ax.plot(*zip(o1, p1), 'ro-', label='Lower A-arm')
    ax.plot(*zip(o2, p2), 'go-', label='Upper A-arm')

    # Knuckle and wheel contact
    ax.plot(*zip(p1, p3), 'bo-', label='Knuckle lower half')
    ax.plot(*zip(p3, p2), 'bo--', label='Knuckle upper half')
    ax.plot(*zip(p3, p8_s), 'co-', label='To wheel contact')

    # Tie rod
    ax.plot(*zip(p4, p5_s), 'ko-', label='Tie rod')

    # Visual markers
    ax.scatter(*o1, c='r', s=50, marker='s', label='o1 (chassis)')
    ax.scatter(*o2, c='g', s=50, marker='s', label='o2 (chassis)')
    ax.scatter(*p8_s, c='c', s=60, marker='o', label='Wheel contact')
    ax.scatter(*p5_s, c='k', s=60, marker='x', label='Steering input')

    # Aesthetic tweaks
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Double Wishbone Suspension Configuration")
    ax.legend()
    ax.view_init(elev=20, azim=-45)
    ax.set_box_aspect([1, 1, 0.5])
    plt.tight_layout()
    plt.show()


# Initial guess (based on paper solution)
init_guess = [-2.5, -3.3, 1.3, 0.35, 0.5, -0.01, 0.01] # gives almost first solution
init_guess = [0.0]*7 # gives second solution in the paper, more or less

# Optimization
result = minimize(cost, init_guess, method='BFGS', options={'disp': True, 'maxiter': 1000})

# Output results
if result.success:
    c1_opt, c2_opt, c3_opt, theta_opt, psi_opt, x_opt, z_opt = result.x
    print(f"Solution:\nc1={c1_opt:.3f}, c2={c2_opt:.3f}, c3={c3_opt:.3f}")
    print(f"theta={theta_opt:.3f} rad, psi={psi_opt:.3f} rad")
    print(f"x={x_opt:.3f} m, z={z_opt:.3f} m")
    visualize_suspension(c1_opt, c2_opt, c3_opt, theta_opt, psi_opt, x_opt, z_opt)

else:
    print("Optimization failed:", result.message)

