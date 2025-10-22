import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def skew_symmetric(c):
    c1, c2, c3 = c
    return np.array([
        [0, -c3, c2],
        [c3, 0, -c1],
        [-c2, c1, 0]
    ])

def rodrigues_rotation_matrix(c):
    c = np.array(c)
    s = skew_symmetric(c)
    c_norm_sq = np.dot(c, c)
    I = np.eye(3)
    return I + (2 / (1 + c_norm_sq)) * (s + np.dot(s, s))

def plot_frames(R, ax):
    origin = np.zeros(3)
    
    # Original frame {0}
    ax.quiver(*origin, 1, 0, 0, color='r', label='x₀', linewidth=2)
    ax.quiver(*origin, 0, 1, 0, color='g', label='y₀', linewidth=2)
    ax.quiver(*origin, 0, 0, 1, color='b', label='z₀', linewidth=2)

    # Rotated frame {2}
    x2 = R @ np.array([1, 0, 0])
    y2 = R @ np.array([0, 1, 0])
    z2 = R @ np.array([0, 0, 1])
    
    ax.quiver(*origin, *x2, color='r', linestyle='dashed', linewidth=1.5, alpha=0.6)
    ax.quiver(*origin, *y2, color='g', linestyle='dashed', linewidth=1.5, alpha=0.6)
    ax.quiver(*origin, *z2, color='b', linestyle='dashed', linewidth=1.5, alpha=0.6)
    
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Frame {0} and Rotated Frame {2}")
    ax.legend()

# Example Rodrigues vector
c = [0.2, 0.3, 0.4]  # Not necessarily unit vector; represents axis * tan(θ/2)

R = rodrigues_rotation_matrix(c)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_frames(R, ax)
plt.show()
