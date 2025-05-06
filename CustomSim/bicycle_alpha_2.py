import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Raw data
mass = np.array([240, 260, 280, 300, 320, 320, 320, 320, 320, 320])
time = np.array([51.33984706, 49.0715557, 47.9047404, 47.17158263, 45.8137506,
                 59.21492, 60.48184, 56.64844, 58.33136, 55.0634])
radius = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3])

# Define grid for interpolation/extrapolation
mass_range = np.linspace(240, 340, 50)
radius_range = np.linspace(0.2, 0.32, 50)
MASS, RADIUS = np.meshgrid(mass_range, radius_range)

# Interpolate data onto grid using 'linear' or 'cubic' (for smoothness)
TIME = griddata((mass, radius), time, (MASS, RADIUS), method='cubic')

# Plotting the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(MASS, RADIUS, TIME, cmap='viridis', edgecolor='none')

# Labels and formatting
ax.set_xlabel('Mass (kg)')
ax.set_ylabel('Tire Radius (m)')
ax.set_zlabel('AutoX Time (s)')
ax.set_title('AutoX Time vs Mass and Tire Radius')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()
