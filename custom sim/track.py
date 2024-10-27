import math

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

base_name = 'track_files/'

class Track():
    def __init__(self, trackfile, grip = 0):
        self.filename = base_name + trackfile
        self.grip = grip
        self.info = self.read_info('Shape')
        self.config = self.read_config()

        self.r = self.info["Corner Radius"]
        self.x = self.info["Length"]
        
        self.r_mesh, self.x_tot_mesh, self.track_length = self.load_stats()
        
        self.n_sections = len(self.r)
        self.n_mesh = len(self.r_mesh)

        # Create a curvature function, note that this takes in the DISTANCE TRAVELLED THUS FAR, not just an index.
        # Super weird error, checkout (https://stackoverflow.com/questions/49459985/numpy-reciprocal-returns-different-values-when-called-repeatedly)
        curvature = np.zeros(len(self.x_tot_mesh))
        np.reciprocal(self.r_mesh, where=self.r_mesh > 0.0001, out=curvature)
        self.curvature_function = interp1d(self.x_tot_mesh, curvature, kind='linear', fill_value='extrapolate')

    def read_info(self, sheet_name=1, start_row=2, end_row=10000, cols="A:C"):
        info = pd.io.excel.read_excel(self.filename, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)
        info.columns = ["Type", "Length", "Corner Radius"]

        return info

    def read_config(self, sheet_name = "Info"):
        config = pd.io.excel.read_excel(self.filename, sheet_name)
        columns = config.columns.tolist()
        config = config.loc[config[columns[0]] == 'Configuration',columns[1]].item()

        return config

    def load_stats(self):
        mesh_size = 0.25
        total_len = sum(self.info["Length"])
        mesh_count = int(total_len // mesh_size) # why the fuck doesn't floor div give an int

        x_tot = np.cumsum(self.info["Length"])
        
        r_mesh = np.zeros(mesh_count)
        x_tot_mesh = np.zeros(mesh_count)

        i = 0 # two pointer thing
        for m in range(mesh_count):
            xt = mesh_size * m
            while xt > x_tot[i]:
                i += 1
            
            r = self.info["Corner Radius"][i]

            r_mesh[m] = r
            x_tot_mesh[m] = xt
        
        return r_mesh, x_tot_mesh, total_len
    
    def plot_curvature(self):
        plt.scatter(self.x_tot_mesh, self.curvature_function(self.x_tot_mesh))
        # plt.ylim(0, 1)
        plt.show()


def plot_track(s, kappa):
    # Initialize arrays for plotting
    x_pos = np.zeros_like(s)  # x positions
    y_pos = np.zeros_like(s)  # y positions
    theta = np.zeros_like(s)  # Heading angles

    # Integrate to get the positions and heading angle
    for i in range(1, len(s)):
        dtheta = kappa[i-1] * (s[i] - s[i-1])
        theta[i] = theta[i-1] + dtheta
        dx = np.cos(theta[i-1]) * (s[i] - s[i-1])
        dy = np.sin(theta[i-1]) * (s[i] - s[i-1])
        x_pos[i] = x_pos[i-1] + dx
        y_pos[i] = y_pos[i-1] + dy

    # Plot the track
    plt.figure(figsize=(10, 5))
    plt.plot(x_pos, y_pos, label='Track')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Track Plot')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

t = Track("Michigan_2022_AutoX.xlsx")

# print(min(r for r in t.r_mesh if r != 0))
# print(max(np.reciprocal(t.r_mesh, where=t.r_mesh != 0)))
# print(list(t.r_mesh))

# t.plot_curvature()
# plt.plot(t.x_tot_mesh, t.r_mesh)
# plt.show()