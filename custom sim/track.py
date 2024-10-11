import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt

base_name = 'track_files/'

class Track():
    def __init__(self, trackfile, grip = 0):
        self.filename = base_name + trackfile
        self.grip = grip
        self.read_info('Shape')
        self.read_config()
        self.load_stats()

    def read_info(self, sheet_name=1, start_row=2, end_row=10000, cols="A:C"):
        self.info = pd.io.excel.read_excel(self.filename, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)
        self.info.columns = ["Type", "Length", "Corner Radius"]

    def read_config(self, sheet_name = "Info"):
        self.config = pd.io.excel.read_excel(self.filename, sheet_name)
        columns = self.config.columns.tolist()
        self.config = self.config.loc[self.config[columns[0]] == 'Configuration',columns[1]].item()

    def load_stats(self):
        mesh_size = 0.25

        #Getting Curvature
        R = self.info.loc[:, "Corner Radius"] #0 or NaN on straights, otherwise a float
        R = np.nan_to_num(R)
        R = R.astype(float)
        R[R==0] = np.inf
        self.n = len(R)

        #Getting type
        type_tmp = self.info.loc[:, "Type"]
        segment_type = np.zeros(self.n)
        segment_type[type_tmp == "Straight"] = 0
        segment_type[type_tmp == "Left"] = 1
        segment_type[type_tmp == "Right"] = -1

        #Getting Position Data
        l = self.info.loc[:, "Length"]
        L = np.sum(l) #total length
        X = np.cumsum(l)  # end position of each segment
        XC = np.cumsum(l) - l / 2  # center position of each segment


        # Coarse Track Meshing
        j = 0  # index
        self.x = np.zeros(len(X) + np.sum(R==np.inf))   # preallocation
        self.r = np.zeros(len(X) + np.sum(R==np.inf))

        for i in range(len(X)): 
            if R[i] == np.inf:  # end of straight point injection
                self.x[j] = X[i] - l[i]
                self.x[j + 1] = X[i]
                j += 2
            else:  # circular segment 
                self.x[j] = XC[i]
                self.r[j] = segment_type[i] / R[i]
                j += 1



        # saving coarse results; these are the values we interpolate to get a mesh
        unique_indices = np.unique(self.x, return_index=True)[1]
        xx = self.x[unique_indices]
        rr = self.r[unique_indices]


        # New fine position vector; this is where we mesh the track
        if np.floor(L) < L:  # check for injecting last point
            self.x = np.concatenate([np.arange(0, np.floor(L), mesh_size), [L]])
        else:
            self.x = np.arange(0, np.floor(L), mesh_size)

        # Distance step vector
        self.dx = np.diff(self.x)
        self.dx = np.concatenate([self.dx, [self.dx[-1]]])

        # Number of mesh points
        self.n = len(self.x)

        # Fine curvature vector; interpolation of unique radii at all unique positions
        self.r = pchip_interpolate(xx, rr, self.x)

        # Fine turn direction vector
        t = np.sign(self.r)

        if self.grip == 0:
            self.factor_grip = np.ones(self.n)
        else:
            self.factor_grip = np.ones(self.n) * self.grip
            
        self.bank = np.zeros(self.n)
        self.incl = np.zeros(self.n)


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

#plot_track(x, r)