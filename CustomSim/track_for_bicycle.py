import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def read_info(workbook_file, sheet_name=1, start_row=2, end_row=10000, cols="A:C"):
    # Setup the Import Options
    opts = pd.io.excel.read_excel(workbook_file, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)
    
    # Specify column names
    opts.columns = ["Type", "Length", "Corner Radius"]
    
    return opts


# Track excel file selection
base_name = "track_files/"
filename = 'track_files/Michigan_2022_AutoX.xlsx' # Michigan_2022_AutoX
filename = 'track_files/test_track.xlsx'


class Track:
    def __init__(self, trackfile, mesh_size = 0.25):
        # Initialize the track by loading the file
        self.mesh_size = mesh_size
        self.load_track(trackfile)

    def load_track(self, filename):
        info = read_info(filename,'Shape')

        #Getting Curvature
        R = info.loc[:, "Corner Radius"] #0 or NaN on straights, otherwise a float
        R = np.nan_to_num(R)
        R = R.astype(float)
        R[R==0] = np.inf
        r2 = np.reciprocal(R)
        n = len(R)

        #Getting type
        type_tmp = info.loc[:, "Type"]
        segment_type = np.zeros(n)
        segment_type[type_tmp == "Straight"] = 0
        segment_type[type_tmp == "Left"] = 1
        segment_type[type_tmp == "Right"] = -1

        #Getting Position Data
        l = info.loc[:, "Length"]
        L = np.sum(l) #total length
        X = np.cumsum(l)  # end position of each segment
        XC = np.cumsum(l) - l / 2  # center position of each segment

        j = 0  # index
        x = np.zeros(len(X) + len(R)) 
        r = np.zeros(len(X) + len(R))

        for i in range(len(X)):
            if R[i] == np.inf:  # end of straight point injection
                x[j] = X[i] - l[i]
                x[j + 1] = X[i]
                j += 2
            else:  # circular segment 
                tol = 1e-1
                x[j] = X[i] - l[i]*(1-tol) #set the curvature at 10% into the curve
                x[j + 1] = X[i] - l[i]*tol #set the curvature 10% out of the curve
                r[j] = segment_type[i] / R[i]
                r[j+1] = segment_type[i] / R[i]
                j += 2



        # saving coarse results; these are the values we interpolate to get a mesh
        unique_indices = np.unique(x, return_index=True)[1]
        xx = x[unique_indices]
        rr = r[unique_indices]


        finer_mesh_multiplier = 1

        # New fine position vector; this is where we mesh the track
        if np.floor(L) < L:  # check for injecting last point
            x = np.concatenate([np.arange(0, np.floor(L), self.mesh_size*finer_mesh_multiplier), [L]])
        else:
            x = np.arange(0, np.floor(L), self.mesh_size*finer_mesh_multiplier)

        # Distance step vector
        dx = np.diff(x)
        dx = np.concatenate([dx, [dx[-1]]])

        # Number of mesh points
        n = len(x)

        # Fine curvature vector; interpolation of unique radii at all unique positions
        r_func = interp1d(xx, rr, fill_value="extrapolate")
        r = r_func(x)

        # Fine turn direction vector
        t = np.sign(r)

        #Track formatting:
        fullness =  len(dx)
        segments = []
        for i in range(fullness):
            segments.append([dx[i], r[i]])


        #All track information should be encoded in x, r
        self.x = x
        self.r = r
        self.curvatures = r

        # Extra stuff (currently unsused)
        self.segments = np.asarray(segments)
        self.track_widths = np.ones(len(dx))*4.0

        self.factor_grip = np.ones(n)
        self.bank = np.zeros(n)
        self.incl = np.zeros(n)


    def reload(self, filename):
        self.load_track(filename)

    # separates curvatures array into multiple parts
    def split(self, parts):
        return np.array_split(self.curvatures, parts)
    
    # Auto-partition; 
    # S = cumsum(x)
    # partition S by 200m; find nearest i where r[i] = 0 such that 250m is max distance
    # maybe loop back through, find spacings of all r[i] = 0

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
    def plot_car(self, trackfile, datafile, mesh_size = 0.25):
        # Initialize arrays for plotting
        c = pd.read_excel('sims_logs/'+datafile)
        s, kappa = self.x, self.r
        x_pos = np.zeros_like(s)  # x positions
        y_pos = np.zeros_like(s)  # y positions
        car_x_pos = np.zeros_like(s)  # car x positions
        car_y_pos = np.zeros_like(s)  # car y positions
        calc_car_x_pos = np.zeros_like(s)  # car x positions
        calc_car_y_pos = np.zeros_like(s)  # car y positions
        theta = np.zeros_like(s)  # Heading angles
        time = c['time']
        velocity = c['v']
        beta = c['beta']
        xi = c['xi']
        assert len(s) == len(c), f"Car data and Track data do not match, check mesh size and track name, len(s) = {len(s)}, len(c) = {len(c)}"
        # Integrate to get the positions and heading angle
        for i in range(1, len(s)):

            dtheta = kappa[i-1] * (s[i] - s[i-1])
            theta[i] = theta[i-1] + dtheta

            dx = np.cos(theta[i-1]) * (s[i] - s[i-1])
            dy = np.sin(theta[i-1]) * (s[i] - s[i-1])
        
            x_pos[i] = x_pos[i-1] + dx
            y_pos[i] = y_pos[i-1] + dy
            
            if i < len(c):
                if time[i]>time[i - 1]:
                    t = time[i] - time[i - 1]
                else:
                    t = time[i]
                v = velocity[i]
                
                angle = theta[i] + beta[i] + xi[i]
                dv = velocity[i] - velocity[i - 1]

                calc_car_x_pos[i] = np.cos(angle) * calc_car_x_pos[i] + v * t + 1/2 * dv * t
                calc_car_y_pos[i] = np.sin(angle) * calc_car_x_pos[i] + v * t + 1/2 * dv * t


        # Plot the track
        plt.figure(figsize=(10, 5))
        plt.plot(x_pos, y_pos, color = 'black', label='Track')

        plt.plot(calc_car_x_pos, calc_car_y_pos, color = 'blue', label = 'Car')

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Track Plot')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot(self):
        self.plot_track(self.x, self.r)