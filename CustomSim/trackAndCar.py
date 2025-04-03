import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import pchip_interpolate



def read_info(workbook_file, sheet_name=1, start_row=2, end_row=10000, cols="A:C"):
    # Setup the Import Options
    opts = pd.io.excel.read_excel(workbook_file, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)
    
    # Specify column names
    opts.columns = ["Type", "Length", "Corner Radius"]
    
    return opts

def get_track_info(trackname, mesh_size = 0.25):
    #****************Settings and data uploading
    # Mode
    log_mode = 'speed & latacc'
    mode = 'shape data'


    # Track excel file selection
    filename = 'track_files/' + trackname
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
    x = np.zeros(len(X) + np.sum(R == np.inf)) 
    r = np.zeros(len(X) + np.sum(R == np.inf))


    # Coarse mesh of the track
    for i in range(len(X)):
        if R[i] == np.inf:  # end of straight point injection
            x[j] = X[i] - l[i]
            x[j + 1] = X[i]
            j += 2
        else:  # circular segment 
            x[j] = XC[i] 
            r[j] = segment_type[i] / R[i]
            j += 1



    # saving coarse results; these are the values we interpolate to get a mesh
    unique_indices = np.unique(x, return_index=True)[1]
    xx = x[unique_indices]
    rr = r[unique_indices]



    # New fine position vector; this is where we mesh the track
    if np.floor(L) < L:  # check for injecting last point
        x = np.concatenate([np.arange(0, np.floor(L), mesh_size), [L]])
    else:
        x = np.arange(0, np.floor(L), mesh_size)

    # Distance step vector
    dx = np.diff(x)
    dx = np.concatenate([dx, [dx[-1]]])

    # Number of mesh points
    n = len(x)

    # Fine curvature vector; interpolation of unique radii at all unique positions
    r = pchip_interpolate(xx, rr, x)

    # Fine turn direction vector
    t = np.sign(r)

    #All information should be encoded in x, r

    #Track formatting:
    fullness =  len(dx)
    segments = []
    for i in range(fullness):
        segments.append([dx[i], r[i]])

    segments = np.asarray(segments)
    track_widths = np.ones(len(dx))*4.0

    factor_grip = np.ones(n)
    bank = np.zeros(n)
    incl = np.zeros(n)
    info.config = 'Closed'

    curvatures = r
    return x, r


def reload(trackfile, grip = 0):
    global x
    global r
    global n
    global dx
    global bank
    global incl
    global factor_grip
    
    
    # Track excel file selection
    base_name = 'track_files\\'
    filename = base_name + trackfile
    info = read_info(filename,'Shape')


    #Getting Curvature
    R = info.loc[:, "Corner Radius"] #0 or NaN on straights, otherwise a float
    R = np.nan_to_num(R)
    R = R.astype(float)
    R[R==0] = np.inf
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


    # Coarse Track Meshing
    j = 0  # index
    x = np.zeros(len(X) + np.sum(R==np.inf))   # preallocation
    r = np.zeros(len(X) + np.sum(R==np.inf))

    for i in range(len(X)): 
        if R[i] == np.inf:  # end of straight point injection
            x[j] = X[i] - l[i]
            x[j + 1] = X[i]
            j += 2
        else:  # circular segment 
            x[j] = XC[i]
            r[j] = segment_type[i] / R[i]
            j += 1



    # saving coarse results; these are the values we interpolate to get a mesh
    unique_indices = np.unique(x, return_index=True)[1]
    xx = x[unique_indices]
    rr = r[unique_indices]


    # New fine position vector; this is where we mesh the track
    if np.floor(L) < L:  # check for injecting last point
        x = np.concatenate([np.arange(0, np.floor(L), mesh_size), [L]])
    else:
        x = np.arange(0, np.floor(L), mesh_size)

    # Distance step vector
    dx = np.diff(x)
    dx = np.concatenate([dx, [dx[-1]]])

    # Number of mesh points
    n = len(x)

    # Fine curvature vector; interpolation of unique radii at all unique positions
    r = pchip_interpolate(xx, rr, x)

    # Fine turn direction vector
    t = np.sign(r)

    if grip == 0:
        factor_grip = np.ones(n)
    else:
        factor_grip = np.ones(n)*grip
        
    bank = np.zeros(n)
    incl = np.zeros(n)
    
#New stuff for testing bicycle model
import matplotlib.pyplot as plt


def plot_track(trackname, outputname, mesh_size = 0.25):
    # Initialize arrays for plotting
    c = pd.read_excel('sims_logs/'+outputname)
    s, kappa = get_track_info(trackname, mesh_size)
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
        t = time[i] - time[i - 1]
        v = velocity[i]
        
        dtheta = kappa[i-1] * (s[i] - s[i-1])
        theta[i] = theta[i-1] + dtheta

        angle = theta[i] + beta[i] + xi[i]
        dv = velocity[i] - velocity[i - 1]

        dx = np.cos(theta[i-1]) * (s[i] - s[i-1])
        dy = np.sin(theta[i-1]) * (s[i] - s[i-1])
    
        x_pos[i] = x_pos[i-1] + dx
        y_pos[i] = y_pos[i-1] + dy

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

