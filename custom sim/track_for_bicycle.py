import pandas as pd
import numpy as np
from scipy.interpolate import interp1d



def read_info(workbook_file, sheet_name=1, start_row=2, end_row=10000, cols="A:C"):
    # Setup the Import Options
    opts = pd.io.excel.read_excel(workbook_file, sheet_name, header=None, skiprows=start_row-1, nrows=end_row-start_row+1, usecols=cols)
    
    # Specify column names
    opts.columns = ["Type", "Length", "Corner Radius"]
    
    return opts

#****************Settings and data uploading
# Mode
log_mode = 'speed & latacc'
mode = 'shape data'

# meshing
mesh_size = 1.25 # [m]


# Track excel file selection
base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\track_files\\"

filename = 'Michigan 2014.xlsx'

filename = base_name + filename
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


#TODO: currently we have an issue where the curvature becomes nonphysical if we end on a corner instead of a straight
#Curvature extrapolates beyond the halfway point and exceeds the real curvature of the arc
#We should hard set the curvature about 10% into and 10% out of each arc
# editing points to a nice format; TODO -> better understand**********************
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
    x = np.concatenate([np.arange(0, np.floor(L), mesh_size*finer_mesh_multiplier), [L]])
else:
    x = np.arange(0, np.floor(L), mesh_size*finer_mesh_multiplier)

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


def reload(trackfile):
    global x
    global r
    global n
    global dx
    global bank
    global incl
    global factor_grip
    
    
    # Track excel file selection
    base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\track_files\\"
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


    # New fine position vector; this is where we mesh the track
    if np.floor(L) < L:  # check for injecting last point
        x = np.concatenate([np.arange(0, np.floor(L), mesh_size*finer_mesh_multiplier), [L]])
    else:
        x = np.arange(0, np.floor(L), mesh_size*finer_mesh_multiplier)

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

    factor_grip = np.ones(n)
    bank = np.zeros(n)
    incl = np.zeros(n)


def reload(trackfile):
    global x
    global r
    global n
    global dx
    global bank
    global incl
    global factor_grip
    
    
    # Track excel file selection
    base_name = "C:\\Users\\EJDRO\\OneDrive\\Documents\\GitHub\\FEBSim\\custom sim\\track_files\\"
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


    # New fine position vector; this is where we mesh the track
    if np.floor(L) < L:  # check for injecting last point
        x = np.concatenate([np.arange(0, np.floor(L), mesh_size*finer_mesh_multiplier), [L]])
    else:
        x = np.arange(0, np.floor(L), mesh_size*finer_mesh_multiplier)

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

    factor_grip = np.ones(n)
    bank = np.zeros(n)
    incl = np.zeros(n)
    
#New stuff for testing bicycle model
import matplotlib.pyplot as plt


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