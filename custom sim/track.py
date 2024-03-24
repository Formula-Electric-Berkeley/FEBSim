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
filename = 'Michigan 2014.xlsx'
info = read_info(filename,'Shape')



#Getting Curvature
R = info.loc[:, "Corner Radius"] #0 or NaN on straights, otherwise a float
R = np.nan_to_num(R)
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

# editing points to a nice format; TODO -> better understand**********************
for i in range(len(X)):
    if R[i] == np.inf:  # end of straight point injection
        x[j] = X[i] - l[i]
        x[j + 1] = X[i]
        j += 2
    else:  # circular segment center
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
r_func = interp1d(xx, rr,kind='cubic', fill_value='extrapolate')
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




#print(segments)