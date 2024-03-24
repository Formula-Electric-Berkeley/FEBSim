import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import track as tr

def tile_track(segments, starting_angle = 0):
    '''
    segments passed should be a 2 dimensional list with items in format [segment length, curvature]
    
    delta_d is the mesh length (arbitrary now)
    If r > 0, we're curving left; if r < 0, we go right

    starting_angle is relative to the horizontal.
    '''

    pos_x = 0
    pos_y = 0
    theta = starting_angle # Initial angle that the track starts at, should just be 0 most of the time but still. 

    snapshots = []

    complete = False
    current_segment = 0
    
    for i in range(len(segments)):
        forward_distance, curvature = segments[i]

        if np.abs(curvature) <= 10.0**-5: # Straight 
            pos_x += forward_distance * np.cos(theta)
            pos_y += forward_distance * np.sin(theta)

        else: # Curve - this relies on weird geometry  i did on a chalkboard but i can make a nice drawing when i have the time if anyone wants proof this works 
            radius = 1/curvature
            d_theta = forward_distance / np.abs(radius)
            
            if radius < 0: #Right
                pos_x -= radius * (np.sin(theta + d_theta) - np.sin(theta))
                pos_y += radius * (np.cos(theta + d_theta) - np.cos(theta))
                
                theta -= d_theta

            if radius > 0: #left
                pos_x += radius * (np.sin(theta + d_theta) - np.sin(theta))
                pos_y -= radius * (np.cos(theta + d_theta) - np.cos(theta))
                
                theta += d_theta

            snapshots.append([pos_x, pos_y, theta])
        
    return snapshots

def closest_perpendicular_distance_to_spline(X, Y, spline_points):
    # Build a KD-tree for efficient nearest neighbor search
    kdtree = cKDTree(spline_points[:, :2])
    
    # Find the closest point on the spline
    distance, index = kdtree.query([X, Y])
    closest_point = spline_points[index]
    
    # Calculate the perpendicular distance
    dx = X - closest_point[0]
    dy = Y - closest_point[1]
    perpendicular_distance = np.sqrt(dx**2 + dy**2)
    
    return perpendicular_distance, closest_point

def perpendicular_Plotter():
    # Example usage
    spline_points = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]
    # Create a cubic spline from the given points
    spline = CubicSpline(*zip(*spline_points), bc_type='not-a-knot')

    # Generate points along the spline for interpolation
    t = np.linspace(0, 1, 1000)
    spline_points = np.column_stack((spline(t), t))

    X, Y = 2.5, 0.5
    distance, closest_point = closest_perpendicular_distance_to_spline(X, Y, spline_points)
    print(distance)

    spline_points = np.array(spline_points)

    # Plot the spline
    plt.figure(figsize=(10, 6))
    plt.plot(spline_points[:, 0], spline_points[:, 1], label='Spline')

    # Plot the object position
    plt.scatter(X, Y, color='red', label='Object Position')

    # Plot the closest point on the spline
    plt.scatter(closest_point[0], closest_point[1], color='green', label='Closest Point on Spline')

    # Add a line connecting the object to the closest point on the spline
    plt.plot([X, closest_point[0]], [Y, closest_point[1]], '--', color='gray')

    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Closest Perpendicular Distance to Spline')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

delta_d = tr.mesh_size

snapshots = np.asarray(tile_track(tr.segments))

kappas = np.abs(tr.segments[:, 1])

#plt.scatter(snapshots[:, 0], snapshots[:, 1])
#plt.show()