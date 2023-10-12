import math
import matplotlib.pyplot as plt

# reset system, initialize track points
from TrackDataSimple import Length, Radius, Type

nPoints = len(Length)
x0 = 0
x1 = 0
y0 = 0
y1 = 0
angle = 0

# create list of all points and segment data
heading=[nPoints]
turn = [nPoints]
ang = [nPoints]
x = [nPoints]
y = [nPoints]

# loop through each section of lap
for n in range(nPoints):
    L = Length[n]
    R = Radius[n]
    DIR = Type[n]
    
    # turn the list of sections into fixed points for the start/end of each segment, 
    # arcs will be plotted between each consecutive point later
    if DIR == "Straight":
        x1 = L * math.cos(math.radians(angle)) + x0
        y1 = L * math.sin(math.radians(angle)) + y0
        headingAng = angle
        turnType = 0
    elif DIR == "Left":
        # secant is direct path from start to end of segment
        secant = 2 * R * math.sin(L / (2 * R))
        # heading angle is calculated as the tangent of the arc at the end of each segment
        headingAng = L * 180 / (math.pi * R) + angle
        # turning angle is calculated as the angle between the "entrance angle" and the secant
        turnAng = L * 90 / (math.pi * R) + angle
        # turn type tells us whether it is CW or CCW, used in plotArc function
        turnType = L / R
        angle = headingAng
        # coordinates for the endpoint of each segement
        x1 = secant * math.cos(math.radians(turnAng)) + x0
        y1 = secant * math.sin(math.radians(turnAng)) + y0
    elif DIR == "Right":
        secant = 2 * R * math.sin(L / (2 * R))
        headingAng = -L * 180 / (math.pi * R) + angle
        turnAng = -L * 90 / (math.pi * R) + angle
        turnType = L / R
        angle = headingAng
        x1 = secant * math.cos(math.radians(turnAng)) + x0
        y1 = secant * math.sin(math.radians(turnAng)) + y0
    else:
        D = 0
    
    # create list of all points and segment data
    heading[n] = headingAng
    turn[n] = turnType
    ang[n] = angle
    x[n] = x1
    y[n] = y1
    x0 = x1
    y0 = y1

# bookeeping for array indexing, nothing important
x = [0] + x
y = [0] + y

fig, ax = plt.subplots()


# plot arcs in between consecutive points using the plotArc subfunction
for i in range(nPoints):
    a = [x[i], y[i]]
    b = [x[i+1], y[i+1]]
    dir = Type[i]
    r = Radius[i]
    ang = turn[i]
    theta1 = math.degrees(math.atan2(a[1], a[0]))
    theta2 = math.degrees(math.atan2(b[1], b[0]))
    dtheta = theta2 - theta1
    if dir == "Left":
        dtheta = -dtheta
    arc = plt.Arc((a[0], a[1]), 2*r, 2*r, theta1, theta2)