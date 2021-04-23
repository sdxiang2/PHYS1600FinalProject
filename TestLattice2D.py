import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from Lattice2D import Lattice2D

lattice_shape = (101, 101)

# Testing out Lattice2D with a gaussian
def gaussian(xy_list):
    x = xy_list[0]
    y = xy_list[1]
    # amplitude 1
    a = 1.
    # peak centered on lattice
    x0 = lattice_shape[0]//2
    y0 = lattice_shape[1]//2
    # st.dev. of both x and y
    c = 10.
    gauss = np.zeros(lattice_shape)
    for i in range(lattice_shape[0]):
        for j in range(lattice_shape[1]):
            gauss[i][j] =  a * np.exp(-((x[i][j] - x0)**2 / (2 * c**2) + (y[i][j] - y0)**2 / (2 * c**2)))
    return gauss

coord = np.meshgrid(range(lattice_shape[0]), range(lattice_shape[1])) # two 2D arrays representing x and y coordinates
lat0 = gaussian(coord)

# Making sure the gaussian looks alright
fig = plt.figure()
ax1 = fig.add_subplot(111)
m1 = ax1.pcolormesh(coord[0], coord[1], lat0, cmap="inferno")
cbar = fig.colorbar(m1)
plt.show()

# Diffusing the gaussian
gausstest = Lattice2D(lat0,tmax=100)
gausstest.runToEq()
numFrames = len(gausstest.lat_series)
gausstest.plotFrameN(0)
gausstest.plotFrameN(numFrames//2)
gausstest.plotFinal()

def naguomo(u, a):
    return u * (1-u) * (u-a)

''' VERSION OF TRAVELLING WAVE WITHOUT JITTABLE STUFF TAKEN OUT
def travelling_wave(zeta, a):
    """Given a propagation angle zeta, builds a lattice with the corresponding
    initial condition and runs Lattice 2D on it.
    The x and y coordinates in reverse order because of how matricies are
    row-then-column indexed."""

    # INITIAL CONDITION SETUP GIVEN ZETA
    IC = np.zeros(lattice_shape)
    n = lattice_shape[1] # dimension in x
    m = lattice_shape[0] # dimension in y

    # identifying the index of the center of the lattice
    center = [m//2, n//2]

    def coord(j, i):
        """Takes an index (i,j) and converts it into a coordinate (center is origin).
        j will always refer to a y coordinate, and i will always refer to an x coordinate."""
        return j - center[0], i - center[1]

    # Finding the "+- infinity" on our graph given angle zeta
    # "+ infinity" is highest_pt and will be held at 1
    # "- infinity" is lowest_pt and will be held at 0
    highest_pt = None
    highest_pt_on_top = False
    closest_dist = np.infty
    for j in range(center[0], m): # checking side boundary in 1st quadrant
        y,x = coord(j, n-1)
        angle = np.arctan(y/x)
        dist = abs(angle - zeta)
        if dist < closest_dist:
            highest_pt = [j,n-1]
            closest_dist = dist
    for i in range(center[1]+1, n-1): # checking top boundary in 1st quadrant
        y,x = coord(m-1, i)
        angle = np.arctan(y/x)
        dist = abs(angle - zeta)
        if dist < closest_dist:
            highest_pt = [m-1, i]
            closest_dist = dist
            highest_pt_on_top = True
    lowest_pt = None
    if highest_pt_on_top:
        lowest_pt = [0, center[1] - coord(highest_pt[0],highest_pt[1])[1]]
    else:
        lowest_pt = [center[0] - coord(highest_pt[0],highest_pt[1])[0], 0]


    # Checking to see whether highest/lowest points are correct
    IC[highest_pt[0]][highest_pt[1]] = 3
    IC[lowest_pt[0]][lowest_pt[1]] = -3
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    m1 = ax1.pcolormesh(IC, cmap="inferno")
    cbar = fig.colorbar(m1)
    plt.show() # this puts them in the right place


    # Building initial condition
    for i in range(center[1]+1, n):
        for j in range(center[0], m):
            if j > highest_pt[0] - (1/np.tan(zeta)) * (i - highest_pt[1]):
                IC[j][i] = 1
    IC[highest_pt[0]][highest_pt[1]] = 1
    IC[lowest_pt[0]][lowest_pt[1]] = 0


    # Checking to see whether IC is correct
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    m1 = ax1.pcolormesh(IC, cmap="inferno")
    cbar = fig.colorbar(m1)
    plt.show() # this flips them


    lat_wave = Lattice2D(lat0=IC, R=lambda u: naguomo(u, a), D=0.1, tmax=10000)
    lat_wave.runToEq()


    # Plotting some frames
    numFrames = len(lat_wave.lat_series)
    lat_wave.plotFrameN(0)
    lat_wave.plotFrameN(numFrames//8)
    lat_wave.plotFrameN(numFrames//4)
    lat_wave.plotFrameN(numFrames//2)
    lat_wave.plotFinal()

    return 1/lat_wave.t # Proxy for speed c_zeta
'''

# Trying to move boundary point finding and IC creation out of travelling_wave for jit compilation
@jit(nopython=True)
def build_IC(n, m, center, zeta):

    def coord(j, i):
        """Takes an index (i,j) and converts it into a coordinate (center is origin).
        j will always refer to a y coordinate, and i will always refer to an x coordinate."""
        return j - center[0], i - center[1]

    highest_pt = None
    highest_pt_on_top = False
    closest_dist = np.infty
    for j in range(center[0], m): # checking side boundary in 1st quadrant
        y,x = coord(j, n-1)
        angle = np.arctan(y/x)
        dist = abs(angle - zeta)
        if dist < closest_dist:
            highest_pt = [j,n-1]
            closest_dist = dist
    for i in range(center[1]+1, n-1): # checking top boundary in 1st quadrant
        y,x = coord(m-1, i)
        angle = np.arctan(y/x)
        dist = abs(angle - zeta)
        if dist < closest_dist:
            highest_pt = [m-1, i]
            closest_dist = dist
            highest_pt_on_top = True
    lowest_pt = None
    if highest_pt_on_top:
        lowest_pt = [0, center[1] - coord(highest_pt[0],highest_pt[1])[1]]
    else:
        lowest_pt = [center[0] - coord(highest_pt[0],highest_pt[1])[0], 0]

    IC = np.zeros((m,n))
    for i in range(center[1]+1, n):
        for j in range(center[0], m):
            if j > highest_pt[0] - (1/np.tan(zeta)) * (i - highest_pt[1]):
                IC[j][i] = 1
    IC[highest_pt[0]][highest_pt[1]] = 1
    IC[lowest_pt[0]][lowest_pt[1]] = 0

    return IC, highest_pt, lowest_pt

def travelling_wave(zeta, a, lattice_shape):
    n = lattice_shape[1] # dimension in x
    m = lattice_shape[0] # dimension in y

    # identifying the index of the center of the lattice
    center = np.array([m//2, n//2])

    IC, hp, lp = build_IC(n, m, center, zeta)

    print("zeta = " + str(zeta))
    lat_wave = Lattice2D(lat0=IC, highest_pt=hp, lowest_pt=lp, R=lambda u: naguomo(u, a), D=0.1, tmax=10000)
    lat_wave.runToEq()

    #'''
    # Plotting some frames
    numFrames = len(lat_wave.lat_series)
    lat_wave.plotFrameN(0)
    lat_wave.plotFrameN(numFrames//8)
    lat_wave.plotFrameN(numFrames//4)
    lat_wave.plotFrameN(numFrames//2)
    lat_wave.plotFinal()
    #'''

    print("speed = " + str(1/lat_wave.t ))
    return 1/lat_wave.t # Proxy for speed c_zeta

travelling_wave(np.pi/3, 0.4, (81,81))

# Plotting propagation speed vs. zeta for a = 0.475
zetas = np.arange(1, 90, 1)
zetas = zetas * (np.pi/180) # converting zetas from degrees to radians
zetas_copy = zetas[:]
speeds = []
for zeta in zetas:
    speed = travelling_wave(zeta, 0.475, (41, 41))
    speeds.append(speed)
speeds = np.array(speeds)

zetas = np.delete(zetas, np.where(speeds > 1))
speeds = np.delete(speeds,np.where(speeds > 1))
print(zetas)
print(speeds)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(zetas, speeds)
ax.grid(True)
plt.show()

# Testing boundary point finder on a very simple case
# coord2 = np.meshgrid(np.linspace(0,2,3,endpoint=True), np.linspace(0,2,3,endpoint=True))
#lat0_2 = np.zeros((3,3))
#lat0_2[1][1] = 1
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# m2 = ax2.pcolormesh(coord2[0], coord2[1], lat0_2, cmap="RdBu")
# cbar = fig2.colorbar(m2)
# plt.show()

#testlat2 = Lattice2D(lat0_2, lambda u: 0, tmax=1000)
#testlat2.runToEq()
#testlat2.plotFrameN(0)
#testlat2.plotFinal()
#testlat2.plotAnimation()
