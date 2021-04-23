import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.interactive(True)


class Lattice2D(object):
    """Class that describes dynamics on a 2-D lattice"""
    def __init__(self, lat0, lowest_pt=None, highest_pt=None, R=lambda u: 0, D=1, tmax=100, dt=0.01):
        self.lat0 = lat0
        self.lowest_pt = lowest_pt
        self.highest_pt = highest_pt
        self.R = R
        self.D = D
        self.tmax = tmax
        self.t = 0.0
        self.dt = dt
        self.delta = dt/1000.
        self.shape = lat0.shape
        self.maxscale = np.amax(lat0)


    def runToEq(self):
        """Runs the lattice until it reaches equilibrium, creating self.lat_series"""
        def distance(self, lat1, lat2):
            """Computes the distance between two lattices"""
            if (lat1.shape == lat2.shape):
                return np.linalg.norm(lat1 - lat2)
            else:
                raise RuntimeError("Distance function failed: lattices of unequal size")

        old = np.full(self.shape, np.inf)
        new = self.lat0
        self.lat_series = [new.copy()]

        while (distance(self, new, old) > self.delta) and (self.t < self.tmax):
            old = new.copy()
            # Array slicing version:
            new[1:-1, 1:-1] = self.dt * (self.D * (old[1:-1, 2:] + old[1:-1, :-2] + old[2:,1:-1] + old[:-2, 1:-1] - 4 * old[1:-1, 1:-1]) + self.R(old[1:-1, 1:-1])) + old[1:-1, 1:-1]
            # Setting edge nodes equal to adjacent interior points
            new[-1, 1:-1] = new[-2, 1:-1] # top
            new[0, 1:-1] = new[1, 1:-1] # bottom
            new[1:-1, -1] = new[1:-1, -2] # right
            new[1:-1, 0] = new[1:-1, 1] # left
            # Setting corner nodes equal to diagonal interior points
            new[0,0] = new[1,1]
            new[0,-1] = new[1,-2]
            new[-1,0] = new[-2,1]
            new[-1,-1] = new[-2,-2]
            # Enforcing boundary conditions
            if(self.lowest_pt != None):
                new[self.highest_pt[0],self.highest_pt[1]] = 1
                new[self.lowest_pt[0],self.lowest_pt[1]] = 0

            self.lat_series.append(new.copy())
            self.t += self.dt
        print("Final time: ", self.t)
        #print("Length of lattice series: ", len(self.lat_series))
        self.lat_series = np.array(self.lat_series)


    def plotFrameN(self, N):
        """Plots lattice at Nth time step"""
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        m1 = ax1.pcolormesh(self.lat_series[N], vmin=0, vmax=self.maxscale, cmap="inferno")
        cbar = fig.colorbar(m1)
        plt.show()


    def plotFinal(self):
        """Plots final lattice"""
        self.plotFrameN(len(self.lat_series) - 1)


    #This is Broken
    def plotAnimation(self):
        """Animates the lattice as it evolves in time"""
        fig, ax = plt.subplots()
        #plt.ion()
        plt.show()
        X, Y = np.meshgrid(range(self.shape[0]), range(self.shape[1]))
        cax = ax.pcolormesh(self.lat_series[0], cmap="RdBu")

        for i in range(self.lat_series.shape[0]):
            self.plotFrameN2(i, fig, ax)
