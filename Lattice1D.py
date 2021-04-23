import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt


class Lattice1D(object):
    """Class that describes dynamics on a 1-D lattice"""
    def __init__(self, lat0, bc, R=lambda u: 0, D=1, tmax=1000, dt=0.01):
        self.lat0 = lat0
        self.R = R
        self.left_bc = bc[0]
        self.right_bc = bc[1]
        self.D = D
        self.tmax = tmax
        self.dt = dt
        self.delta = dt / 1000
        self.n = len(lat0)


    def dirichlet(self, lat):
        """Imposes Dirichlet boundary conditions on a lattice"""
        lat[0] = self.left_bc
        lat[-1] = self.right_bc


    def runToEq(self):
        """Runs the lattice until it reaches equilibrium, creating self.lat_series"""

        def distance(self, lat1, lat2):
            """Computes the distance between two lattices"""
            sum = 0
            if (len(lat1) == len(lat2)):
                for i in range(len(lat1)):
                    sum += (lat1[i] - lat2[i])**2
            else:
                raise RuntimeError("Distance function failed: lattices of unequal length")
            return sqrt(sum)

        old_lat = np.full(self.n, -np.inf)
        new_lat = self.lat0
        self.lat_series = [new_lat.copy()]

        t = 0.0
        while (distance(self, new_lat, old_lat) > self.delta) and (t < self.tmax):
            old_lat = new_lat.copy()
            for i in range(1,self.n-1):
                new_lat[i] = self.dt * (self.D * (old_lat[i-1] + old_lat[i+1] - 2*old_lat[i])
                                        + self.R(old_lat[i])) + old_lat[i]
            self.dirichlet(new_lat)
            self.lat_series.append(new_lat.copy())
            t += self.dt

        print("Reached equilibrium at t = ", t)
        self.finalTime = t
        self.lat_series = np.array(self.lat_series)


    def plotFinal(self):
        """Plots final lattice"""
        x = range(self.n)
        plt.plot(x, self.lat_series[-1], marker='o')
        plt.xlabel("Lattice index")
        plt.ylabel("Equilibrium quanitity at index")
        plt.ylim(0, 1.1)
        plt.show()


    def plotFrameN(self, N):
        """Plots lattice at Nth time step"""
        x = range(self.n)
        plt.plot(x, self.lat_series[N], marker='o')
        plt.xlabel("Lattice index")
        plt.ylabel("Equilibrium quanitity at index")
        plt.ylim(0, 1.1)
        plt.show()

    def plotAnimation(self):
        """Animates the lattice as it evolves in time"""

        fig, ax = plt.subplots()
        ax.set_xlabel("Lattice position")
        x = np.array(range(self.n))
        line, = ax.plot(x, self.lat_series[0], marker='o')

        def animate(i):
            if (i < self.lat_series.shape[0]):
                line.set_ydata(self.lat_series[i])  # update the data.
                return line,
            return self.lat_series[-1]

        anim = animation.FuncAnimation(
            fig, animate, interval=20, blit=True, save_count=50)

        plt.show()
