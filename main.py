# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:40:37 2020

@author: Max Sours
"""
import numpy as np
import matplotlib.pyplot as plt

from init_conds import *
from integration_methods import semi_implicit_euler, verlet, RKF

# I implement 3-D coordinates, but everything is done in a 2D plane

def accel(pos, m):
    """
    Calculate acceleration of each particle

    Parameters
    ----------
    pos : nx3 float array
        Collection of position vectors for the n particles
    m : nx1 float array
        Masses of each particle

    Returns
    -------
    nx3 float array
        Collection of acceleration vectors for the n particles

    """
    n = m.size
    acc = 0 * pos
    for i in range(n):
        x = pos[i, :] #Extract current particle
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xj = pos[mask, :] #Get arrays for all other particles
        mj = m[mask, :]
        diff = xj - x
        # Calculate net acceleration of current particle
        acc[i, :] = G * np.sum(mj * diff / np.linalg.norm(diff, axis = 1).reshape(n-1, 1) ** 3, axis = 0)
    return acc

def potential(pos, m):
    """
    Calculate gravitational potential of the system

    Parameters
    ----------
    pos : nx3 float array
        Collection of position vectors for the n particles
    m : nx1 float array
        Masses of each particle
    
    Returns
    -------
    float
        Potential of the system
    """
    n = m.size
    pot = 0
    for i in range(n):
        x = pos[i, :] #Extract current particle
        mask = np.ones(n, dtype=bool)
        # Get rid of current particle and all particles before
        # it. (Since the potential between current particle
        # and particles before it have already been calculated.)
        mask[:i+1] = False
        xj = pos[mask, :]
        mj = m[mask, :]
        # Calculate the net potential between current particle and particles after it
        pot -= G * m[i] * np.sum(mj / np.linalg.norm(x - xj, axis = 1).reshape(n-i-1, 1))
    return pot

def energy(x, v, m):
    """
    Given masses, simulated positions and velocites,
    calculate total energy of the system.
    (Should be invariant.)

    Parameters
    ----------
    x : nx3xT float array
        Simulated positions
    v : nx3xT float array
        Simulated velocities
    m : nx1 float array
        Masses

    Returns
    -------
    Tx1 float array
        Energy at each timestep

    """
    _, _, T = x.shape
    KE = np.zeros(T)
    PE = np.zeros(T)
    s = np.linalg.norm(v, axis = 1)
    for t in range(T):
        # Calculate kinetic energy
        KE[t] = 0.5 * np.sum(m.flatten() * s[:, t] ** 2)
        # Calculate potential energy
        PE[t] = potential(x[:, :, t], m)
    return KE + PE, KE, PE

def relative_coord(b1, b2):
    """
    Output coordinates of b1 relative to b2's orbit around
    the barycenter

    Parameters
    ----------
    b1 : 3xT float array
        Position of b1
    b2 : 3xT float array
        Position of b2

    Returns
    -------
    3xT float array
        Transformed coordinates of b1

    """
    _, T = b1.shape
    result = np.zeros(b1.shape)
    for t in range(T):
        # Calculate angle of b2 on its orbit
        theta = np.arctan2(b2[1, t] , b2[0, t])
        # Rotate clockwise by the calculated angle
        result[:, t] = np.dot(rotz(-theta), b1[:, t])
    return result

if __name__ == "__main__":
    # Control which parts will run
    part_1b = 1 # 1 = run, 0 = don't run - equivalent to True and False)
    part_1c = 1
    part_1d = 1
    part_2a = 1
    part_2b = 1

    # Calculate the initial condiditons for part 1
    n, m, p, v0 = sun_asteroid()
    labels = ["Sun", "Asteroid"]
    if part_1b:
        # Simulate the system
        t, x, v = semi_implicit_euler(p, v0, m, accel, 300 * s_in_day, 3600)
        # Plot positions of the planets
        for i in range(n):
            plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
        plt.title("Planetary Positions")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.legend()
        plt.show()
        # Scale time by days
        t = t / s_in_day
        # Plot asteroid distance to barycenter over time
        plt.plot(t, np.linalg.norm(x[1, :, :], axis = 0))
        plt.title("Asteroid Distance Over Time")
        plt.xlabel("Time (days)")
        plt.ylabel("m")
        plt.show()
        # Calculate and plot energy over time
        E, _, _ = energy(x, v, m)
        plt.plot(t, E)
        plt.title("Energy vs. Time")
        plt.xlabel("Time (days)")
        plt.ylabel("Energy (J)")
        plt.show()
        
    if part_1c:
        # Simulate the system
        t, x, v = verlet(p, v0, m, accel, 300 * s_in_day, 3600)
        # Plot positions of the planets
        for i in range(n):
            plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
        plt.title("Planetary Positions")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.legend()
        plt.show()
        # Scale time by days
        t = t / s_in_day
        # Plot asteroid distance to barycenter over time
        plt.plot(t, np.linalg.norm(x[1, :, :], axis = 0))
        plt.title("Asteroid Distance Over Time")
        plt.xlabel("Time (days)")
        plt.ylabel("m")
        plt.show()
        # Calculate and plot energy over time
        E, _, _ = energy(x, v, m)
        plt.plot(t, E)
        plt.title("Energy vs. Time")
        plt.xlabel("Time (days)")
        plt.ylabel("Energy (J)")
        plt.show()
    
    if part_1d:
        # Simulate the system
        t, x, v = RKF(p, v0, m, accel, 300 * s_in_day, 2048, 100000, 0.1)
        # Plot positions of the planets
        for i in range(n):
            plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
        plt.title("Planetary Positions")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.show()
        # Scale time by days
        t = t / s_in_day
        # Plot asteroid distance to barycenter over time
        plt.plot(t, np.linalg.norm(x[1, :, :], axis = 0))
        plt.title("Asteroid Distance Over Time")
        plt.xlabel("Time (days)")
        plt.ylabel("m")
        plt.show()
        # Calculate and plot energy over time
        E, _, _ = energy(x, v, m)
        plt.plot(t, E)
        plt.title("Energy vs. Time")
        plt.xlabel("Time (days)")
        plt.ylabel("Energy (J)")
        plt.show()
        # Plot time
        plt.plot(t)
        plt.title("Time vs. Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Time (days)")
        plt.show()
    
    # Store each lagrange point function in a dictionary
    # so they can be easily called upton
    l_inits = {}
    l_inits[1] = sun_earth_L1
    l_inits[2] = sun_earth_L2
    l_inits[3] = sun_earth_L3
    l_inits[4] = sun_earth_L4
    l_inits[5] = sun_earth_L5
    labels = ["Sun", "Earth", "Spacecraft"]
    if part_2a:
        # Do L1, L2, L3
        for j in range(1, 4):
            # Get intial conditions for appropriate L point
            n, m, p, v0 = l_inits[j]()
            # Simulate the system
            t, x, v = RKF(p, v0, m, accel, 300 * s_in_day, 2048, 100000, 0.1)
            # Plot positions in Solar (Inertial) Reference Frame
            for i in range(n):
                plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
            plt.title("L"+str(j)+" - Solar Frame of Reference")
            plt.xlabel("m")
            plt.ylabel("m")
            plt.legend()
            plt.show()
            # Calculate coordinates of spacecraft in Earth's FoR
            D = relative_coord(x[2, :, :], x[1, : ,:])
            # Plot position of spacecraft and Earth in Earth's (Rotating) FoR
            plt.plot(D[0, :], D[1, :], label = "Spacecraft")
            plt.scatter([au], [0], label = "Earth")
            plt.title("L"+str(j)+" - Earth's Frame of Reference")
            plt.xlabel("m")
            plt.ylabel("m")
            plt.legend()
            plt.show()
            
    if part_2b:
        # Do L4, L5
        for j in range(4, 6):
            # Get intial conditions for appropriate L point
            n, m, p, v0 = l_inits[j]()
            # Simulate the system
            t, x, v = RKF(p, v0, m, accel, 300 * s_in_day, 2048, 100000, 0.1)
            # Plot positions in Solar (Inertial) Reference Frame
            for i in range(n):
                plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
            plt.title("L"+str(j)+" - Solar Frame of Reference")
            plt.xlabel("m")
            plt.ylabel("m")
            plt.legend()
            plt.show()
            # Calculate coordinates of spacecraft in Earth's FoR
            D = relative_coord(x[2, :, :], x[1, : ,:])
            # Plot position of spacecraft and Earth in Earth's (Rotating) FoR
            plt.plot(D[0, :], D[1, :], label = "Spacecraft")
            plt.scatter([au], [0], label = "Earth")
            plt.title("L"+str(j)+" - Earth's Frame of Reference")
            plt.xlabel("m")
            plt.ylabel("m")
            plt.legend()
            plt.show()
    