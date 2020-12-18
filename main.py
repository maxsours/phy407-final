# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:40:37 2020

@author: Max Sours
"""
import numpy as np
import matplotlib.pyplot as plt

from init_conds import *
from integration_methods import semi_implicit_euler, verlet, RKF

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
        xj = pos[mask, :] #Get arrays for other particles
        mj = m[mask, :]
        diff = xj - x
        # Calculate acceleration of current particle
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
        mask[:i+1] = False
        xj = pos[mask, :] #Get arrays for other particles
        mj = m[mask, :]
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
        KE[t] = 0.5 * np.sum(m.flatten() * s[:, t] ** 2)
        PE[t] = potential(x[:, :, t], m)
    return KE + PE, KE, PE

def momentum(v, m):
    """
    Given masses, velocities over the simulation,
    calculate momentum over the whole simulation

    Parameters
    ----------
    v : nx3xT float array
        Simulated velocities
    m : nx1 float array
        Masses

    Returns
    -------
    Tx3 float array
        Momentum of system at each timestep

    """
    return np.sum(np.moveaxis(m * np.ones(v.shape), 1, 0) * v, axis = 0)

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
        theta = np.arctan2(b2[1, t] , b2[0, t])
        result[:, t] = np.dot(rotz(-theta), b1[:, t])
    return result

if __name__ == "__main__":
    # Control which parts will run
    part_1b = True
    part_1c = True
    part_1d = True
    part_2a = True
    
    n, m, p, v0 = sun_asteroid()
    labels = ["Sun", "Asteroid"]
    if part_1b:
        t, x, v = semi_implicit_euler(p, v0, m, accel, 300 * s_in_day, 3600)
        for i in range(n):
            plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
        plt.title("Planetary Positions")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.legend()
        plt.show()
        t = t / s_in_day
        E, _, _ = energy(x, v, m)
        plt.plot(t, E)
        plt.title("Energy vs. Time")
        plt.xlabel("Time (days)")
        plt.ylabel("Energy (J)")
        plt.show()
        
    if part_1c:
        t, x, v = verlet(p, v0, m, accel, 300 * s_in_day, 3600)
        for i in range(n):
            plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
        plt.title("Planetary Positions")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.legend()
        plt.show()
        t = t / s_in_day
        E, _, _ = energy(x, v, m)
        plt.plot(t, E)
        plt.title("Energy vs. Time")
        plt.xlabel("Time (days)")
        plt.ylabel("Energy (J)")
        plt.show()
    
    if part_1d:
        t, x, v = RKF(p, v0, m, accel, 300 * s_in_day, 2048, 100000, 0.1)
        for i in range(n):
            plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
        plt.title("Planetary Positions")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.show()
        t = t / s_in_day
        E, _, _ = energy(x, v, m)
        plt.plot(t, E)
        plt.title("Energy vs. Time")
        plt.xlabel("Time (days)")
        plt.ylabel("Energy (J)")
        plt.show()
        plt.plot(t)
        plt.title("Time vs. Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Time (days)")
        plt.show()
    
    
    if part_2a:
        n, m, p, v0 = sun_earth_L1()
        labels = ["Sun", "Earth", "Spacecraft"]
        t, x, v = RKF(p, v0, m, accel, 300 * s_in_day, 2048, 100000, 0.1)
        for i in range(n):
            plt.plot(x[i, 0, :], x[i, 1, :], label = labels[i])
        plt.title("L1 - Solar Frame of Reference")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.legend()
        plt.show()
        D = relative_coord(x[2, :, :], x[1, : ,:])
        plt.plot(D[0, :], D[1, :], label = "Spacecraft")
        plt.scatter([au], [0], label = "Earth")
        plt.title("L1 - Earth's Frame of Reference")
        plt.xlabel("m")
        plt.ylabel("m")
        plt.legend()
        plt.show()
    