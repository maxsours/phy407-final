# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:33:02 2020

@author: Max Sours
"""
import numpy as np

# Constants
G = 6.6743e-11
M_sun = 1.9884e+30
M_earth = 5.972e+24
M_jup = 1.8981e+27
au = 149597870700.0
s_in_day = 24 * 3600

def set_vel(Msun, planets):
    """
    Set velocities of k planets given planet positions and
    solar mass, assuming the solar mass is at (0, 0, 0),
    the planets are in circular orbits that do not affect
    each other. (This is an oversimplification designed
    to quickly give an initial velocity at t = 0.)

    Parameters
    ----------
    Msun : float
        Mass of sun (could be mass of barycenter as well)
    planets : kx3 float array
        Initial positions of planets

    Returns
    -------
    kx3 float array
        Initial velocities of planets

    """
    return np.sqrt(G * Msun / np.linalg.norm(planets, axis = 1))

def rotz(theta):
    """
    Create rotation matrix for a rotation of theta rad
    about the z axis

    Parameters
    ----------
    theta : float
        Radians to rotate

    Returns
    -------
    3x3 float array
        Rotation matrix

    """
    R = np.eye(3)
    R[0, 0] = np.cos(theta)
    R[1, 1] = np.cos(theta)
    R[0, 1] = -np.sin(theta)
    R[1, 0] = np.sin(theta)
    return R

def sun_earth_moon():
    """
    Give initial conditions for Sun-Earth-Moon System

    """
    n = 3
    m = np.zeros((n, 1))
    m[0, :] = M_sun
    m[1, :] = M_earth
    m[2, :] = 0.0123 * M_earth
    x0 = np.zeros((n, 3))
    x0[1, 0] = 1.0
    x0[2, 0] = 1.00257
    x0 *= au
    # Put the barycenter at the origin
    x0 -= np.sum(m * x0, axis = 0) / np.sum(m)
    v0 = np.zeros((n, 3))
    v0[1:, 1] = set_vel(m[0, :], x0[1:, :])
    v0[2:, 1] += set_vel(m[1, :], x0[2:, :] - x0[1:2, :])
    # Set momentum to 0
    v0 -= np.sum(m * v0, axis = 0) / np.sum(m)
    return n, m, x0, v0

def sun_earth():
    """
    Give initial conditions for Sun-Earth System

    """
    n = 2
    m = np.zeros((n, 1))
    m[0, :] = M_sun
    m[1, :] = M_earth
    x0 = np.zeros((n, 3))
    x0[1, 0] = 1.0
    x0 *= au
    # Put the barycenter at the origin
    x0 -= np.sum(m * x0, axis = 0) / np.sum(m)
    v0 = np.zeros((n, 3))
    v0[1:, 1] = set_vel(m[0, :], x0[1:, :])
    # Set momentum to 0
    v0 -= np.sum(m * v0, axis = 0) / np.sum(m)
    return n, m, x0, v0

def sun_asteroid():
    n = 2
    m = np.zeros((n, 1))
    m[0, :] = M_sun
    m[1, :] = M_earth * 0.01
    x0 = np.zeros((n, 3))
    x0[1, 0] = 1.0
    x0 *= au
    # Put the barycenter at the origin
    x0 -= np.sum(m * x0, axis = 0) / np.sum(m)
    v0 = np.zeros((n, 3))
    v0[1, 1] = 10000
    # Set momentum to 0
    v0 -= np.sum(m * v0, axis = 0) / np.sum(m)
    return n, m, x0, v0

def sun_earth_L1():
    """
    Give initial conditions for sun-earth system with
    sattelite at L1
    
    """
    n = 3
    alpha = M_earth / M_sun
    m = np.zeros((n, 1))
    m[0, :] = M_sun
    m[1, :] = M_earth
    m[2, :] = 1
    x0 = np.zeros((n, 3))
    x0[1, 0] = 1.0
    x0[2, 0] = 1.0 * (1 - (alpha / 3) ** (1/3))
    x0 *= au
    # Put the barycenter at the origin
    x0 -= np.sum(m * x0, axis = 0) / np.sum(m)
    v0 = np.zeros((n, 3))
    v0[1:2, 1] = set_vel(m[0, :], x0[1:2, :])
    v0[2, 1] = v0[1, 1] * x0[2, 0] / x0[1, 0]
    # Set momentum to 0
    v0 -= np.sum(m * v0, axis = 0) / np.sum(m)
    return n, m, x0, v0

def sun_earth_L2():
    """
    Give initial conditions for sun-earth system with
    sattelite at L2
    
    """
    n = 3
    alpha = M_earth / M_sun
    m = np.zeros((n, 1))
    m[0, :] = M_sun
    m[1, :] = M_earth
    m[2, :] = 1
    x0 = np.zeros((n, 3))
    x0[1, 0] = 1.0
    x0[2, 0] = 1.0 * (1 + (alpha / 3) ** (1/3))
    x0 *= au
    # Put the barycenter at the origin
    x0 -= np.sum(m * x0, axis = 0) / np.sum(m)
    v0 = np.zeros((n, 3))
    v0[1:2, 1] = set_vel(m[0, :], x0[1:2, :])
    v0[2, 1] = v0[1, 1] * x0[2, 0] / x0[1, 0]
    # Set momentum to 0
    v0 -= np.sum(m * v0, axis = 0) / np.sum(m)
    return n, m, x0, v0

def sun_earth_L3():
    """
    Give initial conditions for sun-earth system with
    sattelite at L2
    
    """
    n = 3
    alpha = M_earth / M_sun
    m = np.zeros((n, 1))
    m[0, :] = M_sun
    m[1, :] = M_earth
    m[2, :] = 1
    x0 = np.zeros((n, 3))
    x0[1, 0] = 1.0
    x0[2, 0] = -1 - 5 * alpha / 12
    x0 *= au
    # Put the barycenter at the origin
    x0 -= np.sum(m * x0, axis = 0) / np.sum(m)
    v0 = np.zeros((n, 3))
    v0[1:2, 1] = set_vel(m[0, :], x0[1:2, :])
    v0[2, 1] = v0[1, 1] * x0[2, 0] / x0[1, 0]
    # Set momentum to 0
    v0 -= np.sum(m * v0, axis = 0) / np.sum(m)
    return n, m, x0, v0

def sun_earth_L4():
    """
    Give initial conditions for sun-earth system with
    sattelite at L4
    
    """
    n = 3
    alpha = M_earth / M_sun
    m = np.zeros((n, 1))
    m[0, :] = M_sun
    m[1, :] = M_earth
    m[2, :] = 1
    x0 = np.zeros((n, 3))
    x0[1, 0] = 1.0
    x0[2, 0] = 0.5 * (M_sun - M_earth) / (M_sun + M_earth)
    x0[2, 1] = np.sqrt(3) / 2
    x0 *= au
    # Put the barycenter at the origin
    x0 -= np.sum(m * x0, axis = 0) / np.sum(m)
    v0 = np.zeros((n, 3))
    v0[1:, 1] = set_vel(m[0, :], x0[1:, :])
    v0[2, :] = np.dot(rotz(np.arctan2(x0[2, 1], x0[2, 0])), v0[2, :])
    # Set momentum to 0
    v0 -= np.sum(m * v0, axis = 0) / np.sum(m)
    return n, m, x0, v0

def sun_earth_L5():
    """
    Give initial conditions for sun-earth system with
    sattelite at L5
    
    """
    n = 3
    alpha = M_earth / M_sun
    m = np.zeros((n, 1))
    m[0, :] = M_sun
    m[1, :] = M_earth
    m[2, :] = 1
    x0 = np.zeros((n, 3))
    x0[1, 0] = 1.0
    x0[2, 0] = 0.5 * (M_sun - M_earth) / (M_sun + M_earth)
    x0[2, 1] = -np.sqrt(3) / 2
    x0 *= au
    # Put the barycenter at the origin
    x0 -= np.sum(m * x0, axis = 0) / np.sum(m)
    v0 = np.zeros((n, 3))
    v0[1:, 1] = set_vel(m[0, :], x0[1:, :])
    v0[2, :] = np.dot(rotz(np.arctan2(x0[2, 1], x0[2, 0])), v0[2, :])
    # Set momentum to 0
    v0 -= np.sum(m * v0, axis = 0) / np.sum(m)
    return n, m, x0, v0

if __name__ == "__main__":
    sun_earth_moon()