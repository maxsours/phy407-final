# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:15:46 2020

@author: Max Sours
"""
import numpy as np

def semi_implicit_euler(x0, v0, m, acc, T, dt):
    """
    Run the simulation using semi-implicit euler method
    
    Parameters
    ----------
    x0 : nx3 float array
        Initial positions
    v0 : nx3 float array
        Initial velocities
    m : nx1 float array
        Masses
    acc : func
        Takes positions and masses, outputs acceleration vectors
    T : float
        Integration time
    dt : float
        Timestep
    
    Returns
    -------
    nx3xN float array
        Positions of particles over the simulation

    """
    n = m.size
    time = np.arange(0, T + dt, dt)
    pos = np.zeros((n, 3, len(time)))
    vel = np.zeros((n, 3, len(time)))
    pos[:, :, 0] = x0
    vel[:, :, 0] = v0
    for i in range(len(time) - 1):
        # Euler step
        vel[:, :, i + 1] = vel[:, :, i] + acc(pos[:, :, i], m) * dt
        pos[:, :, i + 1] = pos[:, :, i] + vel[:, :, i + 1] * dt
    return time, pos, vel

def verlet(x0, v0, m, acc, T, dt):
    """
    Run the simulation using the verlet method
    
    Parameters
    ----------
    x0 : nx3 float array
        Initial positions
    v0 : nx3 float array
        Initial velocities
    m : nx1 float array
        Masses
    acc : func
        Takes positions and masses, outputs acceleration vectors
    T : float
        Integration time
    dt : float
        Timestep
    
    Returns
    -------
    1xN float array
        Time during simulation
    nx3xN float array
        Positions of particles over the simulation
    nx3xN float array
        Velocities of particles over the simulation

    """
    n = m.size
    time = np.arange(0, T + dt, dt)
    pos = np.zeros((n, 3, len(time)))
    vel = np.zeros((n, 3, len(time)))
    pos[:, :, 0] = x0
    vel[:, :, 0] = v0
    for i in range(len(time) - 1):
        # Verlet step
        pos[:, :, i + 1] = pos[:, :, i] + (vel[:, :, i] + 0.5 * acc(pos[:, :, i], m) * dt) * dt
        vel[:, :, i + 1] = vel[:, :, i] + 0.5 * (acc(pos[:, :, i], m) + acc(pos[:, :, i + 1], m)) * dt
    return time, pos, vel

def RKF(x0, v0, m, acc, T, h0, xtol, vtol, hrange=None):
    """
    Run the simulation using adaptive Runge-Kutta-Fehlberg

    Parameters
    ----------
    x0 : nx3 float array
        Initial positions
    v0 : nx3 float array
        Initial velocities
    m : nx1 float array
        Masses
    acc : func
        Takes positions and masses, outputs acceleration vectors
    T : float
        Integration time
    h0 : float
        Initial timestep
    xtol : float
        Tolerance in position
    vtol : float
        Tolerance in velocity
    hrange : [float, float], optional
        Bounds on h. The default is None.

    Returns
    -------
    1xN float array
        Time during simulation
    nx3xN float array
        Positions of particles over the simulation
    nx3xN float array
        Velocities of particles over the simulation

    """
    def f(q):
        """
        ODE function when integrating simulation as system
        of 1st order ODEs. Note that f has no dependence on
        time.
    
        """
        result = q * 0
        result[:, :3] = acc(q[:, 3:], m)
        result[:, 3:] = q[:, :3]
        return result
    
    n = m.size
    time = [0]
    h = h0
    #Integration scheme is 
    #   q = (v, x)^T
    #   q' = f(q) = (acc(x), v)^T
    q = np.zeros((n, 6))
    q[:, :3] = v0
    q[:, 3:] = x0
    x = [x0]
    v = [v0]
    while time[-1] <= T:
        # Calculate k0, ..., k5
        k = []
        k.append(h * f(q))
        k.append(h * f(q + k[0]/4))
        k.append(h * f(q + 3*k[0]/32 + 9*k[1]/32))
        k.append(h * f(q + 1932*k[0]/2197 - 7200*k[1]/2197 + 7296*k[2]/2197))
        k.append(h * f(q + 439*k[0]/216 - 8*k[1] + 3680*k[2]/513 - 815*k[3]/4104))
        k.append(h * f(q - 8*k[0]/27 + 2*k[1] - 3544*k[2]/2565 + 1859*k[3]/4104 - 11*k[4]/40))
        # Calculate the 5th order (qz) and the 4th order (q) Runge-Kutta method
        qz = q + 16*k[0]/135 + 6656*k[2]/12825 + 28561*k[3]/56430 - 9*k[4]/50 + 2*k[5]/55 
        qn = q + 25*k[0]/216 + 1408*k[2]/2565 + 2197*k[3]/4101 - k[4]/5
        # Use the difference between the two methods to determine the scalar needed to multiply h by
        Rv = max(np.linalg.norm(qz[:, :3] - qn[:, :3], axis = 1))
        Rx = max(np.linalg.norm(qz[:, 3:] - qn[:, 3:], axis = 1))
        if (Rx > xtol or Rv > vtol):
            # This is a failed step
            h /= 2
            if hrange:
                # If hrange is specified, make sure h is within the bounds
                h = max(h, hrange[0])
            continue
        #Store old h
        time.append(time[-1] + h)
        # Increase h since the error estimate is below the tolerance
        h *= 2
        h = min(h, T - h) #make sure the last value calculated is T
        if hrange:
            # If hrange is specified, make sure h is within the bounds
            h = min(h, hrange[1])
        # Update q
        q = qn
        # Store new values of x, v
        v.append(q[:, :3])
        x.append(q[:, 3:])
    return np.array(time), np.moveaxis(np.array(x), 0, 2), np.moveaxis(np.array(v), 0, 2)
