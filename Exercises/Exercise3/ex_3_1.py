#!/usr/bin/env python3

import numpy as np
import os
import scipy.constants

def determine_full_path(file_name):
    file_path = os.path.dirname(os.path.abspath('ex_3_1.py'))
    file_path = file_path + "\\Exercises\\Exercise3\\" + file_name
    return file_path

def force(r_ij, m_i, m_j, g):
    force = - g * m_i * m_j / np.linalg.norm(r_ij)**3 * r_ij
    return force

def step_euler(x, v, dt, masses, gravity_const, forces):
    x_new = x + v * dt
    delta_forces = forces(x, masses, gravity_const).transpose()
    massless_forces = delta_forces / masses[np.newaxis,:]
    v_new = v + massless_forces * dt
    # print(massless_forces)
    return x_new, v_new

def forces(x, masses, g):
    space_dim, n = x.shape # number of bodies, dimension of space
    F = np.zeros((n, space_dim)) # array to store forces (6 bodies, 2 dimensions)
    for i in range(n): 
        for j in range(n):
            if i != j: # do not calculate force of body on itself
                distance_vector = (x[:,i] - x[:,j]) # vector pointing from body j to body i
                # delta_F = force(distance_vector, 1.0, 1.0, g)
                delta_F = force(distance_vector, masses[i], masses[j], g)
                F[i,:] = F[i,:] + delta_F
    # print(F)
    return F

# Run simulation for a given time
def simulate_solar_system(x_init, v_init, dt, m, g, forces, t_max):
    space_dim, n = x_init.shape # number of bodies, dimension of space
    t = 0.0 # start time
    steps = int(t_max/dt) # number of time steps
    x = x_init # initialize position array
    v = v_init # initialize velocity array
    x_trajec = np.zeros((space_dim, n, steps)) # array to store trajectory
    v_trajec = np.zeros((space_dim, n, steps)) # array to store velocity
    E_trajec = np.zeros(steps) # array to store total energy
    i = 0 # index of current time step
    while i < steps:
        x_trajec[:,:,i] = x
        v_trajec[:,:,i] = v
        E_trajec[i] = total_energy(x, v, m, g)
        x, v = step_euler(x, v, dt, m, g, forces)
        t = t + dt
        i = i + 1
    return x_trajec, v_trajec, E_trajec

def total_energy(x, v, masses, g):
    n = x.shape[1] # number of bodies
    E = 0.0 # total energy
    for i in range(n):
        E = E + 0.5 * masses[i] * np.linalg.norm(v[:,i])**2 # kinetic energy
        for j in range(n):
            if i != j: # do not calculate force of body on itself
                distance_vector = (x[:,i] - x[:,j]) # vector pointing from body j to body i
                E = E - g * masses[i] * masses[j] / np.linalg.norm(distance_vector) # potential energy
    return E

# define a function that plots trajectories of all objects
def plot_trajectories(x_trajec, names, file_name):
    space_dim, n, t_max = x_trajec.shape
    for i in range(n):
        plt.plot(x_trajec[0,i,:], x_trajec[1,i,:], label=names[i])
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.legend()
    plt.savefig(file_name + ".pdf")
    plt.show()

# define a function that plots the total energy
def plot_energy(E_trajec, file_name):
    plt.plot(E_trajec)
    plt.xlabel("time step")
    plt.ylabel("total energy")
    # set x range from 0 max value reached by E_trajec
    plt.ylim(0, np.max(E_trajec)*1.05)
    plt.savefig(file_name)
    plt.show()


# load initil positions and masses from file
def load_data(file_name):
    file_path = determine_full_path(file_name)
    data = np.load(file_path)

    names = data["names"] # names of the orbiting bodies
    x_init = data["x_init"]
    v_init = data["v_init"] 
    m = data["m"] # masses of the orbiting bodies
    g = data["g"] # one gravitational constant for all bodies

    return names, x_init, v_init, m, g

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    names, x_init, v_init, m, g = load_data("solar_system.npz")

    # run a single euler step
    dt = 0.0001 # time step in years
    t_max = 1.0 # maximum time in years
    x_new, v_new = step_euler(x_init, v_init, dt, m, g, forces)

    # run simulation
    x_trajec, v_trajec, E_trajec = simulate_solar_system(x_init, v_init, dt, m, g, forces, t_max)

    plot_energy(E_trajec, "energy.pdf")
    plot_trajectories(x_trajec, names, "trajectories")


