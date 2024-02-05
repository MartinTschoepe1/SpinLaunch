#!/usr/bin/env python3

import numpy as np
import os
import scipy.constants
import time
from time import sleep

def determine_full_path(file_name):
    file_path = os.path.dirname(os.path.abspath('ex_3_1.py'))
    file_path = file_path + "\\Exercises\\Exercise3\\" + file_name
    return file_path

def force(r_ij, m_i, m_j, g):
    force = - g * m_i * m_j / np.linalg.norm(r_ij)**3 * r_ij
    return force

# Define a function to identify if the projectil touches the surface of an object while a force is acting towards the object
def detect_surface_touch(x, v, dt, r):
    collision = False
    collision_type = 0
    n = x.shape[1] # number of bodies
    n_coll_obj = 0 # number of colliding objects. Should be 0 or 1
    idx_of_coll_obj = np.nan # index of the colliding object. Can be between 0 and n-1
    for i in range(n-1):
        current_pos_proj = x[:,7] # current position of the projectile
        current_pos_obj = x[:,i] # current position of the object
        collision_in_current_step = np.linalg.norm(current_pos_proj-current_pos_obj) < r[i]+r[7]

        future_x = x + v * dt
        future_pos_proj = future_x[:,7] # current position of the projectile
        future_pos_obj = future_x[:,i] # current position of the object
        collision_in_next_step = np.linalg.norm(future_pos_proj-future_pos_obj) < r[i]+r[7]
        if collision_in_current_step:
            collision = True
            collision_type = 1
            n_coll_obj += 1
            idx_of_coll_obj = i
        elif collision_in_next_step:
            collision = True
            collision_type = 2
            n_coll_obj += 1
            idx_of_coll_obj = i

    if n_coll_obj > 1:
        ValueError("The projectile touches the surface of more than one object")

    return collision, collision_type, idx_of_coll_obj

# Define a function to calculate force between colliding objects
def calc_collision_forces(x, orig_forces, idx_of_coll_obj):
    space_dim, n = x.shape # number of bodies, dimension of space
    collision_forces = np.zeros((space_dim, n)) # array to store forces (6 bodies, 2 dimensions)
    idx_of_proj = 7

    collison_vec = x[:,idx_of_coll_obj] - x[:,idx_of_proj] # vector pointing from projectile to colliding object
    collison_vec = collison_vec / np.linalg.norm(collison_vec) # normalize vector
    projected_force = np.dot(orig_forces[:,idx_of_proj], collison_vec) * collison_vec

    collision_forces[:,idx_of_proj] = - projected_force
    collision_forces[:,idx_of_coll_obj] = projected_force

    return collision_forces

def apply_force_correction(x, orig_forces, idx_of_coll_obj):
    collision_forces = calc_collision_forces(x, orig_forces, idx_of_coll_obj)
    corrected_forces = orig_forces + collision_forces
    return corrected_forces

def apply_velocity_correction(x, v, idx_of_coll_obj):
    space_dim, n = x.shape # number of bodies, dimension of space
    collision_velocity = np.zeros((space_dim, n)) # array to store forces (6 bodies, 2 dimensions)
    corrected_velocity = np.zeros((space_dim, n)) # array to store forces (6 bodies, 2 dimensions)
    idx_of_proj = 7

    collison_vec = x[:,idx_of_coll_obj] - x[:,idx_of_proj] # vector pointing from projectile to colliding object
    collison_vec = collison_vec / np.linalg.norm(collison_vec) # normalize vector
    projected_velocity = np.dot(v[:,idx_of_proj], collison_vec) * collison_vec

    collision_velocity[:,idx_of_proj] = - projected_velocity

    corrected_velocity = v + collision_velocity

    return corrected_velocity

def step_euler(x, v, dt, masses, gravity_const, forces, radius):
    space_dim, n = x.shape
    orig_forces = forces(x, masses, gravity_const).transpose()
    massless_forces = orig_forces / masses[np.newaxis,:]

    x_test = x + v * dt
    v_test = v + massless_forces * dt

    collision, collision_type, idx_of_coll_obj = detect_surface_touch(x_test, v_test, dt, radius)

    if collision:
        # if collision_type==1:
        #     corrected_velocity = apply_velocity_correction(x, v, idx_of_coll_obj)
            # v = corrected_velocity

        corrected_forces = apply_force_correction(x, orig_forces, idx_of_coll_obj)
        resulting_messless_forces = corrected_forces / masses[np.newaxis,:]
        x_new = x + v * dt

        for i in range(n-1):
            collison_vec = x_new[:,i] - x_new[:,7]
            if np.linalg.norm(collison_vec) < radius[i]+radius[7]:
                collison_depth = np.linalg.norm(collison_vec) - radius[i] - radius[7]
                # print("Collision depth: ", np.linalg.norm(collison_vec), radius[i], radius[7], collison_depth)
                collison_vec = collison_vec / np.linalg.norm(collison_vec)
                x_new[:,7] =  x_new[:,7] + collison_vec * collison_depth
            
        v_new = v + resulting_messless_forces * dt
    else:
        x_new = x_test
        v_new = v_test

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
def simulate_solar_system(x_init, v_init, dt, m, g, forces, t_max, radius):
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
        x, v = step_euler(x, v, dt, m, g, forces, radius)
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

def plot_trajectories_solarcentric(x_trajec, names, file_name):
    space_dim, n, t_max = x_trajec.shape
    for i in range(n):
        plt.plot(x_trajec[0,i,:], x_trajec[1,i,:], label=names[i])
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.xlim(0.997, 1.003)
    plt.ylim(-0.001, 0.09)
    plt.gcf().set_size_inches(12, 12)
    plt.legend()
    plt.savefig(file_name + ".pdf")
    plt.show()

def plot_trajectories_geocentric(x_trajec, names, file_name):
    space_dim, n, t_max = x_trajec.shape
    scale_factor = scipy.constants.astronomical_unit / 6368000
    for i in range(n):
        euclidian_distance_x = x_trajec[0,i,:] - x_trajec[0,1,:]
        euclidian_distance_y = x_trajec[1,i,:] - x_trajec[1,1,:]

        plt.plot(euclidian_distance_x * scale_factor, euclidian_distance_y * scale_factor, label=names[i])
    plt.setp(plt.gca().lines, linewidth=2)
    plt.xlabel("x [Earth radii]")
    plt.ylabel("y [Earth radii]")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gcf().set_size_inches(12, 12)
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
    x_init = data["x_init"] # initial positions of the orbiting bodies in AU
    v_init = data["v_init"] # initial velocities of the orbiting bodies in AU/yr
    m = data["m"] # masses of the orbiting bodies in earth masses
    radius = data["radius"] # radii of the orbiting bodies in AU
    g = data["g"] # one gravitational constant for all bodies

    return names, x_init, v_init, m, radius, g

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # names, x_init, v_init, m, g = load_data("solar_system.npz")
    # names, x_init, v_init, m, g = load_data("solar_system_mercury.npz")
    names, x_init, v_init, m, radius, g = load_data("solar_system_projectile_radius.npz")

    t0 = time.time() # start clock for timing
    earth_radius = 6.371e6 / scipy.constants.au

    # Did work once, but the trajectory of the projectile looks weird.
    t_max = 9.0e-4 # maximum time in years
    steps = 1e5 # number of time steps
    
    dt = t_max/steps # time step in years
    print("Number of time steps: ", steps)

    print("Time step in seconds: ", dt*scipy.constants.year)
    print("Simulation length = ", t_max*scipy.constants.year/3600,  "hours", \
        t_max*scipy.constants.year/60, "minutes", t_max*scipy.constants.year,  "seconds" )
    
    # run simulation
    x_trajec, v_trajec, E_trajec = simulate_solar_system(x_init, v_init, dt, m, g, forces, t_max, radius)

    t1 = time.time()
    print("Time elapsed: ", t1-t0, "seconds")

    # plot_energy(E_trajec, "energy.pdf")
    plot_trajectories_geocentric(x_trajec, names, "trajectories_geocentric")
    # plot_trajectories_solarcentric(x_trajec, names, "trajectories_solarcentric")

    
    