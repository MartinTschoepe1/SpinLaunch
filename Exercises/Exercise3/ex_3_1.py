#!/usr/bin/env python3

import numpy as np
import os
import scipy.constants
import time
from time import sleep
from scipy.integrate import solve_ivp


def determine_full_path(file_name):
    file_path = os.path.dirname(os.path.abspath('ex_3_1.py'))
    file_path = file_path + "\\Exercises\\Exercise3\\" + file_name
    return file_path

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

# Define a function to calculate force between colliding objects
def calc_collision_forces(x, orig_forces, idx_of_coll_obj):
    space_dim, n = x.shape # dimension of space, number of bodies
    collision_forces = np.zeros((space_dim, n)) # array to store forces (6 bodies, 2 dimensions)
    idx_of_projectile = n-1

    collison_vec = x[:,idx_of_coll_obj] - x[:,idx_of_projectile] # vector pointing from projectile to colliding object
    collison_vec = collison_vec / np.linalg.norm(collison_vec) # normalize vector
    projected_force = np.dot(orig_forces[:,idx_of_projectile], collison_vec) * collison_vec

    collision_forces[:,idx_of_projectile] = - projected_force
    collision_forces[:,idx_of_coll_obj] = projected_force

    return collision_forces

def apply_force_correction(x, orig_forces, idx_of_coll_obj):
    collision_forces = calc_collision_forces(x, orig_forces, idx_of_coll_obj)
    corrected_forces = orig_forces + collision_forces
    return corrected_forces

# Define a function to identify if the projectil touches the surface of an object while a force is acting towards the object
def detect_surface_touch(x, r):
    collision = False
    n = x.shape[1] # number of bodies
    idx_of_projectile = n-1
    idx_of_coll_obj = np.nan # index of the colliding object. Can be between 0 and n-1
    for i in range(n-1):
        current_pos_proj = x[:,idx_of_projectile] # current position of the projectile
        current_pos_obj = x[:,i] # current position of the object
        collision_in_current_step = np.linalg.norm(current_pos_proj-current_pos_obj) < r[i]+r[idx_of_projectile]

        if collision_in_current_step:
            collision = True
            idx_of_coll_obj = i

    return collision, idx_of_coll_obj

# This will find a collision between the projectile and any other object
def detect_collision_projectile(t, y, masses, gravity_const, n_dim, n_bodies, radius):
    x = get_first_half_of_array(y)
    x = x.reshape((n_dim, n_bodies))

    idx_of_projectile = n_bodies-1
    pos_proj  = x[:,idx_of_projectile] # current position of projectile
    product_for_sign_change_criterion = 1.0

    for idx in range(idx_of_projectile):
        if idx == idx_of_projectile: print("Warning: idx == idx_of_projectile should not happen")
        pos_object = x[:,idx]
        distance_between_surfaces = np.linalg.norm(pos_proj-pos_object) - radius[idx] + radius[idx_of_projectile]
        product_for_sign_change_criterion *= distance_between_surfaces

    return product_for_sign_change_criterion

#TODO: Write detection for earth orbit. A stable earth orbit costs enormous amounts of comp time without any interesting result.

# Numerical integration step, checking for collisions, and applying corrections
def step_euler_collision_corrected(x, v, dt, masses, gravity_const, forces, radius):
    n = x.shape[1]
    idx_of_projectile = n-1
    orig_forces = forces(x, masses, gravity_const).transpose()
    massless_forces = orig_forces / masses[np.newaxis,:]

    x_test = x + v * dt
    v_test = v + massless_forces * dt

    collision, idx_of_coll_obj = detect_surface_touch(x_test, radius)

    if collision:
        corrected_forces = apply_force_correction(x, orig_forces, idx_of_coll_obj)
        resulting_messless_forces = corrected_forces / masses[np.newaxis,:]
        x_new = x + v * dt

        for i in range(n-1):
            collison_vec = x_new[:,i] - x_new[:,idx_of_projectile]
            if np.linalg.norm(collison_vec) < radius[i]+radius[idx_of_projectile]:
                collison_depth = np.linalg.norm(collison_vec) - radius[i] - radius[idx_of_projectile]
                collison_vec = collison_vec / np.linalg.norm(collison_vec)
                x_new[:,idx_of_projectile] =  x_new[:,idx_of_projectile] + collison_vec * collison_depth
            
        v_new = v + resulting_messless_forces * dt
    else:
        x_new = x_test
        v_new = v_test

    return x_new, v_new

def get_first_half_of_array(array):
    return array[:len(array)//2]

def get_second_half_of_array(array):
    return array[len(array)//2:]

def update_function(t, y, masses, gravity_const, n_dim, n_bodies, radius):
    #debugging
    x = get_first_half_of_array(y)
    x = x.reshape((n_dim, n_bodies))
    v = get_second_half_of_array(y)
    dvdt = calc_massless_forces(x, masses, gravity_const).flatten()
    res = np.concatenate((v, dvdt))
    return res

def sol_step(t_max, dt, x_init, v_init, masses, gravity_const, radius):
    time_interval = [0, t_max]
    n_dim, n_bodies = x_init.shape

    x_init_flat = x_init.flatten()
    v_init_flat = v_init.flatten()

    # The x_v_init_flat array is the initial state of the system in the follwoing format:
    # [x1, y1, z1, x2, y2, z2, ... , xn, yn, zy, vx1, vy1, vz1, vx2, vy2, vz2, ... , vxn, vyn, vzn]
    # where n is the number of bodies and the first 3*n entries are the initial positions 
    # and the last 3*n entries are the initial velocities
    x_v_init_flat = np.concatenate((x_init_flat, v_init_flat))

    detect_collision_projectile.terminal = True
    detect_collision_projectile.direction = 0

    sol = solve_ivp(update_function, time_interval, x_v_init_flat, method='LSODA', \
        args=(masses, gravity_const, n_dim, n_bodies, radius), first_step=dt, events=detect_collision_projectile, rtol=1e-10, atol=1e-7)

    number_of_steps = len(sol.t)
    print("Number of steps: ", number_of_steps)

    trajectories = get_first_half_of_array(sol.y)
    trajectories = trajectories.reshape(( n_dim, n_bodies, number_of_steps ))

    get_first_half_of_array(sol.y).reshape((n_dim, n_bodies, len(sol.t)))
    return trajectories

# Define function to apply numerical integration step and check for collisions
def integrator_step_collision_checked(x, v, dt, masses, gravity_const, forces, radius):
    x_new, v_new = step_euler(x, v, dt, masses, gravity_const)
    (collision, _ ) = detect_surface_touch(x_new, radius)
    return x_new, v_new, collision

####### Numerical integration step functions #######

def step_euler(x, v, dt, masses, gravity_const):
    x_new = x + v * dt
    massless_forces = calc_massless_forces(x, masses, gravity_const)
    v_new = v + massless_forces * dt

    return x_new, v_new

def calc_massless_forces(x, masses, gravity_const):
    orig_forces = forces(x, masses, gravity_const).transpose()
    massless_forces = orig_forces / masses[np.newaxis,:]
    return massless_forces

####### Force calculation functions #######

def force(r_ij, m_i, m_j, g):
    force = - g * m_i * m_j / np.linalg.norm(r_ij)**3 * r_ij
    return force

def forces(x, masses, g):
    space_dim, n = x.shape # number of bodies, dimension of space
    F = np.zeros((n, space_dim)) # array to store forces (6 bodies, 2 dimensions)
    for i in range(n): 
        for j in range(n):
            # if i != j: # do not calculate force of body on itself
            if i < j: # do not calculate force of body on itself
                distance_vector = (x[:,i] - x[:,j]) # vector pointing from body j to body i
                delta_F = force(distance_vector, masses[i], masses[j], g)
                F[i,:] = F[i,:] + delta_F
                F[j,:] = F[j,:] - delta_F
    return F

####### Solar system simulation functions #######

# Run simulation for a given time or until a collision occurs
def simulate_solar_system(x_init, v_init, dt, m, g, forces, t_max, radius, collision):
    space_dim, n = x_init.shape # number of bodies, dimension of space
    t = 0.0 # start time
    steps = int(t_max/dt) + 1 # number of time steps
    x = x_init # initialize position array
    v = v_init # initialize velocity array
    x_trajec = np.zeros((space_dim, n, steps)) # array to store trajectory
    v_trajec = np.zeros((space_dim, n, steps)) # array to store velocity
    E_trajec = np.zeros(steps) # array to store total energy
    i = 0 # index of current time step
    while i < steps and not collision:
        x_trajec[:,:,i] = x
        v_trajec[:,:,i] = v
        E_trajec[i] = total_energy(x, v, m, g)
        x, v, collision = integrator_step_collision_checked(x, v, dt, m, g, forces, radius)
        t = t + dt
        i = i + 1
    # remove unused entries
    x_trajec = x_trajec[:,:,:i]
    v_trajec = v_trajec[:,:,:i]
    E_trajec = E_trajec[:i]
    return x_trajec, E_trajec, collision

# Debugging Ideas Chain of algo calls:
# Old: single_simulation > simulate_solar_system > integrator_step_collision_checked > step_euler > calc_massless_forces > forces > force
# New(buggy): single_simulation > sol_step > solve_ivp > update_function > calc_massless_forces > forces > force
# Calc_massless_forces > forces > force are called in both cases. Hence, the bug is in sol_step, solve_ivp or update_function
# solve_ivp is a black box, so the bug is likely in sol_step, update_function or in the call of solve_ivp
# Note: Step size is fixed

def single_simulation(x_init, v_init, dt, m, g, forces, t_max, radius):
    # run simulation
    idx_of_projectile = x_init.shape[1]-1
    number_of_angles = 5
    number_of_velocity_steps = 4
    min_velocity = 2.1
    max_velocity = 8.0 # AU/yr = 4744 m/s

    for angle in np.linspace(-90, 0, number_of_angles):
        for velocity in np.linspace(min_velocity, max_velocity, number_of_velocity_steps):
            collision = False
            v_init[0:2,idx_of_projectile] += calc_initial_velocity(angle, velocity)
            #debugging
            x_trajec = sol_step(t_max, dt, x_init, v_init, m, g, radius)
            # x_trajec, E_trajec, collision = simulate_solar_system(x_init, v_init, dt, m, g, forces, t_max, radius, collision)

            # if not collision:
            # plot_energy(E_trajec, "energy.pdf")
            plot_trajectories_geocentric(x_trajec, names, "trajectories_geocentric")
            plot_trajectories_solarcentric(x_trajec, names, "trajectories_solarcentric")

# Calculate inital velocity vector of the projectile that needs to be added to its idle state
def calc_initial_velocity(start_angle, velocity):
    # Todo: determine angle between earth and projectile and add this angle to the start_angle
    angle_rad = start_angle * np.pi / 180
    v_x = velocity * np.cos(angle_rad)
    v_y = velocity * np.sin(angle_rad)
    return np.array([v_x, v_y])

####### Plotting functions #######

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
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.gcf().set_size_inches(12, 12)
    plt.legend()
    plt.savefig(file_name + ".pdf")
    plt.show()

def plot_trajectories_geocentric(x_trajec, names, file_name):
    space_dim, n, t_max = x_trajec.shape
    scale_factor = scipy.constants.astronomical_unit / 6368000
    for i in range(n):
        distance_x = x_trajec[0,i,:] - x_trajec[0,1,:]
        distance_y = x_trajec[1,i,:] - x_trajec[1,1,:]

        plt.plot(distance_x * scale_factor, distance_y * scale_factor, label=names[i])
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    names, x_init, v_init, m, radius, g = load_data("solar_system_projectile_radius_wo_mars.npz")

    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    t0 = time.time() # start clock for timing
    dt = 3.0e-6 # time step in years
    # t_max = 3.0e-1 # maximum time in years
    t_max = 3.0e-2 # maximum time in years

    steps = int(t_max/dt) # number of time steps
    print("dt: ", dt)

    print("Time step in seconds: ", dt*scipy.constants.year)
    print("Simulation length = ", t_max*scipy.constants.year/3600,  "hours", \
        t_max*scipy.constants.year/60, "minutes", t_max*scipy.constants.year,  "seconds" )

    single_simulation(x_init, v_init, dt, m, g, forces, t_max, radius)

    t1 = time.time()
    print("Time elapsed: ", t1-t0, "seconds")
    
    