#!/usr/bin/env python3

import numpy as np
import os
import scipy.constants
import time
from time import sleep
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class ProjectileInitialConditions:
    def __init__(self, angle, velocity):
        self.angle = angle
        self.velocity = velocity
        self.min_distance_to_sun = np.nan
        self.stop_criterion = np.nan

# Define a function to create object of class ProjectileInitialConditions give a min and max angle and velocity
def set_projectile_init_con(min_angle, max_angle, min_velocity, max_velocity, number_of_angles, number_of_velocity_steps):
    condition_array = np.zeros((number_of_angles, number_of_velocity_steps), dtype=ProjectileInitialConditions)
    for idx_angle, angle in enumerate(np.linspace(min_angle, max_angle, number_of_angles)):
        for idx_velocity, velocity in enumerate(np.linspace(min_velocity, max_velocity, number_of_velocity_steps)):
            condition_array[idx_angle, idx_velocity] = ProjectileInitialConditions(angle, velocity)

    return condition_array

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
    idx_first_object = n-1

    collison_vec = x[:,idx_of_coll_obj] - x[:,idx_first_object] # vector pointing from projectile to colliding object
    collison_vec = collison_vec / np.linalg.norm(collison_vec) # normalize vector
    projected_force = np.dot(orig_forces[:,idx_first_object], collison_vec) * collison_vec

    collision_forces[:,idx_first_object] = - projected_force
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
    idx_first_object = n-1
    idx_of_coll_obj = np.nan # index of the colliding object. Can be between 0 and n-1
    for i in range(n-1):
        current_pos_proj = x[:,idx_first_object] # current position of the projectile
        current_pos_obj = x[:,i] # current position of the object
        collision_in_current_step = np.linalg.norm(current_pos_proj-current_pos_obj) < r[i]+r[idx_first_object]

        if collision_in_current_step:
            collision = True
            idx_of_coll_obj = i

    return collision, idx_of_coll_obj


# Numerical integration step, checking for collisions, and applying corrections
def step_euler_collision_corrected(x, v, dt, masses, gravity_const, forces, radius):
    n = x.shape[1]
    idx_first_object = n-1
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
            collison_vec = x_new[:,i] - x_new[:,idx_first_object]
            if np.linalg.norm(collison_vec) < radius[i]+radius[idx_first_object]:
                collison_depth = np.linalg.norm(collison_vec) - radius[i] - radius[idx_first_object]
                collison_vec = collison_vec / np.linalg.norm(collison_vec)
                x_new[:,idx_first_object] =  x_new[:,idx_first_object] + collison_vec * collison_depth
            
        v_new = v + resulting_messless_forces * dt
    else:
        x_new = x_test
        v_new = v_test

    return x_new, v_new



###### Necessary due to the 1D (poc, velo, force) vector in solve_ivp ######

def get_first_half_of_array(array):
    return array[:len(array)//2]

def get_second_half_of_array(array):
    return array[len(array)//2:]

def get_position_array(y, n_dim, n_bodies):
    x_vec = get_first_half_of_array(y)
    x_array = x_vec.reshape((n_dim, n_bodies))
    return x_array

def get_velocity_array(y, n_dim, n_bodies):
    v_vec = get_second_half_of_array(y)
    v_array = v_vec.reshape((n_dim, n_bodies))
    return v_array



###### Update function and event functions for solve_ivp ######

def update_function(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj):
    x_array = get_position_array(y, n_dim, n_bodies)
    v_vec = get_second_half_of_array(y)
    dvdt = calc_massless_forces(x_array, masses, gravity_const).flatten()
    res = np.concatenate((v_vec, dvdt))
    return res

def detect_collision_projectile(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj):
    x_array = get_position_array(y, n_dim, n_bodies)

    idx_first_object = n_bodies-1
    pos_proj  = x_array[:,idx_first_object] # current position of projectile
    product_for_sign_change_criterion = 1.0

    for idx in range(idx_first_object):
        pos_object = x_array[:,idx]
        distance_between_surfaces = np.linalg.norm(pos_proj-pos_object) - radius[idx] + radius[idx_first_object]
        if (distance_between_surfaces < 0):
            idx_coll_obj[0] = idx
        product_for_sign_change_criterion *= distance_between_surfaces
    return product_for_sign_change_criterion

# Detect when projectile is still within moon orbit after one week. Event is terminal. Run time optimization. 
def detect_slow_orbits(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj):
    x_array = get_position_array(y, n_dim, n_bodies)

    idx_first_object = n_bodies-1
    idx_of_earth = 1
    idx_of_moon = 2
    six_days_in_years = 6.0 / 365.0
    pos_proj  = x_array[:,idx_first_object]
    pos_earth = x_array[:,idx_of_earth]
    pos_moon  = x_array[:,idx_of_moon]
    
    is_projectile_within_moon_orbit = np.linalg.norm(pos_proj-pos_earth) < np.linalg.norm(pos_moon-pos_earth)
    is_within_six_days = t < six_days_in_years

    if (is_within_six_days):
        return 1
    else:
        if (is_projectile_within_moon_orbit):
            return -1
        else:
            return 1

# Helper function that is only indirectly called by solve_ivp via detect_inner_solar_apex and detect_outer_solar_apex
def detect_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached):
    x_array = get_position_array(y, n_dim, n_bodies)
    v_array = get_velocity_array(y, n_dim, n_bodies)

    idx_first_object = n_bodies-1
    idx_second_object = 0
    relativ_pos_proj = get_connection_vector(x_array, idx_first_object, idx_second_object)

    velocity_proj = v_array[:,idx_first_object]
    velocity_sun = v_array[:,idx_second_object]

    relativ_velocity_proj = velocity_proj - velocity_sun
    abs_value_distance_to_sun = np.linalg.norm(relativ_pos_proj)

    dot_product = np.dot(relativ_pos_proj, relativ_velocity_proj)
    denominator = np.linalg.norm(relativ_pos_proj) * np.linalg.norm(relativ_velocity_proj)
    angle_between_pos_and_velocity_rad = np.arccos(dot_product / denominator)
    angle_degree = angle_between_pos_and_velocity_rad * 180 / np.pi - 90

    return angle_degree

def get_connection_vector(x_array, idx_first_object, idx_second_object):
    pos1  = x_array[:,idx_first_object]
    pos2 = x_array[:,idx_second_object]
    return pos1 - pos2

# Detect inner apex of projectile trajectory to increase accuracy, event is not terminal
def detect_inner_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj):
    angle_degree = detect_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached)
    if (angle_degree < 0 and is_apex_reached[0] == False): is_apex_reached[0] = True
    return angle_degree

# Detect outer apex after the first inner apex of projectile trajectory to terminate simulation, run time optimization
def detect_outer_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj):
    angle_degree = detect_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached)
    one_week_in_years = 1.0 / 52.0
    if (is_apex_reached[0] == False or t < one_week_in_years):
        return 1.0
    else:
        return angle_degree

def detect_outer_solar_system(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj):
    x_array = get_position_array(y, n_dim, n_bodies)
    idx_first_object = n_bodies-1
    idx_second_object = 0
    distance_sun_jupiter = 5.2 # AU
    pos_proj  = x_array[:,idx_first_object]
    pos_sun = x_array[:,idx_second_object]
    distance_to_sun = np.linalg.norm(get_connection_vector(x_array, idx_first_object, idx_second_object))
    return distance_to_sun - distance_sun_jupiter

def get_stop_criterion(sol, idx_coll_obj):
    
    if (sol.status == 0):
        stop_criterion = 1
    elif (sol.status == 1):
        if (sol.t_events[3].size > 0):
            stop_criterion = 2
        elif (sol.t_events[4].size > 0):
            stop_criterion = 3
        elif (sol.t_events[1].size > 0):
            stop_criterion = 4
        elif (sol.t_events[0].size > 0):
            if (idx_coll_obj[0] == 0):
                stop_criterion = 5
            elif (idx_coll_obj[0] == 1):
                stop_criterion = 6
            else:
                stop_criterion = 7
        else:
            stop_criterion = 8
    return stop_criterion


def sol_step(t_max, dt, x_init, v_init, masses, gravity_const, radius):
    time_interval = [0, t_max]
    n_dim, n_bodies = x_init.shape
    is_apex_reached = [False]
    idx_coll_obj = [False]
    stop_criterion = 0

    x_init_flat = x_init.flatten()
    v_init_flat = v_init.flatten()

    # The x_v_init_flat array is the initial state of the system in the follwoing format:
    # [x1, y1, z1, x2, y2, z2, ... , xn, yn, zy, vx1, vy1, vz1, vx2, vy2, vz2, ... , vxn, vyn, vzn]
    # where n is the number of bodies and the first 3*n entries are the initial positions 
    # and the last 3*n entries are the initial velocities
    x_v_init_flat = np.concatenate((x_init_flat, v_init_flat))

    detect_collision_projectile.terminal = True
    detect_collision_projectile.direction = 0
    detect_slow_orbits.terminal = True
    detect_slow_orbits.direction = 0
    detect_inner_solar_apex.terminal = False
    detect_inner_solar_apex.direction = -1.0
    detect_outer_solar_apex.terminal = True
    detect_outer_solar_apex.direction = 1.0
    detect_outer_solar_system.terminal = True
    detect_outer_solar_system.direction = 1.0

    args_tuple = masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj
    event_tuple = detect_collision_projectile, detect_slow_orbits, detect_inner_solar_apex, \
        detect_outer_solar_apex, detect_outer_solar_system
    # Method comparison:
    # Time: RK45 (30.50s), RK23 (168.53s), DOP853 (31.15s), Radau (210.90s), BDF (75.36s), LSODA (25.92s)
    sol = solve_ivp(update_function, time_interval, x_v_init_flat, args=args_tuple, \
        first_step=dt, events=event_tuple, rtol=1e-10, atol=1e-7)

    number_of_steps = len(sol.t)

    stop_criterion = get_stop_criterion(sol, idx_coll_obj)

    trajectories = get_first_half_of_array(sol.y)
    trajectories = trajectories.reshape(( n_dim, n_bodies, number_of_steps ))

    get_first_half_of_array(sol.y).reshape((n_dim, n_bodies, len(sol.t)))
    return trajectories, number_of_steps, stop_criterion

# Define function to apply numerical integration step and check for collisions
def integrator_step_collision_checked(x, v, dt, masses, gravity_const, forces, radius):
    x_new, v_new = step_euler(x, v, dt, masses, gravity_const)
    (collision, _ ) = detect_surface_touch(x_new, radius)
    return x_new, v_new, collision



###### Numerical integration step functions ######

def step_euler(x, v, dt, masses, gravity_const):
    x_new = x + v * dt
    massless_forces = calc_massless_forces(x, masses, gravity_const)
    v_new = v + massless_forces * dt

    return x_new, v_new

def calc_massless_forces(x, masses, gravity_const):
    orig_forces = forces(x, masses, gravity_const).transpose()
    massless_forces = orig_forces / masses[np.newaxis,:]
    return massless_forces



###### Force calculation functions ######

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



###### Solar system simulation functions ######

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

def single_simulation(x_init, v_init, dt, m, g, forces, t_max, radius, condition_array):
    v_init_new = np.copy(v_init)
    idx_projectile = x_init.shape[1]-1
    numb_of_simulations = condition_array.size

    for numb, conditions in enumerate(condition_array.flatten()):
        collision = False
        v_init_new[0:2,idx_projectile] = v_init[0:2,idx_projectile] + calc_initial_velocity(conditions.angle, conditions.velocity)
        x_trajec, number_of_steps, stop_criterion = sol_step(t_max, dt, x_init, v_init_new, m, g, radius)

        # print("stop_criterion: ", dict_stop_criterion[stop_criterion])
        print("Sim. #", numb, "of ", numb_of_simulations, "# of steps: ", number_of_steps, "angle: ", np.round(conditions.angle, 4),
                "velocity: ", np.round(conditions.velocity, 3))

        conditions.min_distance_to_sun = calc_min_distance_to_sun(x_trajec, radius)
        conditions.stop_criterion = stop_criterion

        # plot_energy(E_trajec, "energy.pdf")
        # plot_trajectories_geocentric(x_trajec, names, "trajectories_geocentric")
        # plot_trajectories_solarcentric(x_trajec, names, "trajectories_solarcentric")

    plot_min_distance_to_sun(condition_array, "min_distance_to_sun.pdf")
    plot_stop_criterion(condition_array, "stop_criterion.pdf")

# Calculate inital velocity vector of the projectile that needs to be added to its idle state
def calc_initial_velocity(start_angle, velocity):
    # Todo: determine angle between earth and projectile and add this angle to the start_angle
    angle_rad = start_angle * np.pi / 180
    v_x = velocity * np.cos(angle_rad)
    v_y = velocity * np.sin(angle_rad)
    return np.array([v_x, v_y])

# Calculate the minimal distance between the projectile and the sun
def calc_min_distance_to_sun(x_trajec, radius):
    space_dim, n, t_max = x_trajec.shape
    idx_sun = 0
    idx_proj = n-1
    delta_trajec = x_trajec[:,idx_sun,:] - x_trajec[:,idx_proj,:]
    dist_proj_sun = np.linalg.norm(delta_trajec, axis=0)
    min_distance_to_sun_center_of_mass = np.min(dist_proj_sun, axis=0)
    return min_distance_to_sun_center_of_mass


###### Plotting functions ######

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
    solar_radius = 696342000 # in meters
    solar_radius_per_au = scipy.constants.astronomical_unit / solar_radius
    x_trajec = x_trajec*solar_radius_per_au # Will not be saved after function call

    for i in range(0,n,1):
        plt.plot(x_trajec[0,i,:], x_trajec[1,i,:], label=names[i], marker='o', markersize=2)
    plt.xlabel("x [Sun radii]")
    plt.ylabel("y [Sun radii]")
    plt.xlim(-6*solar_radius_per_au, 6*solar_radius_per_au)
    plt.ylim(-6*solar_radius_per_au, 6*solar_radius_per_au)
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
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.gcf().set_size_inches(12, 12)
    plt.legend()
    plt.savefig(file_name + ".pdf")
    plt.show()

def plot_energy(E_trajec, file_name):
    plt.plot(E_trajec)
    plt.xlabel("time step")
    plt.ylabel("total energy")
    # set x range from 0 max value reached by E_trajec
    plt.ylim(0, np.max(E_trajec)*1.05)
    plt.savefig(file_name)
    plt.show()

# Extracts the angle, velocity and minimal distance to sun/stop criterium from the condition array
def convert_set_of_objects_into_arrays(condition_array, attribute_name):
    number_of_angles = condition_array.shape[0]
    number_of_velocity_steps = condition_array.shape[1]
    angle_array = np.zeros((number_of_angles))
    velocity_array = np.zeros((number_of_velocity_steps))
    min_distance_to_sun_array = np.zeros((number_of_angles, number_of_velocity_steps))
    
    for idx_angle in range(number_of_angles):
        for idx_velocity in range(number_of_velocity_steps):
            angle_array[idx_angle] = condition_array[idx_angle, idx_velocity].angle
            velocity_array[idx_velocity] = condition_array[idx_angle, idx_velocity].velocity
            min_distance_to_sun_array[idx_angle, idx_velocity] = getattr(condition_array[idx_angle, idx_velocity],attribute_name)
    
    return velocity_array, angle_array, min_distance_to_sun_array

# Creates figure with colorbar that shows the mimal distance between the projectile and sun depending on the angle and velocity
def plot_min_distance_to_sun(condition_array, file_name):
    velocity_array, angle_array, min_distance_to_sun_array = convert_set_of_objects_into_arrays(condition_array, 'min_distance_to_sun')

    conv_AU_yr_to_km_s = 4744
    conv_AU_to_solar_radius = 215
    x, y = np.meshgrid(velocity_array*conv_AU_yr_to_km_s, angle_array)

    min_distance_to_sun_array = min_distance_to_sun_array*conv_AU_to_solar_radius

    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, min_distance_to_sun_array, cmap='viridis', \
        norm=colors.LogNorm(vmin=0.99, vmax=100))

    ax.set_xlabel('velocity [km/s]')
    ax.set_ylabel('angle [°]')
    fig.colorbar(c, ax=ax, label='minimal distance to sun [solar radii]')
    plt.gcf().set_size_inches(18, 12)
    plt.savefig(file_name)
    plt.show()

def plot_stop_criterion(condition_array, file_name):
    velocity_array, angle_array, stop_criterion_array = convert_set_of_objects_into_arrays(condition_array, 'stop_criterion')
            
    # dict_stop_criterion = {0: "no stop", 1: "time limit", 2: "outer apex", 3: "outer solar system", 4: "slow orbit", \
    #     5: "sun collision", 6: "earth collision", 7: "collision with other object", 8: "Error with stop criterion"}

    ticklabels = ['no stop', 'time limit', 'outer apex', 'outer solar system', 'slow orbit', \
        'sun collision', 'earth collision', 'collision with other object']

    x, y = np.meshgrid(velocity_array, angle_array)

    fig, ax = plt.subplots()

    ax.set_xlabel('velocity [AU/yr]')
    ax.set_ylabel('angle [°]')

    cmap = plt.get_cmap('tab10', 8)
    c = ax.pcolormesh(x, y, stop_criterion_array, cmap=cmap,  vmin= -0.5, vmax=7.5)

    cbar = fig.colorbar(c, ax=ax, label='stop criterion', ticks=[0, 1, 2, 3, 4, 5, 6, 7])
    cbar.set_ticklabels(ticklabels)


    plt.gcf().set_size_inches(18, 12)
    plt.savefig(file_name)
    plt.show()


if __name__ == "__main__":

    names, x_init, v_init, m, radius, g = load_data("solar_system_projectile_radius_wo_mars.npz")

    min_angle = -165
    max_angle = -185
    min_velocity =  4.0
    max_velocity = 11.0 # AU/yr = 4744 m/s
    number_of_angles = 8
    number_of_velocity_steps = 8
    condition_array = set_projectile_init_con(min_angle, max_angle, min_velocity, max_velocity, \
        number_of_angles, number_of_velocity_steps)

    # Set print options for numpy
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    t0 = time.time() # start clock for timing
    dt = 3.0e-6 # time step in years
    t_max = 1.005 # maximum time in years

    single_simulation(x_init, v_init, dt, m, g, forces, t_max, radius, condition_array)

    t1 = time.time()
    print("Time elapsed: ", t1-t0, "seconds")
    
    