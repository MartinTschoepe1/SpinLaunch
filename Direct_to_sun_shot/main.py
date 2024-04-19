#!/usr/bin/env python3

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import solve_ivp
from scipy.constants import astronomical_unit as AU

import spin_launch_constants as slc
import Forces
import Plotting
import Event_and_update_functions_for_solve_ivp as euf

#Assumed projectile properties
# Explicitly:
# radius = 0.30 # m
# mass = 1000 # kg
# drag_coefficient = 0.50 # unitless
# Implicitly:
# base area = 0.30^2 * pi = 0.2827 m^2
#
# Not used:
# density = 10000 kg/m^3
# volume = 0.1 m^3
# Shape: Cone (Kegel)
# height = 1.06 m via V = 1/3 * pi * r^2 * h
# projected upright area = r*h = 0.318 m^2


class ProjectileInitialConditions:
    def __init__(self, angle, velocity, daytime):
        self.angle = angle
        self.velocity = velocity
        self.daytime = daytime # in hours
        self.min_distance_to_sun = np.nan
        self.stop_criterion = np.nan

# Define a function to create object of class ProjectileInitialConditions give a min and max angle and velocity
def set_projectile_init_con(angle_values, velocity_values, daytime_values):
    condition_array = np.zeros((len(angle_values), len(velocity_values), len(daytime_values)), dtype=ProjectileInitialConditions)

    for idx_angle, angle in enumerate(angle_values):
        for idx_velocity, velocity in enumerate(velocity_values):
            for idx_daytime, daytime in enumerate(daytime_values):

                condition_array[idx_angle, idx_velocity, idx_daytime] = ProjectileInitialConditions(angle, velocity, daytime)

    return condition_array

def determine_full_path(file_name):
    file_path = os.path.dirname(os.path.abspath('ex_3_1.py'))
    file_path = file_path + "\\\Direct_to_sun_shot\\" + file_name
    return file_path

# load initil positions and masses from file
def load_solar_system_file(file_name):
    file_path = determine_full_path(file_name)
    data = np.load(file_path)

    names = data["names"] # names of the orbiting bodies
    x_init = data["x_init"] # initial positions of the orbiting bodies in m
    v_init = data["v_init"] # initial velocities of the orbiting bodies in m/s
    m = data["m"] # masses of the orbiting bodies in kg
    radius = data["radius"] # radii of the orbiting bodies in m
    g = data["g"] # one gravitational constant for all bodies

    return names, x_init, v_init, m, radius, g


###### Vector operations ######
def get_connection_vector(x_array, idx_first_object, idx_second_object):
    pos1 = x_array[:,idx_first_object]
    pos2 = x_array[:,idx_second_object]
    return pos1 - pos2


def sol_step(t_max, x_init, v_init, masses, gravity_const, radius, use_drag, use_solar_force):
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

    euf.set_event_functions()
    args_tuple = masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj, use_drag, use_solar_force
 
    event_tuple = euf.detect_collision_projectile, euf.detect_slow_orbits, euf.detect_inner_solar_apex, euf.detect_outer_solar_system
    # Method comparison:
    # Time: RK45 (30.50s), RK23 (168.53s), DOP853 (31.15s), Radau (210.90s), BDF (75.36s), LSODA (25.92s)
    sol = solve_ivp(euf.update_function, time_interval, x_v_init_flat, args=args_tuple, first_step=1e-10, \
        events=event_tuple, rtol=1e-7)

    number_of_steps = len(sol.t)

    stop_criterion = euf.get_stop_criterion(sol, idx_coll_obj)

    trajectories = euf.get_first_half_of_array(sol.y)
    trajectories = trajectories.reshape(( n_dim, n_bodies, number_of_steps ))

    euf.get_first_half_of_array(sol.y).reshape((n_dim, n_bodies, len(sol.t)))
    return trajectories, number_of_steps, stop_criterion



###### Solar system simulation functions ######
def get_init_x_and_v(x_init, v_init, x_init_delta, v_init_delta, idx_earth, conditions):
    idx_projectile = x_init.shape[1]-1
    x_init_new = np.copy(x_init)
    v_init_new = np.copy(v_init)
    
    x_init_new[0:2,idx_projectile] = x_init_new[0:2,idx_projectile] + x_init_delta
    v_init_new[0:2,idx_projectile] = v_init_new[0:2,idx_projectile] + v_init_delta
    
    connection_vector = x_init_new[0:2,idx_projectile] - x_init_new[0:2,idx_earth]
    
    v_init_new[0:2,idx_projectile] = v_init[0:2,idx_projectile] + calc_initial_velocity(conditions, connection_vector)
    return x_init_new, v_init_new

def single_simulation(x_init, v_init, m, g, t_max, radius, condition_array, use_drag, use_solar_force):
    number_of_daytimes = condition_array.shape[2]
    idx_earth = 1
    numb_of_simulations = condition_array.size
    t0 = time.time() # start clock for timing
    # show_figs = False
    show_figs = True

    for numb, conditions in enumerate(condition_array.flatten()):
        x_init_delta, v_init_delta = calc_init_pos_and_rest_speed(conditions.daytime)

        x_init_new, v_init_new = get_init_x_and_v(x_init, v_init, x_init_delta, v_init_delta, idx_earth, conditions)

        x_trajec, number_of_steps, stop_criterion = sol_step(t_max, x_init_new, v_init_new, m, g, radius, 
            use_drag, use_solar_force)

        print("Sim. #", numb, "of ", numb_of_simulations, "# of steps: ", number_of_steps, " daytime: ", conditions.daytime,
                "angle: ", np.round(conditions.angle, 4), "velocity: ", np.round(conditions.velocity, 3))

        conditions.min_distance_to_sun = calc_min_distance_to_sun(x_trajec, radius)
        conditions.stop_criterion = stop_criterion

        # Plotting.plot_trajectories_geocentric(x_trajec, names, "trajectories_geocentric", show_figs)
        # Plotting.plot_trajectories_solarcentric(x_trajec, names, "trajectories_solarcentric", show_figs)

    print("Time elapsed: ", time.time()-t0, "seconds")
    
    for idx_daytime in range(number_of_daytimes):
        daytime = condition_array[0,0,idx_daytime].daytime
        Plotting.plot_min_distance_to_sun(condition_array, idx_daytime, daytime, show_figs, use_drag, use_solar_force)
        Plotting.plot_stop_criterion(condition_array, idx_daytime, daytime, show_figs, use_drag, use_solar_force)

# daytime=0 => midnight (backside of earth), daytime=12 => noon (frontside of earth), daytime=6, 18 => sunrise, sunset
def calc_init_pos_and_rest_speed(daytime):
    offset = 1000 # Altitude offset for projectile in m, majorly influences drag force
    x = (slc.earth_radius_in_meter + offset) * np.cos(daytime * np.pi / 12) # fraction shortened, was 2*Pi/24
    y = (slc.earth_radius_in_meter + offset) * np.sin(daytime * np.pi / 12)

    earth_rotation_speed = slc.rotation_speed_earth_in_au_year * AU

    vx = - earth_rotation_speed * np.sin(daytime * np.pi / 12)
    vy = earth_rotation_speed * np.cos(daytime * np.pi / 12)

    init_pos_delta = np.array([x, y])
    init_rest_speed_delta = np.array([vx, vy])
    return init_pos_delta, init_rest_speed_delta

# Calculate inital velocity vector of the projectile that needs to be added to its idle state
def calc_initial_velocity(conditions, connection_vector):
    start_angle_deg = conditions.angle
    projec_velocity = conditions.velocity
    start_angle_rad = start_angle_deg * np.pi / 180
    angle_earth_center_of_mass_to_proj = np.arctan2(connection_vector[1], connection_vector[0])
    start_angle_rad = angle_earth_center_of_mass_to_proj + np.pi/2 - start_angle_rad

    v_x = projec_velocity * np.cos(start_angle_rad)
    v_y = projec_velocity * np.sin(start_angle_rad)

    init_velocity_delta = np.array([v_x, v_y])
    return init_velocity_delta

# Calculate the minimal distance between the projectile and the sun
def calc_min_distance_to_sun(x_trajec, radius):
    space_dim, n, t_max = x_trajec.shape
    idx_sun = 0
    idx_proj = n-1
    delta_trajec = x_trajec[:,idx_sun,:] - x_trajec[:,idx_proj,:]
    dist_proj_sun = np.linalg.norm(delta_trajec, axis=0)
    min_distance_to_sun_center_of_mass = np.min(dist_proj_sun, axis=0)
    return min_distance_to_sun_center_of_mass


def set_parameter_space():
    km_to_m = 1000
    
    min_angle =   0
    max_angle = 180
    min_velocity =  10*km_to_m
    max_velocity = 100*km_to_m
    min_daytime = 18
    max_daytime = 18
    number_of_daytimes = 2
    number_of_angles =  4
    number_of_velocity_steps =  4
    
    # Optimal low energy trajectory
    # min_angle =  90
    # max_angle =  90
    # min_velocity =   59.7*km_to_m
    # max_velocity =   59.7*km_to_m
    # min_daytime = 18
    # max_daytime = 18
    # number_of_daytimes = 1
    # number_of_angles = 1
    # number_of_velocity_steps = 1
    
    angle_values = np.linspace(min_angle, max_angle, number_of_angles)
    velocity_values = np.linspace(min_velocity, max_velocity, number_of_velocity_steps)
    daytime_values = np.linspace(min_daytime, max_daytime, number_of_daytimes)
    
    condition_array = set_projectile_init_con(angle_values, velocity_values, daytime_values)
    return condition_array

# TODO: Extract the following procedure into a separate function
if __name__ == "__main__":

    names, x_init, v_init, m, radius, g = load_solar_system_file("solar_system_projectile_radius_wo_mars_SI.npz")

    use_drag = False
    use_solar_force = True

    condition_array = set_parameter_space()

    # Set print options for numpy
    np.set_printoptions(precision=7, suppress=True, linewidth=200)

    prefac = 1.05

    t_max = prefac*slc.year_in_seconds # maximum time in years

    single_simulation(x_init, v_init, m, g, t_max, radius, condition_array, use_drag, use_solar_force)

# TODO: 
# Implementierung:
# 1. Daten berechnung vom Plotten entkoppeln
# 2. Realistischer Start
# 3. Laufzeitoptimierung
# 4. Outsource all functions that are not directly called by solve_ivp

# 1. Implement Plot that shows speed over time for the lowest energy trajectory for 
# A. The first view seconds
# B. The whole trajectory (Gives 2 insights: 1. Overall time, 2. Speed up through sun gravity in comparison to atmospheric drag)

# 2. Run calculations at daytime 17 and 19 with 40x40 grid with drag.

# 3. Run calculation for daytime 18 in 80x80 gird with drag and solar force. (Done!)

# 4. Run calculations with daytime range 0 to 21 in 8 steps with drag and solar force

# 5. Comparison of projectile masses and radii. I assume that this does only have significant impact on the drag force.
# A. m=   0.1 t, r= 0.14 m
# B. m=   1   t, r= 0.30 m (current)
# C. m=  10   t, r= 0.65 m
# D. m= 100   t, r= 1.40 m

# 6. Plots of different types of trajectories (for of every stop criterion)


