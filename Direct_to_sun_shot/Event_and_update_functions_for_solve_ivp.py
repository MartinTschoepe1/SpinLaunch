import numpy as np

import spin_launch_constants as slc
import Forces

###### Vector operations ######
def get_connection_vector(x_array, idx_first_object, idx_second_object):
    pos1 = x_array[:,idx_first_object]
    pos2 = x_array[:,idx_second_object]
    return pos1 - pos2



# ###### Update function and event functions for solve_ivp ######
# def update_function(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj, \
#         use_drag, use_solar_force):
#     x_array = get_position_array(y, n_dim, n_bodies)
#     v_vec = get_second_half_of_array(y)
#     dvdt = Forces.forces(x_array, v_vec, masses, gravity_const, use_drag, use_solar_force).flatten()
#     res = np.concatenate((v_vec, dvdt))
#     return res

# def detect_collision_projectile(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj, \
#         use_drag, use_solar_force):
#     x_array = get_position_array(y, n_dim, n_bodies)

#     idx_first_object = n_bodies-1
#     pos_proj  = x_array[:,idx_first_object] # current position of projectile
#     product_for_sign_change_criterion = 1.0

#     for idx in range(idx_first_object):
#         pos_object = x_array[:,idx]
#         distance_between_surfaces = np.linalg.norm(pos_proj-pos_object) - radius[idx] + radius[idx_first_object]
#         if (distance_between_surfaces < 0):
#             idx_coll_obj[0] = idx
#         product_for_sign_change_criterion *= distance_between_surfaces

#     return product_for_sign_change_criterion

# # Detect when projectile is still within moon orbit after one week. Event is terminal. Run time optimization. 
# def detect_slow_orbits(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj, \
#         use_drag, use_solar_force):
#     x_array = get_position_array(y, n_dim, n_bodies)

#     idx_first_object = n_bodies-1
#     idx_of_earth = 1
#     idx_of_moon = 2
#     six_days_in_seconds = 6.0 * 24 * 60 * 60
#     pos_proj  = x_array[:,idx_first_object]
#     pos_earth = x_array[:,idx_of_earth]
#     pos_moon  = x_array[:,idx_of_moon]
    
#     is_projectile_within_moon_orbit = np.linalg.norm(pos_proj-pos_earth) < np.linalg.norm(pos_moon-pos_earth)
#     is_within_six_days = t < six_days_in_seconds

#     if (is_within_six_days):
#         return 1
#     else:
#         if (is_projectile_within_moon_orbit):
#             return -1
#         else:
#             return 1

# # Helper function that is only indirectly called by solve_ivp via detect_inner_solar_apex and detect_outer_solar_apex
# def detect_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, \
#         use_drag, use_solar_force):
#     x_array = get_position_array(y, n_dim, n_bodies)
#     v_array = get_velocity_array(y, n_dim, n_bodies)

#     idx_first_object = n_bodies-1
#     idx_second_object = 0
#     relativ_pos_proj = get_connection_vector(x_array, idx_first_object, idx_second_object)

#     velocity_proj = v_array[:,idx_first_object]
#     velocity_sun = v_array[:,idx_second_object]

#     relativ_velocity_proj = velocity_proj - velocity_sun
#     abs_value_distance_to_sun = np.linalg.norm(relativ_pos_proj)

#     dot_product = np.dot(relativ_pos_proj, relativ_velocity_proj)
#     denominator = np.linalg.norm(relativ_pos_proj) * np.linalg.norm(relativ_velocity_proj)
#     angle_between_pos_and_velocity_rad = np.arccos(dot_product / denominator)
#     angle_degree = angle_between_pos_and_velocity_rad * 180 / np.pi - 90

#     return angle_degree

# # Detect inner apex of projectile trajectory to increase accuracy, event is not terminal
# def detect_inner_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj, \
#         use_drag, use_solar_force):
#     angle_degree = detect_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, \
#             use_drag, use_solar_force)
#     return angle_degree

# # Detect outer apex after the first inner apex of projectile trajectory to terminate simulation, run time optimization
# def detect_outer_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj, \
#         use_drag, use_solar_force):
#     angle_degree = detect_solar_apex(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached)
#     one_week_in_years = 1.0 / 52.0
#     if (is_apex_reached[0] == False or t < one_week_in_years):
#         return 1.0
#     else:
#         return angle_degree

# def detect_outer_solar_system(t, y, masses, gravity_const, n_dim, n_bodies, radius, is_apex_reached, idx_coll_obj, \
#         use_drag, use_solar_force):
#     x_array = get_position_array(y, n_dim, n_bodies)
#     idx_first_object = n_bodies-1
#     idx_second_object = 0
#     distance_sun_jupiter = slc.distance_sun_jupiter_in_m
#     pos_proj = x_array[:,idx_first_object]
#     pos_sun = x_array[:,idx_second_object]
#     distance_to_sun = np.linalg.norm(get_connection_vector(x_array, idx_first_object, idx_second_object))
#     return distance_to_sun - distance_sun_jupiter



def update_function(t, y, masses, gravity_const, n_shape, radius, is_apex_reached, idx_coll_obj, \
        use_drag, use_solar_force):
    x_array = get_position_array(y, *n_shape)
    v_vec = get_second_half_of_array(y)
    dvdt = Forces.forces(x_array, v_vec, masses, gravity_const, use_drag, use_solar_force).flatten()
    res = np.concatenate((v_vec, dvdt))
    return res


def detect_collision_projectile(t, y, masses, gravity_const, n_shape, radius, is_apex_reached, idx_coll_obj, \
        use_drag, use_solar_force):
    x_array = get_position_array(y, *n_shape)

    idx_first_object = n_shape[1] - 1
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
def detect_slow_orbits(t, y, masses, gravity_const, n_shape, radius, is_apex_reached, idx_coll_obj, \
        use_drag, use_solar_force):
    x_array = get_position_array(y, *n_shape)

    idx_first_object = n_shape[1] - 1
    idx_of_earth = 1
    idx_of_moon = 2
    six_days_in_seconds = 6.0 * 24 * 60 * 60
    pos_proj  = x_array[:,idx_first_object]
    pos_earth = x_array[:,idx_of_earth]
    pos_moon  = x_array[:,idx_of_moon]
    
    is_projectile_within_moon_orbit = np.linalg.norm(pos_proj-pos_earth) < np.linalg.norm(pos_moon-pos_earth)
    is_within_six_days = t < six_days_in_seconds

    if (is_within_six_days):
        return 1
    else:
        if (is_projectile_within_moon_orbit):
            return -1
        else:
            return 1

# Helper function that is only indirectly called by solve_ivp via detect_inner_solar_apex and detect_outer_solar_apex
def detect_solar_apex(t, y, masses, gravity_const, n_shape, radius, is_apex_reached, \
        use_drag, use_solar_force):
    x_array = get_position_array(y, *n_shape)
    v_array = get_velocity_array(y, *n_shape)

    idx_first_object = n_shape[1] - 1
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

# Detect inner apex of projectile trajectory to increase accuracy, event is not terminal
def detect_inner_solar_apex(t, y, masses, gravity_const, n_shape, radius, is_apex_reached, idx_coll_obj, \
        use_drag, use_solar_force):
    angle_degree = detect_solar_apex(t, y, masses, gravity_const, n_shape, radius, is_apex_reached, \
            use_drag, use_solar_force)
    return angle_degree

# Detect outer apex after the first inner apex of projectile trajectory to terminate simulation, run time optimization
def detect_outer_solar_apex(t, y, masses, gravity_const, n_shape, radius, is_apex_reached, idx_coll_obj, \
        use_drag, use_solar_force):
    angle_degree = detect_solar_apex(t, y, masses, gravity_const, n_shape, radius, is_apex_reached)
    one_week_in_years = 1.0 / 52.0
    if (is_apex_reached[0] == False or t < one_week_in_years):
        return 1.0
    else:
        return angle_degree

def detect_outer_solar_system(t, y, masses, gravity_const, n_shape, radius, is_apex_reached, idx_coll_obj, \
        use_drag, use_solar_force):
    x_array = get_position_array(y, *n_shape)
    idx_first_object = n_shape[1] - 1
    idx_second_object = 0
    distance_sun_jupiter = slc.distance_sun_jupiter_in_m
    pos_proj = x_array[:,idx_first_object]
    pos_sun = x_array[:,idx_second_object]
    distance_to_sun = np.linalg.norm(get_connection_vector(x_array, idx_first_object, idx_second_object))
    return distance_to_sun - distance_sun_jupiter


def get_stop_criterion(sol, idx_coll_obj):
    
    if (sol.status == 0):
        stop_criterion = 0 # time limit reached
    elif (sol.status == 1):
        if (sol.t_events[3].size > 0):
            stop_criterion = 1 # outer solar system reached
        elif (sol.t_events[1].size > 0):
            stop_criterion = 2 # slow orbit reached
        elif (sol.t_events[0].size > 0):
            if (idx_coll_obj[0] == 0):
                stop_criterion = 3 # sun collision
            elif (idx_coll_obj[0] == 1):
                stop_criterion = 4 # earth collision
            elif (idx_coll_obj[0] == 2):
                stop_criterion = 5 # moon collision
            elif (idx_coll_obj[0] == 5):
                stop_criterion = 6 # mercury collision
            else:
                stop_criterion = 7 # collision with other object
        else:
            stop_criterion = 8 # no stop criterion reached
    else:
        stop_criterion = 8 # no stop criterion reached
    return stop_criterion

# Set event functions for solve_ivp
def set_event_functions():
    detect_collision_projectile.terminal = True
    detect_collision_projectile.direction = 0
    detect_slow_orbits.terminal = True
    detect_slow_orbits.direction = 0
    detect_inner_solar_apex.terminal = False
    detect_inner_solar_apex.direction = -1.0
    detect_outer_solar_system.terminal = True
    detect_outer_solar_system.direction = 1.0

###### Necessary due to the 1D (poc, velo, force) vector in solve_ivp ######
def get_first_half_of_array(array):
    return array[:len(array)//2]

def get_second_half_of_array(array):
    return array[len(array)//2:]

# def get_position_array(y, n_dim, n_bodies):
#     x_vec = get_first_half_of_array(y)
#     x_array = x_vec.reshape((n_dim, n_bodies))
#     return x_array

    #same function as above, but n_dim, n_bodies is replaced by n_shape
def get_position_array(y, *n_shape):
    x_vec = get_first_half_of_array(y)
    x_array = x_vec.reshape(n_shape)
    return x_array

def get_position_array_vectorized(y, n_dim, n_bodies, n_conditions):
    x_vec = get_first_half_of_array(y)
    x_array = x_vec.reshape((n_dim, n_bodies, n_conditions))
    return x_array

def get_velocity_array(y, n_dim, n_bodies):
    v_vec = get_second_half_of_array(y)
    v_array = v_vec.reshape((n_dim, n_bodies))
    return v_array

def get_velocity_array_vectorized(y, n_dim, n_bodies, n_conditions):
    v_vec = get_second_half_of_array(y)
    v_array = v_vec.reshape((n_dim, n_bodies, n_conditions))
    return v_array
