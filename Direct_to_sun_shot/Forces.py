import time
import numpy as np
import spin_launch_constants as slc



###### Force calculation functions ######
def forces(x, v, masses, g, use_drag, use_solar_force):
    space_dim, *n_shape = x.shape # dimension of space, number of bodies
    # print("n_shape: ", n_shape)
    v = v.reshape((space_dim, *n_shape))
    idx_sun = 0
    idx_eath = 1
    idx_projectile = n_shape[0]-1

    masses1 = np.repeat(masses[:, np.newaxis], n_shape[0], axis=1)
    masses2 = np.repeat(masses[np.newaxis,:], n_shape[0], axis=0)

    if (len(n_shape) == 2):
        masses1 = np.repeat(masses1[:, :, np.newaxis], n_shape[1], axis=2)
        masses2 = np.repeat(masses2[:, :, np.newaxis], n_shape[1], axis=2)
    elif (len(n_shape) == 1):
        pass
    else:
        print("Error: n_shape has wrong length")

    F = gravitational_force(x, g, masses1, masses2, n_shape[0])
    
    #TODO: Outsourcing to calc only once
    # Effective cross-sectional area of the projectile (to be adjusted based on the shape)
    radius = 0.30 # m
    A = radius**2 * np.pi # m^2

    if (use_drag[0]): add_drag_force(x, idx_projectile, idx_eath, v, F, A, use_drag)

    if (use_solar_force[0]): add_solar_force(x, idx_projectile, idx_sun, A, F)

    # Switch axis 0 and axis 1 to get the correct shape of F
    # F = F.swapaxes(0, 1)
 
    return F.T / masses

# Compare with wikipedia: https://de.wikipedia.org/wiki/Solarkonstante
def solar_flux(distance_to_sun):
    solar_luminosity = 3.828e26  # Solar luminosity in watts
    solar_flux = solar_luminosity / (4 * np.pi * distance_to_sun**2)
    return solar_flux

def force_radiation(r_ij, solar_flux, A):
    # Calculate the magnitude of the distance vector
    # r_mag = np.linalg.norm(r_ij)
    
    # Calculate the radiation pressure force
    radiation_force = solar_flux * A / slc.c * r_ij #/ r_mag
    return radiation_force

# Calculate radiation pressure force on the projectile
def add_solar_force(x, idx_projectile, idx_sun, A, F):
    distance_vector_sun = x[:, idx_projectile] - x[:, idx_sun]
    distance_vector_sun_norm = fast_2D_norm(distance_vector_sun)
    # distance_vector_sun_norm = np.linalg.norm(distance_vector_sun)

    solar_flux_value = solar_flux(distance_vector_sun_norm)
    
    delta_F_radiation_projec = force_radiation(distance_vector_sun/distance_vector_sun_norm, solar_flux_value, A)
    F[idx_projectile, :] += delta_F_radiation_projec

def fast_2D_norm(x):
    return np.sqrt(x[0]**2 + x[1]**2)

def gravitational_force(x, g, masses1, masses2, n):
    r_ij = x[:, :, np.newaxis] - x[:, np.newaxis, :]
    abs_r_ij = np.linalg.norm(r_ij, axis=0)

    # Handle division by zero on the diagonal by setting the diagonal to 1
    abs_r_ij[np.arange(n), np.arange(n)] = 1

    r_ij_unit = r_ij / abs_r_ij
    delta_F_gravity = g * masses1 * masses2 / abs_r_ij**2 * r_ij_unit
    delta_F_gravity[:, np.arange(n), np.arange(n)] = 0

    sum_delta_F_gravity = np.sum(delta_F_gravity, axis=1)
    F = sum_delta_F_gravity.swapaxes(0, 1)
    return F


def add_drag_force(x, idx_projectile, idx_eath, v, F, A, use_drag):
    dist_proj_earth = fast_2D_norm(x[:,idx_projectile] - x[:,idx_eath])
    altitude_in_m = dist_proj_earth - slc.earth_radius_in_meter
    # altitude_in_m = max(altitude_in_m, 0)
    # correction for above line, as "0" has to be array
    altitude_in_m = np.maximum(altitude_in_m, 0)
    
    use_drag = np.where(altitude_in_m > 200000, False, use_drag)

    relative_velocity = v[:,idx_projectile] - v[:,idx_eath]
    delta_F_drag_projec = force_drag(relative_velocity, altitude_in_m, A)
    F[idx_projectile,:] += delta_F_drag_projec

# Warning untested ChatGPT code! Please check if it works as expected!
def air_density(altitude):    
    # Calculate air density using the exponential decay model
    rho = slc.rho0 * np.exp(-altitude / slc.H)
    
    return rho

# Warning untested ChatGPT code! Please check if it works as expected!
def force_drag(v_i, altitude, A):
    # Calculate air density at the given altitude
    if ((altitude < 0).any()):
        print("Warning: altitude is negative: ", altitude)
    rho = air_density(altitude)
    shape = v_i.shape

    if (rho == 0.0):
        return np.zeros(shape)
    else:
        # Calculate velocity magnitude
        cd_constant = 0.5 # unitless
        v_mag = np.linalg.norm(v_i)

        # Calculate drag force using the constant drag coefficient and altitude-dependent air density
        drag_force = -0.5 * rho * v_mag**2 * cd_constant * A * (v_i / v_mag)

        return drag_force
