import numpy as np
import spin_launch_constants as slc



###### Force calculation functions ######
def calc_massless_forces(x, v, masses, gravity_const, use_drag, use_solar_force):
    orig_forces = forces(x, v, masses, gravity_const, use_drag, use_solar_force).transpose()
    massless_forces = orig_forces / masses[np.newaxis,:]
    return massless_forces

def force_radiation(r_ij, solar_flux, A):
    # Calculate the magnitude of the distance vector
    r_mag = np.linalg.norm(r_ij)
    
    # Calculate the radiation pressure force
    radiation_force = solar_flux * A / slc.c * r_ij / r_mag
    if (np.linalg.norm(radiation_force > 1)):
        print("Warning: Radiation force is larger than 1 Newton: ", radiation_force)
    return radiation_force

# Compare with wikipedia: https://de.wikipedia.org/wiki/Solarkonstante
def solar_flux(distance_to_sun):
    solar_luminosity = 3.828e26  # Solar luminosity in watts
    solar_flux = solar_luminosity / (4 * np.pi * distance_to_sun**2)
    return solar_flux

# Calculate radiation pressure force on the projectile
def add_solar_force(x, idx_projectile, idx_sun, A, F):
    distance_vector_sun = x[:, idx_projectile] - x[:, idx_sun]

    solar_flux_value = solar_flux(np.linalg.norm(distance_vector_sun))
    
    delta_F_radiation_projec = force_radiation(distance_vector_sun, solar_flux_value, A)
    F[idx_projectile, :] += delta_F_radiation_projec

# Code in this function was restrucutred after Warning: overflow encountered in scalar power.
def gravitational_force(r_ij, m_i, m_j, g):
    abs_r_ij = np.linalg.norm(r_ij)
    frac1 = m_i / abs_r_ij
    frac2 = m_j / abs_r_ij
    frac3 = r_ij / abs_r_ij
    force = - g * frac1 * frac2 * frac3
    return force

def add_drag_force(x, idx_projectile, idx_eath, v, F, A):
    dist_proj_earth = np.linalg.norm(x[:,idx_projectile] - x[:,idx_eath])
    altitude_in_m = dist_proj_earth - slc.earth_radius_in_meter
    if (altitude_in_m < 0):
        altitude_in_m = 0
    
    relative_velocity = v[:,idx_projectile] - v[:,idx_eath]
    delta_F_drag_projec = force_drag(relative_velocity, altitude_in_m, A)
    F[idx_projectile,:] += delta_F_drag_projec

def forces(x, v, masses, g, use_drag, use_solar_force):
    space_dim, n = x.shape # number of bodies, dimension of space
    v = v.reshape((space_dim, n))
    idx_sun = 0
    idx_eath = 1
    idx_projectile = n-1

    F = np.zeros((n, space_dim)) # array to store forces (6 bodies, 2 dimensions)
    for i in range(n): 
        for j in range(n):
            # if i != j: # do not calculate force of body on itself
            if i < j: # do not calculate force of body on itself
                distance_vector = (x[:,i] - x[:,j]) # vector pointing from body j to body i
                delta_F_gravity = gravitational_force(distance_vector, masses[i], masses[j], g)
                F[i,:] = F[i,:] + delta_F_gravity
                F[j,:] = F[j,:] - delta_F_gravity

    #TODO: Outsourcing to calc only once
    # Effective cross-sectional area of the projectile (to be adjusted based on the shape)
    radius = 0.30 # m
    A = radius**2 * np.pi # m^2

    if (use_drag): add_drag_force(x, idx_projectile, idx_eath, v, F, A)

    if (use_solar_force): add_solar_force(x, idx_projectile, idx_sun, A, F)
 
    return F

# Warning untested ChatGPT code! Please check if it works as expected!
def air_density(altitude):    
    # Calculate air density using the exponential decay model
    rho = slc.rho0 * np.exp(-altitude / slc.H)
    
    return rho

# Warning untested ChatGPT code! Please check if it works as expected!
def force_drag(v_i, altitude, A):
    # Calculate air density at the given altitude
    if (altitude < 0):
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
