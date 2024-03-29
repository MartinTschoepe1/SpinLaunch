#!/usr/bin/env python3

import numpy as np
import os
import scipy.constants
import ex_3_1
import spin_launch_constants as slc

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    old_file_name = "solar_system.npz"
    old_file_path = ex_3_1.determine_full_path(old_file_name)
    data = np.load(old_file_path)

    names = data["names"] # names of the orbiting bodies
    x_init = data["x_init"]
    v_init = data["v_init"] 
    m = data["m"] # masses of the orbiting bodies
    g = data["g"] # one gravitational constant for all bodies

    # Add planet Mercury to the system
    names = np.append(names, "Mercury")
    x_init_mercury = np.array([[0.3876], [0.0]])
    x_init = np.concatenate((x_init, x_init_mercury), axis=1)
    v_init_mercury = np.array([[0.0], [8.19]])
    v_init = np.concatenate((v_init, v_init_mercury), axis=1)
    m = np.append(m, 0.0553)

    new_file_name1 = "solar_system_mercury.npz"
    new_file_path1 = ex_3_1.determine_full_path(new_file_name1)
    np.savez(new_file_path1, names=names, x_init=x_init, v_init=v_init, m=m, g=g)

    # Add projectile to the system
    names = np.append(names, "Projectile")
    # 4.26343e-5 is the radius of the earth in AU
    x_init_projectile = np.array([[1.0 ], [0.0]])
    x_init = np.concatenate((x_init, x_init_projectile), axis=1)
    # 0.09779 is the velocity of the earth rotation in AU/year
    v_init_projectile = np.array([[0.0], [6.28318531]])
    v_init = np.concatenate((v_init, v_init_projectile), axis=1)
    m = np.append(m, 1.67443e-22)

    new_file_name2 = "solar_system_mercury_projectile.npz"
    new_file_path2 = ex_3_1.determine_full_path(new_file_name2)
    np.savez(new_file_path2, names=names, x_init=x_init, v_init=v_init, m=m, g=g)

    # Information about the radius of all objects

    au = scipy.constants.au

    radius = np.array([])
    radius = np.append(radius, 6.96342e8) # Add radius of sun in AU
    radius = np.append(radius, slc.earth_radius_in_meter) # Radius of earth
    radius = np.append(radius, 1737.1e3) # Radius of moon
    radius = np.append(radius, 3396.2e3) # Radius of mars
    radius = np.append(radius, 69911e3 ) # Radius of Jupiter
    radius = np.append(radius, 6051.8e3) # Radius of Venus
    radius = np.append(radius, 2439.7e3) # Radius of Mercury
    radius = np.append(radius, 1       ) # Radius of projectile

    new_file_name3 = "solar_system_projectile_radius.npz"
    new_file_path3 = ex_3_1.determine_full_path(new_file_name3)
    np.savez(new_file_path3, names=names, x_init=x_init, v_init=v_init, m=m, radius=radius, g=g)

    # Remove mars from the system
    names = np.delete(names, 3)
    x_init = np.delete(x_init, 3, axis=1)
    v_init = np.delete(v_init, 3, axis=1)
    m = np.delete(m, 3)
    radius = np.delete(radius, 3)
    
    new_file_name4 = "solar_system_projectile_radius_wo_mars.npz"
    new_file_path4 = ex_3_1.determine_full_path(new_file_name4)
    np.savez(new_file_path4, names=names, x_init=x_init, v_init=v_init, m=m, radius=radius, g=g)

    # Convert the system to SI units
    x_init = x_init * au
    v_init = v_init * au / slc.year_in_seconds
    m = m * slc.earth_mass
    g = 6.67430e-11

    new_file_name5 = "solar_system_projectile_radius_wo_mars_SI.npz"
    new_file_path5 = ex_3_1.determine_full_path(new_file_name5)
    np.savez(new_file_path5, names=names, x_init=x_init, v_init=v_init, m=m, radius=radius, g=g)


