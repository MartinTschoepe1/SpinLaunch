#!/usr/bin/env python3

import numpy as np
import os
import scipy.constants
import ex_3_1

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
    x_init_projectile = np.array([[1.0 + 4.26343e-5], [0.0]])
    x_init = np.concatenate((x_init, x_init_projectile), axis=1)
    v_init_projectile = np.array([[0.0], [6.28318531 + 0.09779]])
    v_init = np.concatenate((v_init, v_init_projectile), axis=1)
    m = np.append(m, 1.67443e-22)

    new_file_name2 = "solar_system_mercury_projectile_2D.npz"
    new_file_path2 = ex_3_1.determine_full_path(new_file_name2)
    np.savez(new_file_path2, names=names, x_init=x_init, v_init=v_init, m=m, g=g)




