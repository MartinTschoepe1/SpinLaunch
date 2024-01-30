#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

def force(mass, gravity_const):
    force_x = 0.0
    force_y = -mass * gravity_const
    return np.array([force_x, force_y])

# Define the integration step for the Euler method
def step_euler(x, v, dt, mass, gravity_const, f):
    x_new = x + v * dt
    v_new = v + f(mass, gravity_const) * dt
    return x_new, v_new

# Generalized function to plot the trajectory of the ball in different planes.
def plot_trajectory(trajec_array, x_label, y_label, file_name, axis1, axis2):
    for index, trajec in enumerate(trajec_array):
        label_mass = "Mass = " + str(mass_list[index])
        plt.plot(trajec[:,axis1], trajec[:,axis2], label=label_mass)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    file_path = os.path.dirname(os.path.abspath('ex_2_1.py'))
    file_path = file_path + "\\Exercises\\Exercise2\\" + file_name
    plt.savefig(file_path)
    plt.show()

def Calc_trajec_for_all_masses(init_pos_vec, init_vel_vec, start_time, force, step_euler, mass_list, gravity_const, dt, trajec_list):
    for mass in mass_list:
        trajec = []
        Calc_trajec_for_one_mass(init_pos_vec, init_vel_vec, start_time, force, step_euler, mass, gravity_const, dt, trajec)
        trajec_list.append(trajec)

def Calc_trajec_for_one_mass(pos_vec, velo_vec, time, force, step_euler, mass, gravity_const, dt, trajec):
    for t in np.arange(time, 20, dt):
        pos_vec_list = pos_vec.tolist()
        pos_vec_list.append(t)
        pos_and_time = np.array(pos_vec_list)
        trajec.append(pos_and_time)
        pos_vec, velo_vec = step_euler(pos_vec, velo_vec, dt, mass, gravity_const, force)
        if pos_vec[1] <= 0.0:
            break
    else:
        print("Warning: Maximum time was too small!")
    
    trajec.append(Interpolate(trajec[-2][0], trajec[-1][0], trajec[-2][1], trajec[-1][1], trajec[-2][2], trajec[-1][2]))

# Interpolate last two points of trajectory to determine position of ball hitting the ground.
def Interpolate(x1, x2, y1, y2, time1, time2):
    if x1 == x2 and time1 == time2:
        print("Warning: x1 == x2 and time1 == time2")
        return x1, y1, time1
    if x1 == x2 and time1 != time2 or x1 != x2 and time1 == time2:
        raise
        ValueError("Interpolation not possible!")

    x_y_slope = (y2 - y1) / (x2 - x1)
    t_y_slope = (y2 - y1) / (time2 - time1)

    if x_y_slope == 0.0:
        print("Warning: x_y_slope == 0.0")
        x3 = x1
    else:
        x3 = x1 - y1 / x_y_slope
    y3 = 0.0
    if t_y_slope == 0.0:
        print("Warning: t_y_slope == 0.0")
        time3 = time1
    else:
        time3 = time1 - y1 / t_y_slope

    return x3, y3, time3

def Set_variables():
    start_time = 0.0
    init_pos_vec = np.array([0.0, 0.0])
    init_vel_vec = np.array([50.0, 50.0])

    mass_list = [1.0, 2.0, 5.0]
    gravity_const = 9.81
    dt = 0.01
    return start_time,init_pos_vec,init_vel_vec,mass_list,gravity_const,dt

if __name__ == "__main__":
    start_time, init_pos_vec, init_vel_vec, mass_list, gravity_const, dt = Set_variables()
    
    # Define empty lists
    trajec_list = []
    trajec_array = []

    # Loop over time and break, if the ball hits the ground
    Calc_trajec_for_all_masses(init_pos_vec, init_vel_vec, start_time, force, step_euler, mass_list, gravity_const, dt, trajec_list)
    
    # Convert trajectory to numpy array
    for trajec in trajec_list:
        trajec_array.append(np.array(trajec))

    # Write the first trajectory in file.
    file_path = os.path.dirname(os.path.abspath('ex_2_1.py'))
    file_path = file_path + "\\Exercises\\Exercise2\\trajectory.txt"
    np.savetxt(file_path, trajec_array[0], delimiter="\t")

    plot_trajectory(trajec_array, "x [m]", "y [m]", "trajectory_x_y.pdf", 0, 1)
    plot_trajectory(trajec_array, "time [s]", "x [m]", "trajectory_x_t.pdf", 2, 0)
    plot_trajectory(trajec_array, "time [s]", "y [m]", "trajectory_y_t.pdf", 2, 1)