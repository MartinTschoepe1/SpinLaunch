#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os

start_time = 0.0
init_pos_vec = np.array([0.0, 0.0])
init_vel_vec = np.array([50.0, 50.0])

def force(mass, gravity_const):
    force_x = 0.0
    force_y = -mass * gravity_const
    return np.array([force_x, force_y])

# Define the integration step for the Euler method
def step_euler(x, v, dt, mass, gravity_const, f):
    x_new = x + v * dt
    v_new = v + f(mass, gravity_const) * dt
    return x_new, v_new

# Plot the trajectory of the ball in x-y-plane
def Plot_x_y(trajec_array):
    for index, trajec in enumerate(trajec_array):
        plt.plot(trajec[:,0], trajec[:,1], label="Mass = " + str(mass_list[index]))
    
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    file_path = os.path.dirname(os.path.abspath('ex_2_1.py'))
    file_path = file_path + "\\Exercises\\Exercise2\\trajectory_x_y.pdf"
    plt.savefig(file_path)
    plt.show()

# Plot the trajectory of the ball in x-t-plane
def Plot_x_t(trajec_array):
    for index, trajec in enumerate(trajec_array):
        plt.plot(trajec[:,2], trajec[:,0], label="Mass = " + str(mass_list[index]))
    
    plt.xlabel("time [s]")
    plt.ylabel("x [m]")
    plt.legend()
    file_path = os.path.dirname(os.path.abspath('ex_2_1.py'))
    file_path = file_path + "\\Exercises\\Exercise2\\trajectory_x_t.pdf"
    plt.savefig(file_path)
    plt.show()

# Plot the trajectory of the ball in y-t-plane
def Plot_y_t(trajec_array):
    for index, trajec in enumerate(trajec_array):
        plt.plot(trajec[:,2], trajec[:,1], label="Mass = " + str(mass_list[index]))
    
    plt.xlabel("time [s]")
    plt.ylabel("y [m]")
    plt.legend()
    file_path = os.path.dirname(os.path.abspath('ex_2_1.py'))
    file_path = file_path + "\\Exercises\\Exercise2\\trajectory_y_t.pdf"
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
        # print(pos_and_time)
        trajec.append(pos_and_time)
        pos_vec, velo_vec = step_euler(pos_vec, velo_vec, dt, mass, gravity_const, force)
        if pos_vec[1] <= 0.0:
            break
    else:
        print("Warning: Maximum time was too small!")
    
    trajec.append(Interpolate(trajec[-2][0], trajec[-1][0], trajec[-2][1], trajec[-1][1], trajec[-2][2]))

# Interpolate last two points of trajectory to determine position of ball hitting the ground.
def Interpolate(x1, x2, y1, y2, time1):
    x_y_slope = (y2 - y1) / (x2 - x1)
    y_t_slope = (time1 - time1) / (y2 - y1)
    x3 = x1 - y1 / x_y_slope
    y3 = 0.0
    time3 = time1 - y1 / y_t_slope
    #time3 = time1 - y1 / x_y_slope
    return x3, y3, time3

if __name__ == "__main__":
    mass_list = [1.0, 2.0, 5.0]
    gravity_const = 9.81
    dt = 0.01
    
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

    # Plot the trajectory and create PDF file
    Plot_x_y(trajec_array)
    Plot_x_t(trajec_array)
    Plot_y_t(trajec_array)
