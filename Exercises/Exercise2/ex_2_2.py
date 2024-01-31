#!/usr/bin/env python3

import numpy as np
import os

import ex_2_1

# Introduce a class for parameters
class Parameters:
    def __init__(self, mass, gamma, v_wind):
        self.mass = mass # Mass of the ball in kg
        self.gamma = gamma # Friction constant in kg/s
        self.v_wind = v_wind # Wind speed in m/s

def Calc_trajec_set(init_pos_vec, init_vel_vec, start_time, force_fric_wind, step_euler, parameter_list, gravity_const, dt, trajec_list):
    for parameters in parameter_list:
        trajec = []
        Calc_trajec_single(init_pos_vec, init_vel_vec, start_time, force_fric_wind, step_euler, parameters, gravity_const, dt, trajec)
        trajec_list.append(trajec)

def Calc_trajec_single(pos_vec, velo_vec, time, force_fric_wind, step_euler, parameters, gravity_const, dt, trajec):
    for t in np.arange(time, 20, dt):
        pos_vec_list = pos_vec.tolist()
        pos_vec_list.append(t)
        pos_and_time = np.array(pos_vec_list)
        trajec.append(pos_and_time)
        pos_vec, velo_vec = step_euler(pos_vec, velo_vec, dt, parameters, gravity_const, force_fric_wind)
        if pos_vec[1] <= 0.0:
            break
    else:
        print("Warning: Maximum time was too small!")
    
    trajec.append(ex_2_1.Interpolate(trajec[-2][0], trajec[-1][0], trajec[-2][1], trajec[-1][1], trajec[-2][2], trajec[-1][2]))        

def step_euler(x, v, dt, parameters, gravity_const, f):
    x_new = x + v * dt
    v_new = v + f(parameters.mass, gravity_const, v, parameters.gamma, parameters.v_wind ) * dt
    return x_new, v_new

# Force insidering gravity, friction and wind.
def force_fric_wind(mass, gravity_const, v_object, gamma, v_wind):
    return ex_2_1.force(mass, gravity_const) - gamma * (v_object - v_wind)

# Generalized function to plot the trajectory of the ball in different planes.
def plot_trajectory(trajec_array, x_label, y_label, file_name, axis1, axis2, parameter_list):
    for index, trajec in enumerate(trajec_array):
        #create label for legend with all parameter values
        label_param = "Mass = " + str(parameter_list[index].mass) + \
            ", gamma = " + str(parameter_list[index].gamma) + \
            ", v_wind = " + str(parameter_list[index].v_wind)
        plt.plot(trajec[:,axis1], trajec[:,axis2], label=label_param)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, 90)
    plt.legend()
    file_path = ex_2_1.determine_full_path(file_name)
    plt.gcf().set_size_inches(12, 7)
    plt.savefig(file_path)
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    start_time, init_pos_vec, init_vel_vec, mass_list, gravity_const, dt = ex_2_1.Set_variables()

    parameter_list = []
    # Create object of class Parameters
    parameter_list.append(Parameters(2.0, 0.0, np.array([0.0, 0.0])))
    parameter_list.append(Parameters(2.0, 0.1, np.array([0.0, 0.0])))
    parameter_list.append(Parameters(2.0, 0.1, np.array([-50.0, 0.0])))
    parameter_list.append(Parameters(2.0, 0.1, np.array([-100.0, 0.0])))
    parameter_list.append(Parameters(2.0, 0.1, np.array([-200.0, 0.0])))
    parameter_list.append(Parameters(2.0, 0.1, np.array([-250.0, 0.0])))

    trajec_list = []

    Calc_trajec_set(init_pos_vec, init_vel_vec, start_time, force_fric_wind, step_euler, \
        parameter_list, gravity_const, dt, trajec_list)
    
    trajec_list = ex_2_1.Convert_list_of_lists_to_list_of_nparraies(trajec_list)

    plot_trajectory(trajec_list, "x [m]", "y [m]", "ex_2_2_xy.pdf", 0, 1, parameter_list)


    

    
