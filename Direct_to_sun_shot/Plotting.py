import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import spin_launch_constants as slc


###### Plotting functions ######
def plot_trajectories_solarcentric(x_trajec, names, file_name, show_figs):
    space_dim, n, t_max = x_trajec.shape
    solar_radius_per_m = 1.0 / slc.solar_radius_in_meters
    x_trajec = x_trajec*solar_radius_per_m # Will not be saved after function call

    for i in range(0,n,1):
        plt.plot(x_trajec[0,i,:], x_trajec[1,i,:], label=names[i], marker='o', markersize=2)
    plt.xlabel("x [Sun radii]")
    plt.ylabel("y [Sun radii]")
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    plt.gcf().set_size_inches(12, 12)
    plt.legend()
    plt.savefig(file_name + ".pdf")
    plt.show()

def plot_trajectories_geocentric(x_trajec, names, file_name, show_figs):
    space_dim, n, t_max = x_trajec.shape
    idx_earth = 1
    idx_proj = n-1
    scale_factor = 1000
    inital_dist_proj_earth = x_trajec[:,idx_proj,0] - x_trajec[:,idx_earth,0]
    for i in range(n):
        distance_x = x_trajec[0,i,:] - x_trajec[0,idx_earth,:] - inital_dist_proj_earth[0]
        distance_y = x_trajec[1,i,:] - x_trajec[1,idx_earth,:] - inital_dist_proj_earth[1]

        plt.plot(distance_x / scale_factor, distance_y / scale_factor, label=names[i], marker='o', markersize=2)
    plt.setp(plt.gca().lines, linewidth=2)
    plt.xlabel("x [km]")
    plt.ylabel("y [km]")
    plt.xlim(-500000, 500000)
    plt.ylim(-500000, 500000)
    plt.gcf().set_size_inches(12, 12)
    plt.legend()
    plt.savefig(file_name + ".pdf")
    plt.show()

def plot_min_distance_to_sun(condition_array, idx_daytime, daytime, show_figs, use_drag, use_solar_force):
    file_name = "dist_to_sun"
    angle_array, velocity_array, daytimes_array, min_distance_to_sun_array = \
        convert_set_of_objects_into_arrays(condition_array, 'min_distance_to_sun')
    
    conv_m_to_solar_radius = 1.0 / slc.solar_radius_in_meters
    min_distance_to_sun_array = min_distance_to_sun_array*conv_m_to_solar_radius
    angle_velo_array = min_distance_to_sun_array[:,:,idx_daytime]

    ax, x, y, fig = plot_preparation(velocity_array, angle_array)
    
    c = ax.pcolormesh(x, y, angle_velo_array, cmap='viridis', \
        norm=colors.LogNorm(vmin=1, vmax=100))

    set_plot_size_and_margins(True)

    c.set_clim(1.0, 100)
    c.cmap.set_under('red')

    fig.colorbar(c, ax=ax, label='minimal distance to sun [solar radii]')
    plt.savefig(get_daytime_filename(file_name, daytime, use_drag, use_solar_force))

    show_or_close_figure(fig, show_figs)

def plot_stop_criterion(condition_array, idx_daytime, daytime, show_figs, use_drag, use_solar_force):
    file_name = "stop_criterion"
    angle_array, velocity_array, daytimes_array, stop_criterion_array = \
        convert_set_of_objects_into_arrays(condition_array, 'stop_criterion')

    angle_velo_array = stop_criterion_array[:,:,idx_daytime]
            
    ticklabels = ['time limit', 'outer solar system', 'slow orbit', \
        'sun collision', 'earth collision', 'moon collision', 'mercury collision', 'collision with other object']

    ax, x, y, fig = plot_preparation(velocity_array, angle_array)

    cmap = plt.get_cmap('tab10', 8)
    c = ax.pcolormesh(x, y, angle_velo_array, cmap=cmap,  vmin= -0.5, vmax=7.5)

    set_plot_size_and_margins(False)

    cbar = fig.colorbar(c, ax=ax, label='stop criterion', ticks=[0, 1, 2, 3, 4, 5, 6, 7])
    cbar.set_ticklabels(ticklabels)

    plt.savefig(get_daytime_filename(file_name, daytime, use_drag, use_solar_force))

    show_or_close_figure(fig, show_figs)

###### Preparation functions ######
# Creates figure with colorbar that shows the mimal distance between the projectile and sun depending on the angle and velocity
def plot_preparation(velocity_array, angle_array):
    x, y = np.meshgrid(velocity_array, angle_array)
    fig, ax = plt.subplots()
    ax.set_xlabel('velocity [m/s]')
    ax.set_ylabel('angle [°]')
    plt.gcf().set_size_inches(18, 12)
    return ax, x, y, fig

def get_daytime_filename(file_name, daytime, use_drag, use_solar_force):
    if (use_drag):
        file_name = file_name + "_with_drag"
    else:
        file_name = file_name + "_without_drag"

    if (use_solar_force):
        file_name = file_name + "_with_solar"
    else:
        file_name = file_name + "_without_solar"

    rounded_daytime = str(int(np.round(daytime)))
    return file_name + "_time_" + rounded_daytime + ".pdf"

def show_or_close_figure(fig, show_figs):
    if show_figs:
        plt.show()
    else:
        plt.close(fig)

def set_plot_size_and_margins(wide):
    if (wide):
        right_margin = 0.98
    else:
        right_margin = 0.80
    
    plt.gcf().set_size_inches(8, 5)
    plt.subplots_adjust(left=0.10, right=right_margin, top=0.95, bottom=0.10)
    # return 

# Extracts the angle, velocity and minimal distance to sun/stop criterium from the condition array
def convert_set_of_objects_into_arrays(condition_array, attribute_name):
    number_of_angles, number_of_velocity_steps, number_of_daytimes = condition_array.shape
    
    angle_array = np.zeros((number_of_angles))
    velocity_array = np.zeros((number_of_velocity_steps))
    daytimes_array = np.zeros((number_of_daytimes))
    min_distance_to_sun_array = np.zeros((number_of_angles, number_of_velocity_steps, number_of_daytimes))
    
    for idx_angle in range(number_of_angles):
        for idx_velocity in range(number_of_velocity_steps):
            for idx_daytime in range(number_of_daytimes):
                current_conditions = condition_array[idx_angle, idx_velocity, idx_daytime]
                angle_array[idx_angle] = current_conditions.angle
                velocity_array[idx_velocity] = current_conditions.velocity
                daytimes_array[idx_daytime] = current_conditions.daytime
                min_distance_to_sun_array[idx_angle, idx_velocity, idx_daytime] = getattr(current_conditions,attribute_name)
    
    return angle_array, velocity_array, daytimes_array, min_distance_to_sun_array
