import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fsperau=2.4188843e-17/1.0e-15
# Base directory path
BASE_DIR = "./"

# Function to read and plot data from a directory
def read_and_plot(ax, directory):
    data_file = os.path.join(directory, "spec_data.npy")
    dipole = os.path.join(directory,'dipole_example.dat')
    x_file = os.path.join(directory,'x_axis.npy')
    y_file = os.path.join(directory,'y_axis.npy')
    ax[0].clear()
    ax[1].clear()
    if os.path.exists(data_file):
        x = np.load(x_file)
        y = np.load(y_file)
        data = np.load(data_file)
        X, Y = np.meshgrid(x*fsperau,y)
        ax[1].pcolormesh(X, Y, data, label=f"Data from {directory}",cmap='turbo')
        ax[1].set_title(f"Plot from {directory}")
        ax[1].legend()
        
        dipoles = np.loadtxt(dipole)
        ax[0].matshow(dipoles)
        
    else:
        ax[0].text(0.5, 0.5, "File not found", ha='center', va='center', fontsize=12)
        ax[1].text(0.5, 0.5, "File not found", ha='center', va='center', fontsize=12)
        ax[1].set_title(f"No data in {directory}")
    
    ax[0].figure.canvas.draw()
    ax[1].figure.canvas.draw()

# Main function to create the interactive plot
def main():
    # Initial directory index
    initial_index = 0
    initial_dir = os.path.join(BASE_DIR, f"example_{initial_index}")

    # Create the plot
    fig, ax = plt.subplots(1,2)
    plt.subplots_adjust(bottom=0.2)

    # Plot initial data
    read_and_plot(ax, initial_dir)

    # Add a slider for changing the directory
    slider_ax = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
    slider = Slider(slider_ax, "Dir Index", 0, 49, valinit=initial_index, valstep=1)

    # Update the plot when the slider changes
    def update(val):
        dir_index = int(slider.val)
        current_dir = os.path.join(BASE_DIR, f"example_{dir_index}")
        read_and_plot(ax, current_dir)

    slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()