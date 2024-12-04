import os
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fsperau=2.4188843e-17/1.0e-15; auI = 3.50944e16; evperAU=27.2114079527e0
fsm1toeV = fsperau*evperAU

def slider():
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

def single_plot(number):
    data_file = './example_%i/spec_data.npy'%(number)
    dipole = './example_%i/dipole_example.dat'%(number)
    x_file = './example_%i/x_axis.npy'%(number)
    y_file = './example_%i/y_axis.npy'%(number)
    fig, ax = plt.subplots(2,1)
    
    if os.path.exists(data_file):
        x = np.load(x_file)
        y = np.load(y_file)
        data = np.load(data_file)
        X, Y = np.meshgrid(x*fsperau,y)
        ax[0].pcolormesh(X, Y, data,cmap='turbo')
        ax[0].set_title(f"Norm squared of closed channel function.")
        ax[0].set_xlabel('Time delay (fs)')
        ax[0].set_ylabel('Energy (eV)')
        ax[0].set_ylim(24.8,25.2)
        
        freqs = fft.fftfreq(len(x),d=(x[1]-x[0]))[:len(x)//2] * 2 * np.pi * evperAU
    
        spec_fft = fft.fft(data,axis=1)[:,:len(x)//2]
        
        
        X,Y = np.meshgrid(freqs, y)
        
        ax[1].pcolormesh(X,Y, np.sqrt(np.abs(spec_fft)), cmap='turbo')
        ax[1].set_ylim(24.8,25.2)
        plt.show()
        plt.close()
        
        fig, ax  = plt.subplots(2,1)
        avg = np.mean(data, axis=-1)
        ax[0].plot(y, avg)
        X,Y = np.meshgrid(x*fsperau,y)
        newdat = np.zeros_like(data)
        for i in range(len(x)):
            newdat[:,i] = data[:,i]-avg
        ax[1].pcolormesh(X,Y,newdat,cmap='turbo',vmin=0)
        ax[1].set_ylim(24.8,25.2)
        plt.show()
        
        
    