# The idea of this script is to generate different folders with different dipole values to observe the change of the spectrogram. 
import numpy as np
import FWM_spec as fs
import os 
for i in range(50):
    os.makedirs('./example_%i'%i)
    dipoles = np.random.random((3,3))
    dipoles[0,:2] = 2*(dipoles[0,:2]*2-1); dipoles[0,2] = 1e-4*(dipoles[0,2]*2-1)
    dipoles[1,:2] = 2*(dipoles[1,:2]*2-1); dipoles[1,2] = 1e-4*(dipoles[1,2]*2-1)
    dipoles[2,:] = 1e-5*(dipoles[2,:]*2-1)
    np.savetxt('dipole_example.dat', dipoles)
    print(" ")