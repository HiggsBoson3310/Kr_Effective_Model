# The idea of this script is to generate different folders with different dipole values to observe the change of the spectrogram. 
import numpy as np
import FWM_spec as fs
import os 

np.random.seed(24)

for i in range(3):
    os.makedirs('./example_%i'%i,exist_ok=True)
    dipoles = np.random.random((3,3))
    dipoles[0,:2] = 2*(dipoles[0,:2]*2-1); dipoles[0,2] = 1e-4*(dipoles[0,2]*2-1)
    dipoles[1,:2] = 2*(dipoles[1,:2]*2-1); dipoles[1,2] = 1e-4*(dipoles[1,2]*2-1)
    dipoles[2,:] = 1e-5*(dipoles[2,:]*2-1)
    np.savetxt('./example_%i/dipole_example.dat'%i, dipoles)
    
    file = open('./example_%i/job.sub'%i,'w')
    file.write(f'''#!/bin/bash

#SBATCH --job-name=Kr_test_{i}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 20:00:00

# Load the modules
module load anaconda
python ../testing.py
''')
    
