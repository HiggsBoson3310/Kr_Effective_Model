import FWM_spec as fs
import numpy as np
import MQDT_core as mqdt
import scipy.interpolate as inter
import matplotlib.pyplot as plt
import math_util as MU
import time as time
import os
# Thresholds
I1 = 13.9996055 + (13.514322) #(*4s.4p6*)
I2 = 13.9996055 + (13.988923*6 + 14.2695909*4 + 14.5809157*2)/12 #(*4s2.4p4.5s.(4P)*)
I3 = 13.9996055 + (0.665808*2)/6 #(*4s2.4p5*)


# Locations of the states according to CM and other references.
 
state_loc_1 = [26.84082546, 26.7851551088777, 25.1694016, 24.96927929, 26.286652]

state_lab_1 = ['CM15', '$4s4p^6 7p$', 'CM4', '$4s4p^6 5p$', '$4s4p^6 6p$']

state_loc_2 = [25.83,26.59,26.93,\
            25.73,26.52,26.89]
state_lab_2 = ['$4s4p^6 6s$','$4s4p^6 7s$','$4s4p^6 8s$',\
            '$4s4p^6 4d$','$4s4p^6 5d$', '$4s4p^6 6d$']


# position of CM15 state
u2 = 26.84082546

p_S1 = lambda x: np.array([[-0.150, 0.243, 0.3746],[0.243, 0.2246, 0.4608],[0.3746, 0.4608, 0.164]])+\
                (x-u2)*np.array([[0.02799,-0.05474,0],[-0.05474,-0.130368,0],[0,0,0]])

m_S1 = lambda x: np.array([[-0.281,0.00798,-0.13],[0.00798,0.7384,0.196],[-0.13,0.196,0.122]]) +\
                (x-u2)*np.array([[0,0,0],[0,-0.043,0],[0,0,0]])

p_S2 = lambda x: np.array([[-2.942,0.61,-0.17],[0.61,-2.834,-1.43],[-0.17,-1.43,-0.75]]) #+ (x-)


Is_S1 = np.array([I1, I2, I3])
ls_S1 = np.array([1.,1.,2.])

Is_S2 = np.array([I1,I1,I3])
ls_S2 = np.array([0.,2.,1.])

e_axis = np.linspace(24.5,25.31,180)
delays = np.linspace(0/fs.fsperau,8*np.pi/0.05 * fs.evperAU,90)


param_dict = {
        'gam': 30/fs.fsperau * 1/np.sqrt(4*np.log(2.0)),
        'guv': 15/fs.fsperau * 1/np.sqrt(4*np.log(2.0)),
        'w' : 0.88/ fs.evperAU,
        'wuv' : 26.9 / fs.evperAU,
        'per' : 0.05 / fs.evperAU,
        'Fo' : np.sqrt(1e12/fs.auI),
        'limits' : 2.5,
        'state_loc_1': state_loc_1
        }

e1 = 23.4
e2 = (param_dict['guv']**-2 *(e_axis[-1]/27.211+2*param_dict['w'])+ 
      param_dict['gam']**-2 *(param_dict['wuv']) )/(param_dict['guv']**-2+param_dict['gam']**-2) + \
      param_dict['limits']*np.sqrt(np.log(8)/param_dict['guv']**2) #I1-evperAU*0.5/12**2

e2 = 27.211*e2

erange = I1-fs.evperAU * 0.5/np.linspace(mqdt.nu(e1/fs.evperAU,I1/fs.evperAU),mqdt.nu(e2/fs.evperAU,I1/fs.evperAU),1000)**2

cs_S1, Ts_S1, phs_S1, c_phase_S1 = fs.compute_c_coeff(erange, Is_S1, ls_S1,p_S1)
cs_S2, Ts_S2, phs_S2, c_phase_S2 = fs.compute_c_coeff(erange, Is_S2, ls_S2,p_S2)

coefs1 = []
coefs2 = []

for i in range(3):
    coefs1.append(inter.interp1d(erange/fs.evperAU,cs_S1[:,i]/Ts_S1*\
        np.exp(-1j*(np.pi*phs_S1-ls_S1[2]*np.pi/2+c_phase_S1)),kind='cubic',
                                 bounds_error=False, fill_value=0.0))
    coefs2.append(inter.interp1d(erange/fs.evperAU,cs_S2[:,i]/Ts_S2*\
        np.exp(-1j*(np.pi*phs_S2-ls_S2[2]*np.pi/2+c_phase_S2)),kind='cubic',
                                 bounds_error=False, fill_value=0.0))


#Deigen = np.array([[1.35,0.00,0.058],[0.00,-2.34,0.0487],[0.018,0.017,0.0]])

def main():
    # Get the current working directory
    current_working_directory = os.getcwd()

    # Construct the file path
    file_path = os.path.join(current_working_directory, 'dipole_example.dat')

    Deigen = np.loadtxt(file_path)
    #spec = fs.compute_spectrogram(coefs1, coefs2, Deigen, e_axis, delays, param_dict)
    spec = fs.compute_spec_parallel(coefs1, coefs2, Deigen, e_axis, delays, param_dict)
    
    #np.save('spec_test.npy',spec)

if __name__ == '__main__':
    st = time.time()
    main()
    en = time.time()
    print(f"This run took {en-st} seconds")