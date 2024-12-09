import MQDT_core as mqdt
import numpy as np
import math_util as MU
import matplotlib.pyplot as plt
import scipy.interpolate as interpol
import scipy.signal as sg
import numpy.polynomial.legendre as le
from matplotlib.widgets import Slider

from concurrent.futures import ThreadPoolExecutor


# Identified
state_loc_1 = [26.84082546, 26.7851551088777, 25.1694016, 24.96927929, 26.286652]

state_lab_1 = ['CM15', '$4s4p^6 7p$', 'CM4', '$4s4p^6 5p$', '$4s4p^6 6p$']

state_loc_2 = [25.83,26.59,26.93,\
            25.73,26.52,26.89]
state_lab_2 = ['$4s4p^6 6s$','$4s4p^6 7s$','$4s4p^6 8s$',\
            '$4s4p^6 4d$','$4s4p^6 5d$', '$4s4p^6 6d$']

# Constants
fsperau=2.4188843e-17/1.0e-15; auI = 3.50944e16; evperAU=27.2114079527e0
fsm1toeV = fsperau*evperAU

# Thresholds
I1 = 13.9996055 + (13.514322) #(*4s.4p6*)
I2 = 13.9996055 + (13.988923*6 + 14.2695909*4 + 14.5809157*2)/12 #(*4s2.4p4.5s.(4P)*)
I3 = 13.9996055 + (0.665808*2)/6 #(*4s2.4p5*)




# Laser parameters
gam = 30/fsperau * 1/np.sqrt(4*np.log(2.0))
guv = 15/fsperau * 1/np.sqrt(4*np.log(2.0))
w = 0.93/evperAU #0.88/ evperAU
wuv = 26.9 / evperAU
per = 0.05 / evperAU
Fo = np.sqrt(1e12/auI)
limits = 2.5
# Spectrogram axis
e_axis = np.linspace(24.5,25.31,80)
delays = np.linspace(0/fsperau,8*np.pi/per,50)
spec = np.zeros((len(e_axis),len(delays)))

# Energy limits
e1 = 23.4
e2 = (guv**-2 *(e_axis[-1]/27.211+2*w) + gam**-2 *(wuv) )/(guv**-2+gam**-2) + limits*np.sqrt(np.log(8)/guv**2) #I1-evperAU*0.5/12**2
e2 = 27.211*e2

if(e2>I1):
    ValueError("The code has not been modified to deal with more than one open channel")
    
u2 = 26.84082546

print("Limits of the A coeff calculation")
print(e1,e2)

# Channels
Is_S1 = np.array([I1, I2, I3])/evperAU
ls_S1 = np.array([1.,1.,2.])
state_loc_1_nu = [mqdt.nu(ss/evperAU, Is_S1[0]) for ss in state_loc_1]
print(state_loc_1_nu)

Is_S2 = np.array([I1,I1,I3])/evperAU
ls_S2 = np.array([0.,2.,1.])
state_loc_2_nu = [mqdt.nu(ss/evperAU, Is_S2[0]) for ss in state_loc_2]
print(state_loc_2_nu)

erange = I1-evperAU * 0.5/np.linspace(mqdt.nu(e1/evperAU,I1/evperAU),mqdt.nu(e2/evperAU,I1/evperAU),5000)**2


p_S1 = lambda x: np.array([[-0.281,0.00798,-0.13],[0.00798,0.7384,0.196],[-0.13,0.196,0.122]]) +\
                (x-u2)*np.array([[0,0,0],[0,-0.043,0],[0,0,0]])
                
m_S1 = lambda x: np.array([[-0.150, 0.243, 0.3746],[0.243, 0.2246, 0.4608],[0.3746, 0.4608, 0.164]])+\
                (x-u2)*np.array([[0.02799,-0.05474,0],[-0.05474,-0.130368,0],[0,0,0]])
        
p_S2 = lambda x: np.array([[-2.942,0.61,-0.17],[0.61,-2.834,-1.43],[-0.17,-1.43,-0.75]]) #+ (x-)
                
phases_S1 = np.zeros((len(erange)),dtype=float)
c_coef_S1 = np.zeros((len(erange),3),dtype=float)
T_norm_S1 = np.zeros((len(erange)),dtype=float)
U1 = np.zeros((len(erange),3,3),dtype=float)
c_phase_S1 = np.zeros((len(erange)))

phases_S2 = np.zeros((len(erange)),dtype=float)
c_coef_S2 = np.zeros((len(erange),3),dtype=float)
T_norm_S2 = np.zeros((len(erange)),dtype=float)
U2 = np.zeros((len(erange),3,3),dtype=float)
c_phase_S2 = np.zeros((len(erange)))

mus1 = np.zeros((len(erange),3),dtype=float)
mus2 = np.zeros((len(erange),3),dtype=float)

cmuA1 = np.zeros((len(erange),3),dtype=float)
cmuA2 = np.zeros((len(erange),3),dtype=float)

print(mqdt.nu(erange[0]/evperAU,Is_S1[0]),mqdt.nu(erange[-1]/evperAU,Is_S1[0]))



s1 = 1
su1 = [1,1,1]
n1 = 0

s2 = 1
su2 = [1,1,1]
n2 = 0


for i in range(len(erange)):
    
    KK1 = mqdt.Km(p_S1(erange[i]))
    eig, U1[i,:,:] = np.linalg.eigh(KK1)
    mus1[i,:] = np.arctan(eig)/np.pi
    
    res = mqdt.GEV_P(erange[i]/evperAU, KK1, Is_S1, ls_S1)
    phases_S1[i] = res[0][0]+n1
    c_coef_S1[i,:] = res[1][0]*s1
    for k in range(len(Is_S1)):
        U1[i,:,k] = U1[i,:,k]*su1[k]
    
    
    
    KK2 = mqdt.Km(p_S2(erange[i]))
    eig, U2[i,:,:] = np.linalg.eigh(KK2)
    mus2[i,:] = np.arctan(eig)/np.pi
    
    res = mqdt.GEV_P(erange[i]/evperAU, KK2, Is_S2, ls_S2)
    phases_S2[i] = res[0][0]+n2
    c_coef_S2[i,:] = res[1][0]*s2
    for k in range(len(Is_S2)):
        U2[i,:,k] = U2[i,:,k]*su2[k]
    
    if(i!=0):
        if(phases_S1[i-1]>phases_S1[i]):
            n1+=1
            phases_S1[i] += 1
        
        if(np.any(np.abs(c_coef_S1[i,:]-c_coef_S1[i-1,:])>1e-1)):
            s1 *= -1
            c_coef_S1[i,:] = -1*c_coef_S1[i,:]
        
        for k in range(len(Is_S1)):
            if(np.any(np.abs(U1[i,:,k]-U1[i-1,:,k])>1e-1)):
                su1[k] *= -1
                U1[i,:,k] = -1*U1[i,:,k] 
            
            
            
        if(phases_S2[i-1]>phases_S2[i]):
            n2+=1
            phases_S2[i]+=1
        
        if(np.any(np.abs(c_coef_S2[i,:]-c_coef_S2[i-1,:])>2e-1)):
            s2 *= -1
            c_coef_S2[i,:] = -1*c_coef_S2[i,:]
            
        for k in range(len(Is_S2)):
            if(np.any(np.abs(U2[i,:,k]-U2[i-1,:,k])>1e-1)):
                su2[k] *= -1
                U2[i,:,k] = -1*U2[i,:,k] 
                
    
    d1 = np.linalg.det(np.transpose(U1[i,:,:])@U1[i,:,:])
    if(abs(d1-1)>1e-8):
        print(f'Determinant of U1 is not one, is {d1}')
        
    d2 = np.linalg.det(np.transpose(U2[i,:,:])@U2[i,:,:])
    if(abs(d2-1)>1e-8):
        print(f'Determinant of U1 is not one, is {d2}')
            
    cmuA1[i,:] = np.transpose(U1[i,:,:])@c_coef_S1[i,:]/np.cos(np.pi*mus1[i,:])
    cmuA2[i,:] = np.transpose(U2[i,:,:])@c_coef_S2[i,:]/np.cos(np.pi*mus2[i,:])
    
    T_norm_S1[i] = np.cos(np.pi*phases_S1[i])*c_coef_S1[i,2] + np.sin(np.pi*phases_S1[i])*np.dot(KK1[2,:],c_coef_S1[i,:])
    
    T_norm_S2[i] = np.cos(np.pi*phases_S2[i])*c_coef_S2[i,2] + np.sin(np.pi*phases_S2[i])*np.dot(KK2[2,:],c_coef_S2[i,:])
    

c_phase_S1 = np.exp(-1j*(np.pi*phases_S1-np.pi/2*ls_S1[2]+MU.sigma(erange/evperAU,ls_S1[2],Is_S1[2])))

c_phase_S2 = np.exp(-1j*(np.pi*phases_S2-np.pi/2*ls_S2[2]+MU.sigma(erange/evperAU,ls_S2[2],Is_S2[2])))
    
    


def sanity_plots():

    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),np.real(c_coef_S1[:,0]),'-',c='C0')
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),np.real(c_coef_S1[:,1]),'-',c='C1')
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),np.real(c_coef_S1[:,2]),'-',c='C2')

    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),np.real(c_coef_S2[:,0]),':',c='C0')
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),np.real(c_coef_S2[:,1]),':',c='C1')
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),np.real(c_coef_S2[:,2]),':',c='C2')

    plt.savefig('./Figures/c_detail.png')



    

    fig, ax = plt.subplots(figsize=(16,5))
    l1, = plt.plot(erange, np.real(phases_S1),'-o',label='$\\mathcal{S}_1$')
    l2, = plt.plot(erange, np.real(phases_S2),'-o',label='$\\mathcal{S}_2$')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    ax2.set_xticks(ticks=state_loc_1+state_loc_2,labels=state_lab_1+state_lab_2,rotation=45,ha='left',va='bottom')
    for e in zip(state_loc_1):
        ax2.axvline(e,color='red')
        
    for e in zip(state_loc_2):
        ax2.axvline(e,color='blue')
    ax.legend(loc=0)
    ax.set_ylabel("Time Delay (a.u.)")
    ax.set_xlabel('Energy from Kr g.s.')
    plt.savefig('./Figures/level_pos_phase.png',dpi=120)

    fig, ax = plt.subplots(figsize=(16,5))
    l1, = plt.plot(erange, np.gradient(np.real(phases_S1),(erange[1]-erange[0])/evperAU),'-o',label='$\\mathcal{S}_1$')
    l2, = plt.plot(erange, np.gradient(np.real(phases_S2),(erange[1]-erange[0])/evperAU),'-o',label='$\\mathcal{S}_2$')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax.set_yscale('log')

    ax2.set_xticks(ticks=state_loc_1+state_loc_2,labels=state_lab_1+state_lab_2,rotation=45,ha='left',va='bottom')
    for e in zip(state_loc_1):
        ax2.axvline(e,color='red')
        
    for e in zip(state_loc_2):
        ax2.axvline(e,color='blue')
    ax.legend(loc=0)
    ax.set_ylabel("Time Delay (a.u.)")
    ax.set_xlabel('Energy from Kr g.s.')
    plt.savefig('./Figures/level_pos_tdel.png',dpi=120)

    fig, ax = plt.subplots(figsize=(8,5))
    l1, = plt.plot(mqdt.nu(erange/evperAU,Is_S1[0]), np.gradient(np.real(phases_S1),(erange[1]-erange[0])/evperAU),'-',label='$\\mathcal{S}_1$')
    l2, = plt.plot(mqdt.nu(erange/evperAU,Is_S2[0]), np.gradient(np.real(phases_S2),(erange[1]-erange[0])/evperAU),'-',label='$\\mathcal{S}_2$')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax.set_yscale('log')

    ax2.set_xticks(ticks=state_loc_1_nu+state_loc_2_nu,labels=state_lab_1+state_lab_2,rotation=45,ha='left',va='bottom')
    for e in zip(state_loc_1_nu):
        ax2.axvline(e,color='red')
        
    for e in zip(state_loc_2_nu):
        ax2.axvline(e,color='blue')
    ax.legend(loc=0)
    ax.set_ylabel("Time Delay (a.u.)")
    ax.set_xlabel('Effective quantum number $\\nu_{4s 4p^6}$')
    ax.set_xlim(2,5)
    ax2.set_xlim(2,5)
    plt.savefig('./Figures/level_pos_tdel_nu.png',dpi=120)

    ax.set_yscale('linear')
    l1.set_ydata(np.real(1/T_norm_S1))
    l2.set_ydata(np.real(1/T_norm_S2))
    ax.relim()
    ax.autoscale_view()
    plt.savefig('./Figures/T_norm.png',dpi=120)


    l1.set_ydata(np.real(c_coef_S1[:,0]))
    ax.plot(erange,np.real(c_coef_S1[:,1]),'--',c='C0')
    ax.plot(erange,np.real(c_coef_S1[:,2]),':',c='C0')
        
    l2.set_ydata(np.real(c_coef_S2[:,0]))
    ax.plot(erange,np.real(c_coef_S2[:,1]),'--',c='C1')
    ax.plot(erange,np.real(c_coef_S2[:,2]),':',c='C1')

    ax.relim()
    ax.autoscale_view()
    #ax.set_xlim(26.5,27.0)
    plt.savefig('./Figures/c_coeffs.png',dpi=120)

    plt.close()

    fig, ax = plt.subplots()
    plt.plot(erange,cmuA1[:,0])
    plt.plot(erange,cmuA1[:,1])
    plt.plot(erange,cmuA1[:,2])

    plt.plot(erange,cmuA2[:,0],':',c='C0')
    plt.plot(erange,cmuA2[:,1],':',c='C1')
    plt.plot(erange,cmuA2[:,2],':',c='C2')

    plt.savefig('./Figures/mus.png',dpi=120)
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),cmuA1[:,0])
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),cmuA1[:,1]/T_norm_S1)
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),cmuA1[:,2]/T_norm_S1)

    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),cmuA2[:,0]/T_norm_S2,':',c='C0')
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),cmuA2[:,1]/T_norm_S2,':',c='C1')
    plt.plot(mqdt.nu(erange/evperAU,I1/evperAU),cmuA2[:,2]/T_norm_S2,':',c='C2')
    for e in state_loc_1:
        plt.axvline(mqdt.nu(e/evperAU,I1/evperAU),color='red')
        
    for e in state_loc_2:
        plt.axvline(mqdt.nu(e/evperAU,I1/evperAU),color='blue')
    
    
    plt.xlim(mqdt.nu(24.8/evperAU,I1/evperAU),mqdt.nu(27/evperAU,I1/evperAU))
    plt.savefig('./Figures/mus_nu.png',dpi=120)
    

    return None

cfunc = lambda x: MU.norm_gauss((x-wuv),2/guv)

#print("Doing the plots")

A1_funcs = [] 
A2_funcs = []

fig, ax = plt.subplots(1,1)
ax2 = ax.twinx()


# Find the locations of the resonances in our data 

peaks_s1, props1 = sg.find_peaks(np.gradient(np.real(phases_S1),(erange[1]-erange[0])/evperAU))
peaks_s2, props2 = sg.find_peaks(np.gradient(np.real(phases_S2),(erange[1]-erange[0])/evperAU))

peaks_s1 = mqdt.nu(erange[peaks_s1]/evperAU, I1/evperAU)
peaks_s2 = mqdt.nu(erange[peaks_s2]/evperAU, I1/evperAU)

np.savetxt('symmetry_one_peaks.txt',peaks_s1)
np.savetxt('symmetry_two_peaks.txt',peaks_s2)

sanity_plots()

STOP
                                 
for i in range(3):
    A1_funcs.append(interpol.interp1d(erange/evperAU,c_coef_S1[:,i]/(T_norm_S1) * c_phase_S1,bounds_error=False,fill_value=0.0))
    A2_funcs.append(interpol.interp1d(erange/evperAU,c_coef_S2[:,i]/(T_norm_S2) * c_phase_S2,bounds_error=False,fill_value=0.0))
    
    ax.plot(mqdt.nu(erange/evperAU, Is_S1[0]),c_coef_S1[:,i]/T_norm_S1,color='C%i'%i)
    ax.plot(mqdt.nu(erange/evperAU, Is_S2[0]),c_coef_S2[:,i]/T_norm_S2,":",color='C%i'%i)

ax2.plot(mqdt.nu(erange/evperAU, Is_S1[0]),MU.gauss((erange/evperAU-wuv)/(2/guv)),'-',c='green')
ax2.plot(mqdt.nu(erange/evperAU, Is_S1[0]),MU.gauss((erange/evperAU-(wuv-w))/(2/gam)),'-',c='blue')

for e in peaks_s1[[3,4]]:
    E = I1/evperAU - 0.5/e**2
    ax2.plot(mqdt.nu(erange/evperAU, Is_S1[0]),MU.gauss((erange/evperAU-(E-w))/(2/gam)),':',c='cornflowerblue')
    ax2.plot(mqdt.nu(erange/evperAU, Is_S1[0]),MU.gauss((erange/evperAU-(E-2*w))/(4/gam)),':',c='cornflowerblue')


ax.plot(mqdt.nu(erange/evperAU, Is_S1[0]),np.cos(np.pi*phases_S1),color='k')
ax.plot(mqdt.nu(erange/evperAU, Is_S2[0]),np.cos(np.pi*phases_S2),":",color='k')
ax.set_xlim(2,15)
ax2.set_xlim(2,15)

for e in peaks_s1:
    ax2.axvline(e,ls=':',color='red')

for e in state_loc_1_nu:
    ax2.axvline(e,color='red')

for e in peaks_s2:
    ax2.axvline(e,ls=':', color='blue')
        
for e in state_loc_2_nu:
    ax2.axvline(e,color='blue')
  
  
ax.set_xlim(2,5)
ax2.set_xlim(2,5)  
#plt.show()
plt.savefig('./Figures/plots_with_lasers_and_states.png',dpi=120)
#STOP
plt.close()

def adjuster():
    e_axis = I1/evperAU - 0.5/np.linspace(2,5,160)**2
    def dip_map(Deigen):
        dipoles_r = np.zeros((160,160))
        dipoles_i = np.zeros((160,160))
        i = 0
        for e in e_axis:
            A1_vec = np.array([A(e) for A in A1_funcs])
            j = 0
            for ee in e_axis:
                A2_vec = np.array([A(ee) for A in A2_funcs])
                dip = np.dot(np.conjugate(A1_vec),Deigen@A2_vec)
                dipoles_r[i,j] = np.real(dip)
                dipoles_i[i,j] = np.imag(dip)
                j+=1
            i+=1
        return dipoles_r, dipoles_i

    d_r, d_i = dip_map(Deigen)

    fig, ax = plt.subplots(1,1)
    X,Y = np.meshgrid(mqdt.nu(e_axis,I1/evperAU),mqdt.nu(e_axis,I1/evperAU))
    im = ax.pcolormesh(X,Y,d_r**2 + d_i**2)
    plt.colorbar(im)
    ax.set_ylabel('$\\nu$ sym 1')
    ax.set_xlabel('$\\nu$ sym 2')

    for e in zip(state_loc_1_nu):
        plt.axhline(e,color='red')
        
    for e in zip(state_loc_2_nu):
        plt.axvline(e,color='blue')
    # Define sliders for each matrix element
    slider_ax_positions = [
        [0.25, 0.25, 0.65, 0.03],
        [0.25, 0.20, 0.65, 0.03],
        [0.25, 0.15, 0.65, 0.03],
        [0.25, 0.10, 0.65, 0.03],
        [0.25, 0.05, 0.65, 0.03],
        [0.25, 0.00, 0.65, 0.03],
    ]
    sliders = []

    # Initialize the sliders for each matrix entry
    for i, pos in enumerate(slider_ax_positions):
        row, col = divmod(i, 3)
        slider_ax = plt.axes(pos)
        slider = Slider(slider_ax, f'a{row+1}{col+1}', 0, 10, valinit=1)
        sliders.append(slider)
        
    # Update function to refresh the heatmap when a slider is changed
    def update(val):
        row = 0
        col = 0
        for i, slider in enumerate(sliders):
            print(row,col)
            Deigen[row, col] = slider.val  
            Deigen[col, row] = slider.val
            if((i+1+row)%3==0):
                row += 1 
                col = row
            else:
                col +=1
            
        d_r, d_i = dip_map(Deigen)
        im.set_array(d_r**2 + d_i**2)  # Update heatmap data
        fig.canvas.draw_idle()  # Redraw the figure

    # Attach the update function to each slider
    for slider in sliders:
        slider.on_changed(update)

    # Display the interactive plot
    plt.show()


Deigen = np.array([[1.35,0.00,0.058],[0.00,-2.34,0.0487],[0.018,0.017,0.0]])

for i in range(len(e_axis)):
    Ei = e_axis[i]/evperAU
    d_center = (guv**-2 *(Ei+2*w) + gam**-2 *(wuv) )/(guv**-2+gam**-2)
    delta_mesh = np.linspace(d_center-limits*np.sqrt(np.log(8)/gam**2),d_center+limits*np.sqrt(np.log(8)/guv**2),7)
    for j in range(len(delays)):
        print(i,j)
        to = delays[j]
        
        if(to>100/fsperau):        
            spec[i,j] = np.abs(MU.cfin_sum_in_eta_int(Ei,1,0,0,0,Deigen,A1_funcs,A2_funcs,Fo,to,cfunc,delta_mesh,gam,w,20,plot=False,limits=2.5))**2 * (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2 )
        else:
            if(to>0):
                spec[i,j] = np.abs(MU.cfin_sum_in(Ei,1,0,0,0,Deigen,A1_funcs,A2_funcs,Fo,to,cfunc,delta_mesh,gam,w,20))**2  * (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2 )
            else:
                spec[i,j] = np.abs(MU.cfin_sum_in(Ei,1,0,0,0,Deigen,A1_funcs,A2_funcs,Fo,to,cfunc,delta_mesh,gam,w,20))**2 * (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2 )
    
    
print(np.any(np.isnan(spec)))
fig, axx = plt.subplots(1,2)
im = axx[0].imshow(spec, origin='lower', extent=[delays[0]*fsperau, delays[-1]*fsperau, e_axis[0], e_axis[-1]],aspect='auto',cmap='turbo')
fig.colorbar(im,ax=axx[0])
im = axx[1].imshow(np.sqrt(spec), origin='lower', extent=[delays[0]*fsperau, delays[-1]*fsperau, e_axis[0], e_axis[-1]],aspect='auto',cmap='turbo')
fig.colorbar(im,ax=axx[1])

axx[0].set_title('Closed channel function norm squared')
axx[1].set_title('Square root of the norm squared (to ajust the color scale)')

for ax in axx:
    ax.axhline(state_loc_1[2],color='blue')
    ax.axhline(state_loc_1[3],color='blue')
    #plt.axhline(state_loc_2[0]-w*evperAU,color='red')
    #plt.axhline(state_loc_2[3]-w*evperAU,color='red')
    ax.axhline(state_loc_1[0]-2*w*evperAU,color='green')
    ax.axhline(state_loc_1[1]-2*w*evperAU,color='green')
    ax.axhline((wuv-2*w)*evperAU)

for i in range(8):
    plt.axvline(fsperau * i*np.pi/per,color='white')
plt.show()