import MQDT_core as mqdt
import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
import Legacy_Code.math_util_legacy as MU
import json
from multiprocessing import Pool


# Constants
fsperau=2.4188843e-17/1.0e-15; auI = 3.50944e16; evperAU=27.2114079527e0
fsm1toeV = fsperau*evperAU

## Reference Threshold
I1 = 13.9996055 + (13.514322)

def compute_c_coeff(erange, Is, ls, p_params):
    """
    Compute c_coef, T_norm, and phases arrays over the given energy range.

    Parameters:
    erange : array_like
        Energy range (in eV).
    Is : array_like
        Ionization thresholds (in eV).
    ls : array_like
        Angular momentum quantum numbers.
    p_params : function_like
        Function that defines the mu parameters to compute the K matrix
    
    Returns:
    c_coef : ndarray
        Coefficient array.
    T_norm : ndarray
        Normalization factor array.
    phases : ndarray
        Phases array.
    """
    # Constants
    

    # Initialize arrays
    phases = np.zeros(len(erange), dtype=float)
    c_coef = np.zeros((len(erange), len(Is)), dtype=float)
    Z_coef = np.zeros((len(erange)),dtype=object)
    Z_coef_calc = np.zeros(len(erange),dtype=object)
    smat_l = np.zeros((len(erange)),dtype=object)
    T_norm = np.zeros(len(erange), dtype=float)
    U = np.zeros((len(erange), len(Is), len(Is)), dtype=float)

    # Variables for continuity corrections
    s = 1
    su = np.ones(len(Is))
    n = 0
    
    for i in range(len(erange)):
        E = erange[i]  # Energy in eV
        # Compute K-matrix
        Km = mqdt.Km(p_params(E))
        # Eigenvalues and eigenvectors
        eigvals, U[i, :, :] = np.linalg.eigh(Km)
        mus = np.arctan(eigvals) / np.pi

        # Solve generalized eigenvalue problem
        taus, cs, Z, rSmat = mqdt.GEV_P(E / evperAU, Km, Is / evperAU, ls)
        phases[i] = taus[0] + n
        c_coef[i, :] = cs[0] * s

        # Continuity corrections
        if i > 0:
            if phases[i - 1] > phases[i]:
                n += 1
                phases[i] += 1
            c_cond = np.abs(c_coef[i-1, :] - c_coef[i, :]) > np.abs(c_coef[i-1,:]+c_coef[i, :])
            if (sum(c_cond) >= len(c_cond)/2.):
                s *= -1
                c_coef[i, :] *= -1
            for k in range(len(Is)):
                u_cond = np.abs(U[i, :, k] - U[i - 1, :, k]) > np.abs(U[i,:,k]+U[i-1,:,k])
                if (sum(u_cond) >= len(u_cond)/2.0):
                    su[k] *= -1
                    U[i, :, k] *= -1

        # Compute T_norm
        T_norm[i] = np.cos(np.pi * phases[i]) * c_coef[i, -1] + np.sin(np.pi * phases[i]) * np.dot(Km[-1, :], c_coef[i, :])
        
        Z_coef[i] = Z
        
        bet = np.array(mqdt.beta(E/evperAU,Is[:2]/evperAU,ls[:2]))
        Z_coef_calc[i] = np.diag(1/np.sin(bet))@(Km@c_coef[i])[:2]
        
        smat_l[i] = rSmat
    # Coulomb phase
    
    c_phase = np.exp(-1j*(np.pi*phases+mqdt.eta(E/evperAU,Is[2]/evperAU,ls[2])))
    
    return c_coef, T_norm, phases, c_phase, Z_coef, smat_l, Z_coef_calc

def view_line(i, A1_funcs, A2_funcs, Deigen, e_axis, delta_mesh, params):
    # Extract parameters
    Fo = params['Fo']
    w = params['w']
    wuv = params['wuv']
    gam = params['gam']
    guv = params['guv']
    limits = params['limits']
    
    # Normalized gaussian for the XUV
    Zcoeffs_init = lambda x: 4*A1_funcs[0](x)+A1_funcs[1](x)
    cfunc = lambda x: MU.norm_gauss((x-wuv),2/guv) *Zcoeffs_init(x)
    
    Ei = e_axis[i] / evperAU
    
    if(i==0):
        fig, ax = plt.subplots()
        ax.plot(delta_mesh, np.real(cfunc(delta_mesh)))
        ax.plot(delta_mesh, np.imag(cfunc(delta_mesh)))
        plt.savefig('cfunc_plot.png', dpi=120)
        
    
    spec_l_r, spec_l_i = MU.Xi_int_for_viewing(
                Ei, 1, 0, 0, 0, Deigen, A1_funcs, A2_funcs, Fo, cfunc,
                delta_mesh, gam, w,I1/evperAU, interm_lim=[2,4.5], plot=False, limits=2.5
            )#* (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2)
    
    return (i, spec_l_r, spec_l_i)


def compute_view_parallel(A1_funcs, A2_funcs, Deigen, e_axis, params):
    state_loc_1 = params['state_loc_1']
    w = params['w']
    wuv = params['wuv']
    per = params['per']
    
    NN = 150
    
    delta_mesh = np.linspace(I1/evperAU-0.5/3.5**2, I1/evperAU-0.5/5**2,NN)
    spec_r = np.zeros((len(e_axis),NN))
    spec_i = np.zeros((len(e_axis),NN))
    #func = lambda x: spec_line(x,A1_funcs,A2_funcs,Deigen,e_axis,params)
    print('We are passing to the pool of worker the following dipole: ')
    print(Deigen)
    print('And laser frequencies: ')
    print("IR: ", w*evperAU, "and XUV: ", wuv*evperAU)
    print("Creating a parallel process with ", cpu_count())
    pool = Pool()
    arguments = [[i,A1_funcs, A2_funcs, Deigen, e_axis, delta_mesh, params] for i in range(len(e_axis))]
    res = pool.starmap(view_line,arguments)
    
    for i in range(len(res)):
        spec_r[res[i][0],:] = res[i][1]
        spec_i[res[i][0],:] = res[i][2]
        
    # Plot spectrogram
    fig, axx = plt.subplots(1, 2,figsize=(10,5))
    fig.suptitle('Photionization Probability')
    
    xx, yy = np.meshgrid(mqdt.nu(delta_mesh,I1/evperAU), mqdt.nu(e_axis/evperAU,I1/evperAU))
    
    im = axx[0].pcolormesh(xx, yy, spec_r, cmap='turbo')
    fig.colorbar(im, ax=axx[0])
    
    im = axx[1].pcolormesh(xx, yy, spec_i, cmap='turbo')
    fig.colorbar(im, ax=axx[1])
    
    axx[0].set_title('Real Part')
    axx[1].set_title('Imaginary Part')

    for ax in axx:
        ax.axhline(mqdt.nu(state_loc_1[2]/evperAU,I1/evperAU), color='blue')
        ax.axhline(mqdt.nu(state_loc_1[3]/evperAU,I1/evperAU), color='blue')
        ax.axvline(mqdt.nu(state_loc_1[0]/evperAU,I1/evperAU), color='green')
        ax.axvline(mqdt.nu(state_loc_1[1]/evperAU,I1/evperAU), color='green')
        
        ax.axhline(mqdt.nu((state_loc_1[0]-2*w*evperAU)/evperAU,I1/evperAU), color='green', linestyle='--')
        ax.axhline(mqdt.nu((state_loc_1[1]-2*w*evperAU)/evperAU,I1/evperAU), color='green', linestyle='--')
        
        ax.axhline(mqdt.nu((wuv - 2 * w) * evperAU/evperAU,I1/evperAU))
        ax.axvline(mqdt.nu(wuv * evperAU/evperAU,I1/evperAU), color='red')

    
    
    plt.savefig('Delta_integrand.png',dpi=210)
    plt.close()
    
    
    np.save('x_axis_integrand.npy',delta_mesh*evperAU)
    np.save('y_axis_integrand.npy',e_axis)
    np.save('integrand_data.npy',spec_r+1j*spec_i)

    # Now let us try to get the fourier as an atempt to calculate the integral
    
    freqs = fft.fftshift(fft.fftfreq(len(delta_mesh),d=(delta_mesh[1]-delta_mesh[0])) * 2 * np.pi )
    spec_fft = fft.fftshift(fft.fft(spec_r+1j*spec_i,axis=1),axes=1)
    
    fig, ax = plt.subplots(1,1)
    
    xx, yy = np.meshgrid(e_axis, freqs * fsperau)
    
    ax.pcolormesh(xx,yy, np.abs(spec_fft[::,::-1].T)**2,cmap='turbo')
    ax.set_ylim(-50,400)
    
    for i in range(8):
        ax.axhline(fsperau * i * np.pi / per, color='white')
            
    plt.savefig('Spectrogram_from_fft.png',dpi=120)
    plt.close()
    
    np.save('time_delay_fft.npy',-1*freqs[::-1]*fsperau)
    np.save('energy_axis_fft.npy',e_axis)
    np.save('spec_from_fft.npy',spec_fft[::,::-1])
    
    # And now we do the double fourier freq, but notice that we only do it on the positive times
    freqs = freqs[::-1]
    spec_fft = spec_fft[::,::-1]
    
    nfeq = np.argmin(np.abs(freqs))
    
    feqs2 = fft.fftshift(fft.fftfreq(len(freqs[nfeq:]), d= (freqs[nfeq+1]-freqs[nfeq]))) * 2 * np.pi * evperAU
    
    spec_fft_fft =  fft.fftshift(fft.fft(np.abs(spec_fft[::,nfeq:])**2,axis=1),axes=1)
    
    fig, ax = plt.subplots(1,2)
    
    xx, yy = np.meshgrid(feqs2, e_axis)
    
    ax[0].pcolormesh(xx,yy,np.abs(spec_fft_fft)**2,cmap='turbo')
    ax[1].pcolormesh(xx,yy,np.angle(spec_fft_fft),cmap='hsv')
    
    ax[0].set_xlim(0,0.3)
    ax[1].set_xlim(0,0.3)
    
    plt.savefig('Spectrogram_from_fft_fft.png',dpi=120)
    
    np.save('freqs_fft.npy',feqs2)
    np.save('energy_axis_fft.npy',e_axis)
    np.save('spec_from_fft_fft.npy',spec_fft_fft)
    
    # Save to file
    with open("params.json", "w") as f:
        json.dump(params, f)
    
    plt.close()
    
    
    
    
    return None

def simple_test(A1_funcs, A2_funcs, Deigen, e_axis, params):
    # Extract parameters
    Fo = params['Fo']
    w = params['w']
    wuv = params['wuv']
    gam = params['gam']
    guv = params['guv']
    limits = params['limits']
    

    # Normalized gaussian for the XUV
    Zcoeffs_init = lambda x: A1_funcs[0](x)+A1_funcs[1](x)
    cfunc = lambda x: MU.norm_gauss((x-wuv),2/guv)*Zcoeffs_init(x)
    
    
    Ei = 24.99/ evperAU
    d_center = (guv**-2 * (Ei + 2 * w) + gam**-2 * wuv) / (guv**-2 + gam**-2)
    delta_mesh = np.linspace(
        d_center - limits * np.sqrt(np.log(8) / gam**2),
        d_center + limits * np.sqrt(np.log(8) / guv**2),
        25
    )
    
    print('plot to show the eta stuff')
    eta =  MU.cfin_sum_in_eta_int(
                Ei, 1, 0, 0, 0, Deigen, A1_funcs, A2_funcs, Fo, 100/fsperau, cfunc,
                delta_mesh, gam, w, 20, plot=True, limits=2.5)
    
    print('plot to show the non-eta')
    no_eta = MU.cfin_sum_in(
                Ei, 1, 0, 0, 0, Deigen, A1_funcs, A2_funcs, Fo, 100/fsperau, cfunc,
                delta_mesh, gam, w, 20, plot=True
            )
    
    print(f'Integration with the eta derivation {eta} and without the eta derivaiton {no_eta}')
    

def spec_line(i, A1_funcs, A2_funcs, Deigen, e_axis, delays, params):
    # Extract parameters
    Fo = params['Fo']
    w = params['w']
    wuv = params['wuv']
    gam = params['gam']
    guv = params['guv']
    limits = params['limits']
    

    # Normalized gaussian for the XUV
    Zcoeffs_init = lambda x: A1_funcs[0](x)+A1_funcs[1](x)
    cfunc = lambda x: MU.norm_gauss((x-wuv),2/guv)*Zcoeffs_init(x)
   
    spec_l = np.zeros(len(delays))
    
    Ei = e_axis[i] / evperAU
    d_center = (guv**-2 * (Ei + 2 * w) + gam**-2 * wuv) / (guv**-2 + gam**-2)
    delta_mesh = np.linspace(
        d_center - limits * np.sqrt(np.log(8) / gam**2),
        d_center + limits * np.sqrt(np.log(8) / guv**2),
        7
    )
    for j in range(len(delays)):
        to = delays[j]
        if to > 100 / fsperau:
            spec_l[j] = np.abs(MU.cfin_sum_in_eta_int(
                Ei, 1, 0, 0, 0, Deigen, A1_funcs, A2_funcs, Fo, to, cfunc,
                delta_mesh, gam, w, 20, plot=False, limits=2.5
            ))**2 * (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2)
        else:
            spec_l[j] = np.abs(MU.cfin_sum_in(
                Ei, 1, 0, 0, 0, Deigen, A1_funcs, A2_funcs, Fo, to, cfunc,
                delta_mesh, gam, w, 20
            ))**2 * (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2)
    
    return (i, spec_l)

def spec_line_fft(i, A1_funcs, A2_funcs, Deigen, e_axis, freqs, params):
    # Extract parameters
    Fo = params['Fo']
    w = params['w']
    wuv = params['wuv']
    gam = params['gam']
    guv = params['guv']
    limits = params['limits']
    

    # Normalized gaussian for the XUV
    Zcoeffs_init = lambda x: A1_funcs[0](x)+A1_funcs[1](x)
    cfunc = lambda x: MU.norm_gauss((x-wuv),2/guv) * Zcoeffs_init(x)
   
    spec_l = np.zeros(len(freqs))
    
    Ei = e_axis[i] / evperAU
    d_center = (guv**-2 * (Ei + 2 * w) + gam**-2 * wuv) / (guv**-2 + gam**-2)
    delta_mesh = np.linspace(
        d_center - limits * np.sqrt(np.log(8) / gam**2),
        d_center + limits * np.sqrt(np.log(8) / guv**2),
        7
    )
    for j in range(len(freqs)):
        to = freqs[j]
        spec_l[j] = np.abs(MU.c_omega_sum_in(
                Ei, 1, 0, 0, 0, Deigen, A1_funcs, A2_funcs, Fo, cfunc,
                delta_mesh, gam, w, 20,to
            ) * (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2))
    
    return (i, spec_l)

def compute_spec_parallel(A1_funcs, A2_funcs, Deigen, e_axis, delays, params):
    state_loc_1 = params['state_loc_1']
    w = params['w']
    wuv = params['wuv']
    per = params['per']
    
    # Initialize spectrogram array
    spec = np.zeros((len(e_axis), len(delays)))
    #func = lambda x: spec_line(x,A1_funcs,A2_funcs,Deigen,e_axis,params)
    print('We are passing to the pool of worker the following dipole: ')
    print(Deigen)
    pool = Pool()
    arguments = [[i,A1_funcs, A2_funcs, Deigen, e_axis, delays, params] for i in range(len(e_axis))]
    res = pool.starmap(spec_line,arguments)
    
    for i in range(len(res)):
        spec[res[i][0],:] = res[i][1]
        
    # Plot spectrogram
    fig, axx = plt.subplots(1, 2,figsize=(10,5))
    im = axx[0].imshow(spec, origin='lower',
                       extent=[delays[0] * fsperau, delays[-1] * fsperau, e_axis[0], e_axis[-1]],
                       aspect='auto', cmap='turbo')
    fig.colorbar(im, ax=axx[0])
    im = axx[1].imshow(np.sqrt(spec), origin='lower',
                       extent=[delays[0] * fsperau, delays[-1] * fsperau, e_axis[0], e_axis[-1]],
                       aspect='auto', cmap='turbo')
    fig.colorbar(im, ax=axx[1])

    axx[0].set_title('Closed channel function norm squared')
    axx[1].set_title('Square root of the norm squared (to adjust the color scale)')

    for ax in axx:
        ax.axhline(state_loc_1[2], color='blue')
        ax.axhline(state_loc_1[3], color='blue')
        ax.axhline(state_loc_1[0] - 2 * w * evperAU, color='green')
        ax.axhline(state_loc_1[1] - 2 * w * evperAU, color='green')
        ax.axhline((wuv - 2 * w) * evperAU)

    for i in range(8):
        ax.axvline(fsperau * i * np.pi / per, color='white')

    plt.savefig('Spectrogram_parallel.png',dpi=210)
    plt.close()
    
    
    np.save('x_axis.npy',delays)
    np.save('y_axis.npy',e_axis)
    np.save('spec_data.npy',spec)
    
    freqs = fft.fftfreq(len(delays),d=(delays[1]-delays[0])/fsperau)[:len(delays)//2] * 2 * np.pi * evperAU
    
    spec_fft = fft.fft(spec,axis=1)[:,:len(delays)//2]
    
    fig, ax = plt.subplots()
    
    X,Y = np.meshgrid(freqs, e_axis)
    
    ax.pcolormesh(X,Y, np.abs(spec_fft), cmap='turbo')
    
    plt.savefig('Spectrogram_fft.png',dpi=210)
    
    return spec

def compute_spec_parallel_fft(A1_funcs, A2_funcs, Deigen, e_axis, freqs, params):
    state_loc_1 = params['state_loc_1']
    w = params['w']
    wuv = params['wuv']
    per = params['per']
    
    # Initialize spectrogram array
    spec = np.zeros((len(e_axis), len(freqs)))
    #func = lambda x: spec_line(x,A1_funcs,A2_funcs,Deigen,e_axis,params)
    print('We are passing to the pool of worker the following dipole: ')
    print(Deigen)
    pool = Pool()
    print("The created pool is using "+str(pool._processes)+" workers.")
    arguments = [[i,A1_funcs, A2_funcs, Deigen, e_axis, freqs, params] for i in range(len(e_axis))]
    res = pool.starmap(spec_line_fft,arguments)
    
    for i in range(len(res)):
        spec[res[i][0],:] = res[i][1]
        
    # Plot spectrogram
    fig, axx = plt.subplots(1, 2,figsize=(10,5))
    im = axx[0].imshow(spec, origin='lower',
                       extent=[freqs[0] * evperAU, freqs[-1] * evperAU, e_axis[0], e_axis[-1]],
                       aspect='auto', cmap='turbo')
    fig.colorbar(im, ax=axx[0])
    im = axx[1].imshow(np.sqrt(spec), origin='lower',
                       extent=[freqs[0] * evperAU, freqs[-1] * evperAU, e_axis[0], e_axis[-1]],
                       aspect='auto', cmap='turbo')
    fig.colorbar(im, ax=axx[1])

    axx[0].set_title('Closed channel function norm squared')
    axx[1].set_title('Square root of the norm squared (to adjust the color scale)')

    for ax in axx:
        ax.axhline(state_loc_1[2], color='blue')
        ax.axhline(state_loc_1[3], color='blue')
        ax.axhline(state_loc_1[0] - 2 * w * evperAU, color='green')
        ax.axhline(state_loc_1[1] - 2 * w * evperAU, color='green')
        ax.axhline((wuv - 2 * w) * evperAU)
        
    beats = []
    for f in state_loc_1:
        for g in state_loc_1:
            if(f!=g):
                beats.append(abs(f-g)/evperAU)
                
    for ax in axx:
        for f in beats:
            if(f<freqs[-1]):
                ax.axvline(f*evperAU)

    plt.savefig('Spectrogram_parallel_fft_comp.png',dpi=210)
    plt.close()
    
    
    np.save('x_axis_fft.npy',freqs)
    np.save('y_axis_fft.npy',e_axis)
    np.save('spec_data_fft.npy',spec)
    return spec

def compute_spectrogram(A1_funcs, A2_funcs, Deigen, e_axis, delays, params):
    """
    Compute and plot the spectrogram.

    Parameters:
    A1_funcs : list of functions
        List of functions that calculate normalized coefficients for symmetry S1.
    A2_funcs : list of functions that calculate normalized coefficients for symmetry S2.
    Deigen : ndarray
        Dipole matrix.
    e_axis : array_like
        Energy axis for the spectrogram (in eV).
    delays : array_like
        Delay times for the spectrogram.
    params : dict
        Dictionary of parameters needed for the computation (e.g., Fo, w, wuv, gam, guv, etc.).
    """
    # Extract parameters
    Fo = params['Fo']
    w = params['w']
    wuv = params['wuv']
    gam = params['gam']
    guv = params['guv']
    limits = params['limits']
    per = params['per']
    state_loc_1 = params['state_loc_1']
    # Initialize spectrogram array
    spec = np.zeros((len(e_axis), len(delays)))

    # Normalized gaussian for the XUV
    cfunc = lambda x: MU.norm_gauss((x-wuv),2/guv)
    
    # Compute spectrogram
    for i in range(len(e_axis)):
        Ei = e_axis[i] / evperAU
        d_center = (guv**-2 * (Ei + 2 * w) + gam**-2 * wuv) / (guv**-2 + gam**-2)
        delta_mesh = np.linspace(
            d_center - limits * np.sqrt(np.log(8) / gam**2),
            d_center + limits * np.sqrt(np.log(8) / guv**2),
            7
        )
        for j in range(len(delays)):
            to = delays[j]
            if to > 100 / fsperau:
                spec[i, j] = np.abs(MU.cfin_sum_in_eta_int(
                    Ei, 1, 0, 0, 0, Deigen, A1_funcs, A2_funcs, Fo, to, cfunc,
                    delta_mesh, gam, w, 20, plot=False, limits=2.5
                ))**2 * (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2)
            else:
                spec[i, j] = np.abs(MU.cfin_sum_in(
                    Ei, 1, 0, 0, 0, Deigen, A1_funcs, A2_funcs, Fo, to, cfunc,
                    delta_mesh, gam, w, 20
                ))**2 * (np.abs(A1_funcs[0](Ei))**2 + np.abs(A1_funcs[1](Ei))**2)

    # Plot spectrogram
    fig, axx = plt.subplots(1, 2,figsize=(10,5))
    im = axx[0].imshow(spec, origin='lower',
                       extent=[delays[0] * fsperau, delays[-1] * fsperau, e_axis[0], e_axis[-1]],
                       aspect='auto', cmap='turbo')
    fig.colorbar(im, ax=axx[0])
    im = axx[1].imshow(np.sqrt(spec), origin='lower',
                       extent=[delays[0] * fsperau, delays[-1] * fsperau, e_axis[0], e_axis[-1]],
                       aspect='auto', cmap='turbo')
    fig.colorbar(im, ax=axx[1])

    axx[0].set_title('Closed channel function norm squared')
    axx[1].set_title('Square root of the norm squared (to adjust the color scale)')

    for ax in axx:
        ax.axhline(state_loc_1[2], color='blue')
        ax.axhline(state_loc_1[3], color='blue')
        ax.axhline(state_loc_1[0] - 2 * w * evperAU, color='green')
        ax.axhline(state_loc_1[1] - 2 * w * evperAU, color='green')
        ax.axhline((wuv - 2 * w) * evperAU)

    for i in range(8):
        ax.axvline(fsperau * i * np.pi / per, color='white')

    plt.savefig('Spectrogram_1.png',dpi=210)
    
    np.save('x_axis.npy',delays)
    np.save('y_axis.npy',e_axis)
    np.save('spec_data.npy',spec)
    
    return spec