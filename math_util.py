import numpy as np
import scipy as sc
import sympy.physics.wigner as wg
import numpy.polynomial.legendre as le
import matplotlib.pyplot as plt

# Fadeeva function
def w(z):
    return sc.special.wofz(z) #np.exp(-z**2)*(1+1j*sc.special.erfi(z))

# Gaussian function
def gauss(z):
    return np.exp(-z**2)

# Normalized gaussian, with width g
def norm_gauss(z,g):
    return np.exp(-(z/g)**2) / np.sqrt(np.sqrt(np.pi*g**2/2))

# Complex gamma function part of the Coulomb phase-shift. 
def sigma(E,l,I):
    Gam = sc.special.gamma(l+1-1j*np.sqrt(1/(2*(E-I))))
    return np.angle(Gam)

# Performs the integration over the intermediate state energies and returns the two dimensional array
# of the integrals as a function of E and the energy of the initial state delta that should be 
# integrated to get the transition amplitude.

def Xi_int_for_viewing(E, Js1,Js2, M1, M2, Deigen, As1_funcs, As2_funcs, Fo, c_func, delta_mesh, g, wo, Iref,interm_lim,plot=False):
    """
    Compute the real and imaginary parts of the integral for viewing purposes.
    Parameters:
    E (float): Energy value.
    Js1 (int): Total angular momentum quantum number for state 1.
    Js2 (int): Total angular momentum quantum number for state 2.
    M1 (int): Magnetic quantum number for state 1.
    M2 (int): Magnetic quantum number for state 2.
    Deigen (numpy.ndarray): Dipole matrix elements.
    As1_funcs (list of functions): List of functions for state 1's coefficients.
    As2_funcs (list of functions): List of functions for state 2's coefficients.
    Fo (float): Field amplitude.
    c_func (function): Function for c the initial amplitude of the wavepacket.
    delta_mesh (numpy.ndarray): Mesh of delta values.
    g (float): width of the IR laser.
    wo (float): Frequency of the IR laser.
    Iref (float): Reference threshold.
    interm_lim (tuple): Intermediate limits for integration, as effective quantum number to the reference threshold.
    plot (bool, optional): Whether to plot the results. Default is False
    Returns:
    tuple: Real and imaginary parts of the integral as numpy arrays.
    """

    prefac = (-1)**(Js1+Js2-M1-M2) * float(wg.wigner_3j(Js1,1,Js2,-M1,0,M2)) \
            * float(wg.wigner_3j(Js2,1,Js1,-M2,0,M1)) * -1j * 2 * np.pi * g**2 * Fo**2
    
    # Degree of the quadrature.
    quad_degree = 150 
    
    # Handy form of the .
    As1 = lambda x: np.array([a(x) for a in As1_funcs])
    As2 = lambda x: np.array([a(x) for a in As2_funcs])
    
    nXi_int_r = lambda delta:  sc.integrate.fixed_quad(lambda x: \
        np.real(np.dot(np.conjugate(np.transpose(As1(E))),Deigen@As2(x))*\
        np.dot(np.transpose(np.conjugate(Deigen@As2(x))),As1(delta))*\
        w((E+delta-2*x)*g/np.sqrt(8))),Iref-0.5/interm_lim[0]**2, Iref-0.5/interm_lim[1]**2,n=quad_degree)
    
    nXi_int_i = lambda delta:  sc.integrate.fixed_quad(lambda x: \
        np.imag(np.dot(np.transpose(np.conjugate(As1(E))),Deigen@As2(x))*\
        np.dot(np.transpose(np.conjugate(Deigen@As2(x))),As1(delta))*\
        w((E+delta-2*x)*g/np.sqrt(8))),Iref-0.5/interm_lim[0]**2, Iref-0.5/interm_lim[1]**2,n=quad_degree)
    
    nXi_int = lambda x: nXi_int_r(x)[0] + 1j*nXi_int_i(x)[0]
    
    Xi_int = np.vectorize(nXi_int)
    
    nintegrand_r  = lambda x: np.real(Xi_int(x) * c_func(x)*\
                                gauss(g/np.sqrt(8) * (E+2*wo-x)))
        
    nintegrand_i  = lambda x: np.imag(Xi_int(x) * c_func(x)*\
                            gauss(g/np.sqrt(8) * (E+2*wo-x)))
        
    integrand_r = np.vectorize(nintegrand_r)
    integrand_i = np.vectorize(nintegrand_i)
        
    return prefac * integrand_r(delta_mesh), prefac * integrand_i(delta_mesh)