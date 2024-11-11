import numpy as np
import scipy as sc
import sympy.physics.wigner as wg
import numpy.polynomial.legendre as le
import matplotlib.pyplot as plt


def w(z):
    return sc.special.wofz(z) #np.exp(-z**2)*(1+1j*sc.special.erfi(z))

def gauss(z):
    return np.exp(-z**2)

def norm_gauss(z,g):
    return np.exp(-(z/g)**2) / np.sqrt(np.sqrt(np.pi*g**2/2))
    
def sigma(E,l,I):
    Gam = sc.special.gamma(l+1-1j*np.sqrt(1/(2*(E-I))))
    return np.angle(Gam)


def cfin(E, Js1,Js2, M1, M2,dim1, dim2, Deigen, As1, As2, Fo, to, c_delta, eranges, g, wo):
    res = 0.0+1j*0.0
    prefac = (-1)**(Js1+Js2-M1-M2) * wg.wigner_3j(Js1,1,Js2,-M1,0,M2) * wg.wigner_3j(Js2,1,Js1,-M2,0,M1) * -1j * 2 * np.pi * g**2 * Fo**2
    eindx = np.argmin(np.abs(eranges-E))
    for a in range(dim1):
     for ap in range(dim1):
      for b in range(dim2):
       for bp in range(dim2):
           term = Deigen[a,b]*Deigen[ap,bp]*As1[eindx,a]
           
           xi_int = np.zeros_like(eranges,dtype=complex)
           for i,e in enumerate(eranges):
            xi_int[i] = np.trapz(As2[:,bp]*As2[:,b]*w((E+e-2*eranges)*g/np.sqrt(8)),x=eranges)
           
           delta_int = np.trapz(c_delta*
                                gauss((E+2*wo-eranges)*g/np.sqrt(8))*
                                As1[:,ap]*
                                np.exp(1j*(E-eranges)*to)*
                                xi_int)
           
           res += term * delta_int
    
    return prefac * res  

def cfin_sum_in(E, Js1,Js2, M1, M2, Deigen, As1_funcs, As2_funcs, Fo, to, 
                c_func, delta_mesh, g, wo, degree):
    
    prefac = (-1)**(Js1+Js2-M1-M2) * wg.wigner_3j(Js1,1,Js2,-M1,0,M2) * wg.wigner_3j(Js2,1,Js1,-M2,0,M1) * -1j * 2 * np.pi * g**2 * Fo**2
    
    #fig, ax = plt.subplots()
    #ax2 = ax.twinx()
    ndelta = len(delta_mesh)-1
    ir_range = 2.5*np.sqrt(np.log(1e2)/(g**2 / 8))
    dim1 = len(As1_funcs)
    dim2 = len(As2_funcs)
    
    prim_points, prim_weights = le.leggauss(degree)
    # correct the points to be from 0 to 1
    prim_points = (prim_points+1)/2
    prim_weights = prim_weights/2
    As1_e= np.zeros(dim1)
    for k in range(dim1):
        As1_e[k] = As1_funcs[k](E)
    
    delta_int  = 0
    for i in range(ndelta):
        #ax.axvline(delta_mesh[i])
        d_size = delta_mesh[i+1]-delta_mesh[i]
        delt_ps = prim_points*d_size+delta_mesh[i]
        #ax.plot(delt_ps, np.ones_like(delt_ps))
        delt_ws = prim_weights*d_size 
        As1_delta = np.zeros(dim1,dtype=complex)
        xi_int = np.zeros(degree,dtype=complex)
        for d in range(degree):
            for k in range(dim1):
                As1_delta[k] = As1_funcs[k](delt_ps[d])
            
            xilo = 0.5*(E+delt_ps[d])-ir_range
            xi_points = prim_points*2*ir_range + xilo
            xi_ws = prim_weights*2*ir_range
            As2_xi = np.zeros((degree, dim2),dtype=complex)
            
            for k in range(dim2):
                As2_xi[:,k] = As2_funcs[k](xi_points)
                
                #ax2.plot(xi_points,np.real( As2_xi[:,k]))
        
                #ax2.plot(xi_points,np.imag( As2_xi[:,k]),'--')
            
            dipoles = np.array([np.dot(np.conjugate(As1_delta),Deigen@As2_xi[k,:])*\
                                np.dot(np.conjugate(Deigen@As2_xi[k,:]),As1_e) for k in range(degree)])
            
            ww = w((E+delt_ps[d]-2*xi_points)*g/np.sqrt(8))
            
            
            xi_int[d] = np.sum(ww*dipoles*xi_ws)

            #for x in xi_points: ax.axvline(x,alpha=0.1)
        
            #plt.show()
            #STOP
        
        
        delta_int += np.sum(xi_int*c_func(delt_ps)*
                            gauss(g/np.sqrt(8) * (E+2*wo-delt_ps)) *
                            np.exp(1j*(E-delt_ps)*to)*delt_ws)
    
    
    
    return delta_int * prefac


def cfin_sum_in_eta_int(E, Js1,Js2, M1, M2, Deigen, As1_funcs, As2_funcs, Fo, to, 
                c_func, delta_mesh, g, wo, degree,plot=False,limits=1.5):
    
    prefac = (-1)**(Js1+Js2-M1-M2) * wg.wigner_3j(Js1,1,Js2,-M1,0,M2) * wg.wigner_3j(Js2,1,Js1,-M2,0,M1) * -1j * 2 * np.pi * g**2 * Fo**2
    
    # In this case we would like to make sure that there are enough points to capture the oscillation of the delay term
    
    eta_mesh = (delta_mesh-E)*to
    eta_mesh_fine = np.linspace(eta_mesh[0],eta_mesh[-1],int((eta_mesh[-1]-eta_mesh[0])/(np.pi)))
    print(f"Mesh in eta for energy {E} and delay {to}, has {len(eta_mesh_fine)} sectors, compare to the sectors in the original delta mesh {len(delta_mesh)}")
    if(plot): fig, ax = plt.subplots()
    #ax2 = ax.twinx()
    neta = len(eta_mesh_fine)-1
    ir_range = limits*np.sqrt(np.log(1e2)/(g**2 / 8))
    dim1 = len(As1_funcs)
    dim2 = len(As2_funcs)
    
    prim_points, prim_weights = le.leggauss(degree)
    # correct the points to be from 0 to 1
    prim_points = (prim_points+1)/2
    prim_weights = prim_weights/2
    As1_e= np.zeros(dim1)
    
    for k in range(dim1):
        As1_e[k] = As1_funcs[k](E)
    
    delta_int  = 0
    for i in range(neta):
        d_size = eta_mesh_fine[i+1]-eta_mesh_fine[i]
        eta_ps = prim_points*d_size+eta_mesh_fine[i]
        eta_ws = prim_weights*d_size 
        As1_delta = np.zeros(dim1,dtype=complex)
        xi_int = np.zeros(degree,dtype=complex)
        
        for d in range(degree):
            for k in range(dim1):
                As1_delta[k] = As1_funcs[k](eta_ps[d]/to+E)
            
            xilo = 0.5*(2*E+eta_ps[d]/to)-ir_range
            xi_points = prim_points*2*ir_range + xilo
            xi_ws = prim_weights*2*ir_range
            As2_xi = np.zeros((degree, dim2),dtype=complex)
            
            for k in range(dim2):
                As2_xi[:,k] = As2_funcs[k](xi_points)
            
            dipoles = np.array([np.dot(np.conjugate(As1_delta),Deigen@As2_xi[k,:])*\
                                np.dot(np.conjugate(Deigen@As2_xi[k,:]),As1_e) for k in range(degree)])
            
            ww = w((2*E+eta_ps[d]/to-2*xi_points)*g/np.sqrt(8))
            
            
            xi_int[d] = np.sum(ww*dipoles*xi_ws)

        
        
        delta_int += np.sum(xi_int*c_func(eta_ps/to+E)*
                            gauss(g/np.sqrt(8) * (2*wo-eta_ps/to)) *
                            np.exp(-1j*eta_ps)*eta_ws)
        if(plot):
            plt.axvline(eta_mesh_fine[i])
            plt.plot(eta_ps, np.real(xi_int*c_func(eta_ps/to+E)*
                            gauss(g/np.sqrt(8) * (2*wo-eta_ps/to)) *
                            np.exp(1j*eta_ps)),c=f'C{i}')
            
            plt.plot(eta_ps, np.imag(xi_int*c_func(eta_ps/to+E)*
                            gauss(g/np.sqrt(8) * (2*wo-eta_ps/to)) *
                            np.exp(1j*eta_ps) ),'--',c=f'C{i}')
    
    if(plot): 
        plt.axvline(eta_mesh_fine[-1])
        plt.show()
    
    return delta_int * prefac/to