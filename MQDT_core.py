import numpy as np
import scipy as sc

# Definition of the K matrix from the mu matrix
def Km(mu_mat):
    return sc.linalg.tanm(np.pi*mu_mat)

def nu(e,I):
    return np.sqrt(0.5/(I-e))

def k(e,I):
    return np.sqrt(2*(e-I))

def beta(e,I,l):
    return np.pi*(nu(e,I)-l)

def eta(e, I, l):
    sigma = np.angle(sc.special.gamma(l+1-1j/k(e,I)))
    return 1/k(e,I) * np.log(2*k(e,I)) + sigma -1/2 * np.pi * l
# Definition of the solution of the generalized eigen-value problem
def GEV_P(E, Km, Is, Ls):
    # determine open channels
    nc = len(Is)
    op = []
    cl = []
    for i in range(nc):
        if(Is[i]>E):
            cl.append(i)
        else:
            op.append(i)
            
    # Form the matrices that go into the generalized problem
    LHS = np.zeros((nc,nc))
    RHS = np.zeros((nc,nc))
    
    for i in range(nc):
        if(i in op):
            LHS[i,:] = Km[i,:]
            RHS[i,i] = 1.0
        else:
            row = np.zeros_like(Is)
            sinbet = np.sin(beta(E,Is[i],Ls[i]))
            cosbet = np.cos(beta(E,Is[i],Ls[i]))
            row[i] = 1
            LHS[i,:] = sinbet*row + cosbet*Km[i,:] 
    
    # now we solve the generalized problem
    
    eigvals, eigvec = sc.linalg.eig(LHS,RHS)
    
    taus = []
    cs = []
    
    for i in range(nc):
        if(np.isfinite(eigvals[i])):
            taus.append(np.real(np.arctan(eigvals[i])/np.pi))
            cs.append(eigvec[:,i])
    
    # Z coeffs
    # Now we compute the Z coefficients from the fomula 
    
    Smatd = sc.linalg.inv(np.eye(nc)+1j*Km)@(np.eye(nc)-1j*Km) 
    et = np.zeros(len(op),dtype=complex)
    bet = np.zeros(len(cl), dtype=complex)
    
    o = 0
    c = 0
    for i in range(nc):
        if(i in op):
            et[o] = eta(E,Is[i],Ls[i])
            o+=1
        else:
            bet[c] = beta(E,Is[i],Ls[i])
            c+=1
    
    Scc = Smatd[np.ix_(cl,cl)]
    Sco = Smatd[np.ix_(cl,op)]
    Soo = Smatd[np.ix_(op,op)]
    Soc = Smatd[np.ix_(op,cl)]
    
    Z = np.diag(np.exp(1j*bet))@sc.linalg.inv(Scc-np.diag(np.exp(2j*bet)))@Sco@np.diag(np.exp(-1j*et))
    
    Sphy = np.diag(np.exp(-1j*et))@(Soo-Soc@sc.linalg.inv(Scc-np.diag(np.exp(2j*bet)))@Sco)@np.diag(np.exp(-1j*et))
    
    
    return taus, cs, Z, np.real(Sphy)