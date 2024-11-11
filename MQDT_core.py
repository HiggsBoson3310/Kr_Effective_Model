import numpy as np
import scipy as sc

# Definition of the K matrix from the mu matrix
def Km(mu_mat):
    return sc.linalg.tanm(np.pi*mu_mat)

def nu(e,I):
    return np.sqrt(0.5/(I-e))

def beta(e,I,l):
    return np.pi*(nu(e,I)-l)

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
            taus.append(np.arctan(eigvals[i])/np.pi)
            cs.append(eigvec[:,i])
            
    return taus, cs