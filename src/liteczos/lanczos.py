import numpy as np
from numpy import pi
from scipy.linalg import eigh_tridiagonal, eigh

LARGEST_FLOAT = np.finfo(np.float32).max

def get_ground_state(H, max_iter=100, tol=1e-9):
    # containers for results
    dim = len(H)
    a_list = np.zeros(max_iter)
    b_list = np.zeros(max_iter)
    normalized_phi_list = np.zeros(shape=(max_iter, dim))  # to store 
    
    # initialization
    phi = np.random.uniform(-1,1, dim)
    prev_phi = np.zeros_like(phi)
    prev_phi_phi = 1
    prev_e0 = LARGEST_FLOAT

    for n in range(0, max_iter):
        # compute relevant quantities
        H_phi = H@phi
        phi_phi = phi@phi
        a = phi@H_phi / phi_phi
        b_square = phi_phi / prev_phi_phi

        # store the current unit vector
        normalized_phi_list[n] = phi / np.sqrt(phi_phi)
        
        # accumulate tridiagonal hamiltonian
        a_list[n] = a
        b_list[n] = np.sqrt(b_square)

        # is the matrix 2x2 at least?
        if n > 0:
            # get smallest eigenvalue and corresponding eigenvector
            eigvals, eigvecs = eigh_tridiagonal(a_list[:n], b_list[1:n])
            gs_index = eigvals.argmin()
            e0 = eigvals[gs_index]
            phi_v0 = eigvecs[:,gs_index]
            
            # check for convergence
            if prev_e0 - e0 < tol or b_square < tol: 
                break
            prev_e0 = e0

        # compute next vector
        next_phi = H_phi - a*phi - b_square*prev_phi        
        prev_phi = phi
        prev_phi_phi = phi_phi
        phi = next_phi
    
    # express v0 in the original basis and normalize
    v0 = phi_v0@normalized_phi_list[:n]
    v0 = v0 / np.sqrt(v0@v0)
    return e0, v0