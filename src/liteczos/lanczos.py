import numpy as np
from numpy import pi
from scipy.linalg import eigh_tridiagonal, eigh

LARGEST_FLOAT = np.finfo(np.float32).max

def maxc_hamiltonian(t, U):
    return np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0,-t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0,-t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0,-t, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, U,-t, 0, 0,-t, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,-t, 0, 0, 0, 0,-t, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, U, 0, 0, 0,-t, 0, 0, 0, 0],
        [0, 0, 0, 0,-t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0,-t, 0, 0, 0, 0,-t, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0,-t, 0, 0,-t, U, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0,-t, 0, 0, 0, U, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, U,-t, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-t, U, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,2*U],
    ])


def get_ground_state(H, max_iter=100, tol=1e-9):
    dim = len(H)
    a_list = np.zeros(max_iter)
    b_list = np.zeros(max_iter)
    normalized_phi_list = np.zeros(shape=(max_iter, dim))  # to store 
    
    phi = np.random.uniform(-1,1, dim)
    prev_phi = np.zeros_like(phi)
    prev_phi_phi = 1
    prev_e0 = LARGEST_FLOAT

    for n in range(0, max_iter):
        # store the unit vector
        phi_phi = phi@phi
        normalized_phi_list[n] = phi / np.sqrt(phi_phi)
        
        # compute next vector
        H_phi = H@phi
        a = phi@H_phi / phi_phi
        b_square = phi_phi / prev_phi_phi
        new_phi = H_phi - a*phi - b_square*prev_phi

        # accumulate tridiagonal hamiltonian
        a_list[n] = a
        b_list[n] = np.sqrt(b_square)

        # get eigenvalues
        if n > 0:
            eigvals, eigvecs = eigh_tridiagonal(a_list[:n], b_list[1:n])
            gs_index = eigvals.argmin()
            e0 = eigvals[gs_index]
            phi_v0 = eigvecs[:,gs_index]            
            if prev_e0 - e0 < tol or b_square < tol: break
            prev_e0 = e0

        prev_phi = phi
        phi = new_phi
        prev_phi_phi = phi_phi
    
    v0 = phi_v0@(normalized_phi_list[:n])
    v0 = v0 / np.sqrt(v0@v0)
    return e0, v0