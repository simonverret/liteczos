import numpy as np
from numpy import pi
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal


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


def get_ground_state(H, max_iter=20, tol=1e-9):
    dim = len(H)

    phi = np.random.randn(dim)
    prev_phi = np.zeros_like(phi)
    prev_phi_phi = 1
    e = 1e10
    
    a_list = np.zeros(max_iter)
    b_list = np.zeros(max_iter)

    for n in range(max_iter):
        H_phi = H@phi
        phi_phi = phi@phi
        a = phi@H_phi / phi_phi
        b = phi_phi / prev_phi_phi
        
        next_phi = H_phi - a*phi - b*prev_phi
        
        prev_phi = phi
        phi = next_phi
        prev_phi_phi = phi_phi

        a_list[n] = a
        b_list[n] = b

        if n > 1:
            prev_e = e
            e = eigvalsh_tridiagonal(a_list[:n], b_list[1:n]).min()  #first b is not in the matrix
            if b < tol or np.allclose(e, prev_e, atol=tol):
                return e
        

def main():
    H = maxc_hamiltonian(1,8)
    print("problem:")
    print(H)

    print()
    print("solution")
    print(get_ground_state(H, max_iter=16))
    # print(ground_state(H))