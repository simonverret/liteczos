import numpy as np
from numpy import pi
from scipy.linalg import eigh_tridiagonal, eigvalsh_tridiagonal, eigvalsh


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
    max_iter = min(max_iter, dim)

    a_list = np.zeros(max_iter)
    b_list = np.zeros(max_iter)
    
    #n=0
    phi = np.random.uniform(-1,1, dim)
    prev_phi = np.zeros_like(phi)

    H_phi = H@phi
    phi_phi = phi@phi
    a = phi@H_phi / phi_phi
    b_square = 0
    
    a_list[0] = a
    b_list[0] = np.sqrt(b_square)

    for n in range(1, max_iter):
        new_phi = H_phi - a*phi - b_square*prev_phi
        prev_phi = phi
        phi = new_phi
        prev_phi_phi = phi_phi

        H_phi = H@phi
        phi_phi = phi@phi
        a = phi@H_phi / phi_phi
        b_square = phi_phi / prev_phi_phi

        a_list[n] = a
        b_list[n] = np.sqrt(b_square)

        e = min(eigvalsh_tridiagonal(a_list[:n], b_list[1:n]))
    
    return e
        

def main():
    from matplotlib import pyplot as plt
    t = 1
    U_list = np.linspace(-8,8, 50)
    L_list = np.zeros_like(U_list)
    E_list = np.zeros_like(U_list)
    N_list = np.zeros_like(U_list)
    for i, U in enumerate(U_list):
        H = maxc_hamiltonian(t,U)
        L_list[i] = get_ground_state(H)
        N_list[i] = min(eigvalsh(H))
        E_list[i] = min(
            0,
            U,
            t,
            U+t, 
            U-t,
            (U+np.sqrt(16*t**2 + U**2))/2,
            (U-np.sqrt(16*t**2 + U**2))/2
        )
    
    plt.plot(U_list, E_list)
    plt.plot(U_list, L_list)
    plt.plot(U_list, N_list)
    plt.show()