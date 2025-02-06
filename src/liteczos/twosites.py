import numpy as np


def hamiltonian(t, U):
    H = np.array([
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,-t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0,-t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0,-t, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, U,-t, 0, 0,-t, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0,-t, 0, 0, 0, 0,-t, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, U, 0, 0, 0,-t, 0, 0, 0, 0],
        [ 0, 0, 0, 0,-t, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0,-t, 0, 0, 0, 0,-t, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0,-t, 0, 0,-t, U, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0,-t, 0, 0, 0, U, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, U,-t, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-t, U, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,2*U],
    ])
    return H - (U/2)*number()  # mu=U/2 at half filling


def number():
    return np.array([
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    ])

def cdag1up():
    return np.array([
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])


def lowest_n2_eigval(t,U):
    return 0.5*(U - np.sqrt(16*t**2 + U**2))


def ground_state_energy(t, U):
    return lowest_n2_eigval(t,U) - U  # - mu*N where mu = U/2 and N = 2 (half filling)


def ground_state_vector(t, U):
    a, b = get_alpha_beta(t,U)
    return np.array([0, 0, 0, 0, 0, a, b, 0, 0, b, a, 0, 0, 0, 0, 0])


def get_alpha_beta(t,U):
    lm = lowest_n2_eigval(t,U)
    norm = np.sqrt(2*(lm**2 + 4*t**2))
    a = -lm/norm
    b = 2*t/norm
    return a, b


def get_green_function(t, U):
    ## Charlebois thesis eq (1.13, C.12, C.13)
    lm = lowest_n2_eigval(t,U)
    a, b = get_alpha_beta(t,U)
    def green_function(z):
        g1e = .5*(a-b)**2/(z-(U+t-lm))
        g2e = .5*(a+b)**2/(z-(U-t-lm))
        # g1h = .5*(a+b)**2/(z-(lm+t))
        # g2h = .5*(a-b)**2/(z-(lm-t))
        return g1e + g2e# + g1h + g2h

    # returns a python function of frequency
    return green_function