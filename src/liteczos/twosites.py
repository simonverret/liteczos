import numpy as np


def hamiltonian(t, U):
    return np.array([
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


def ground_state_energy(t, U):
    return 0.5*(U - np.sqrt(16*t**2 + U**2))


def ground_state_vector(t, U):
    ## Work as well
    # lp = 0.5*(U + np.sqrt(16*t**2 + U**2))
    # return np.array([0, 0, 0, 0, 0, 2*t, lp, 0, 0, lp, 2*t, 0, 0, 0, 0, 0])
    gs = ground_state_energy(t,U)
    norm = np.sqrt(2*(4*t**2 + gs**2))
    return np.array([0, 0, 0, 0, 0, -gs, 2*t, 0, 0, 2*t, -gs, 0, 0, 0, 0, 0])/norm


def get_green_function(mu, t, U):
    gs = ground_state_energy(t,U)
    norm = np.sqrt(2*(gs**2+4*t**2))
    a = -gs/norm
    b = 2*t/norm

    # the green function will be an actual python function of frequency z
    def green_function(z):
        ## Charlebois thesis eq (C.12)
        g1e = .5*(a-b)**2/(z-(U+t-gs))
        g2e = .5*(a+b)**2/(z-(U-t-gs))
        giie = g1e + g2e
        gije = g1e - g2e

        ## Charlebois thesis eq (C.12)
        g1h = .5*(a+b)**2/(z-(gs+t))
        g2h = .5*(a-b)**2/(z-(gs-t))
        giih = g1h+g2h
        gijh = g1h-g2h
        
        ## Charlebois thesis eq (1.13)
        return np.array([
            #   1up    2up    1dn    2dn
            [ giie+giih, gije+gijh,         0,         0], # 1up
            [ gije+gijh, giie+giih,         0,         0], # 2up
            [         0,         0, giie+giih, gije+gijh], # 1dn
            [         0,         0, gije+gijh, giie+giih], # 2dn
        ])
    
    # We return the python function
    return green_function