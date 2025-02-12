import pytest
import numpy as np
from scipy.linalg import eigvalsh, inv
from liteczos.twosites import *


@pytest.fixture(
    name="t",
    params=[1],
    ids=["t=1"]
)
def t(request):
    return request.param


@pytest.fixture(
    name="U",
    params=[0., 1.2, 8.],
    ids=["U=0", "U=1", "U=8"]
)
def U(request):
    return request.param


def test_hamiltonian(t, U):
    H = hamiltonian(t, U)
    assert H.shape == (16,16)


def test_ground_state_energy(t, U):
    H = hamiltonian(t, U)    
    expected_e0 = eigvalsh(H).min()
    e0 = ground_state_energy(t, U)
    assert np.allclose(e0, expected_e0)


def test_ground_state_vector(t, U):
    H = hamiltonian(t, U)
    expected_e0 = eigvalsh(H).min()    
    v0 = ground_state_vector(t, U)
    assert np.allclose((H@v0)/expected_e0, v0)


def test_cdag1up(t,U):
    v0 = ground_state_vector(t, U)
    v1 = cdag1up() @ v0
    a, b = get_alpha_beta(t,U)
    expected_v1 = np.zeros(16)
    expected_v1[13] = a
    expected_v1[14] = b
    assert np.allclose(v1, expected_v1)


# def test_green_function(t, U):
#     gf = get_green_function(t, U)
#     assert callable(gf)

#     cdag1u_gs = cdag1up() @ ground_state_vector(t,U)
#     H = hamiltonian(t, U)
#     expected_gf = lambda z: cdag1u_gs @ inv(z*np.eye(16) - H) @ cdag1u_gs
#     assert gf(0+1j*0.1) == expected_gf(0+1j*0.1)