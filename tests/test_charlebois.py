import pytest
import numpy as np
from scipy.linalg import eigh, eigvalsh
from liteczos.charlebois import hamiltonian, number, ground_state_energy, ground_state_vector


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


@pytest.fixture
def mu_half_filled(U):
    return U/2


def test_hamiltonian(t, U):
    H = hamiltonian(t, U)
    assert H.shape == (16,16)

def test_ground_state_energy(mu_half_filled, t, U):
    H = hamiltonian(t, U) - mu_half_filled*number()

    
    expected_e0 = eigvalsh(H).min()
    e0 = ground_state_energy(t, U) - mu_half_filled*2
    assert np.allclose(e0, expected_e0)


def test_ground_state_vector(mu_half_filled, t, U):
    H = hamiltonian(t, U) - mu_half_filled*number()

    ## The following fails because of degenerancy:
    # eigvals, eigvecs = eigh(H)
    # expected_v0 = eigvecs[eigvals.argmin()]
    # assert np.allclose(v0, expected_v0) or np.allclose(v0, -expected_v0) 

    expected_e0 = eigvalsh(H).min()    
    v0 = ground_state_vector(t, U)
    assert np.allclose((H@v0)/expected_e0, v0)

