import numpy as np
from scipy.linalg import eigh           
import pytest
from liteczos.lanczos import maxc_hamiltonian, get_ground_state

@pytest.fixture(
    name="U",
    params=[0, 1.2, 8],
    ids=["U=0", "U=1", "U=8"]
)
def Uval(request):
    return request.param

@pytest.fixture(
    name="t",
    params=[1],
    ids=["t=1"]
)
def t(request):
    return request.param

def test_maxc_hamiltonian(t, U):
    H = maxc_hamiltonian(t, U)
    assert H.shape == (16,16)

def test_ground_state(t, U):
    H = maxc_hamiltonian(t, U)
    
    eigvals, eigvecs = eigh(H)
    gs_index = eigvals.argmin() 
    expected_e0 = eigvals[gs_index]
    expected_v0 = eigvecs[:,gs_index]

    e0, v0 = get_ground_state(H)
    
    assert np.allclose(e0, expected_e0)
    assert np.allclose(v0, expected_v0, atol=1e-5)
