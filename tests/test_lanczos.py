import numpy as np
from scipy.linalg import eigvals
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

def test_maxc_hamiltonian(U, t):
    H = maxc_hamiltonian(U,t)
    assert H.shape == (16,16)

def test_ground_state(U, t):
    H = maxc_hamiltonian(U, t)
    E0 = get_ground_state(H)
    expected = min(eigvals(H))
    assert np.allclose(E0, expected)