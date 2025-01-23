import pytest
from liteczos.lanczos import maxc_hamiltonian

@pytest.fixture(
    name="U",
    params=[0, 1, 8],
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

    