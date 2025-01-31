import pytest
from liteczos.hamiltonians import maxc_hamiltonian

@pytest.fixture(
    name="t",
    params=[1],
    ids=["t=1"]
)
def t(request):
    return request.param

@pytest.fixture(
    name="U",
    params=[0, 1.2, 8],
    ids=["U=0", "U=1", "U=8"]
)
def U(request):
    return request.param

def test_maxc_hamiltonian(t, U):
    H = maxc_hamiltonian(t, U)
    assert H.shape == (16,16)

    