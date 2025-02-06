import pytest
import numpy as np
from liteczos import twosites, second_quantized

def test_number_operator():
    assert np.allclose(twosites.number(), second_quantized.number(num_qbits=4))
