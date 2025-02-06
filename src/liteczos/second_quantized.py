import numpy as np

def number(num_qbits):
    return np.diag([np.bitwise_count(i) for i in range(2**num_qbits)])