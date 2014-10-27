import numpy as np
import sfs
import pytest


linear_cases = [
    ((3, 1), [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
    ((2, 1), [[-0.5, 0, 0], [0.5, 0, 0]]),
    ((1, 1), [[0, 0, 0]]),
    ((3, 0.5), [[-0.5, 0, 0], [0, 0, 0], [0.5, 0, 0]]),
    ((2, 1, [0.5, 2, 3]), [[0, 2, 3], [1, 2, 3]]),
    ((2, 1, np.array([0.5, 2, 3])), [[0, 2, 3], [1, 2, 3]]),
    ((2, 1, np.array([[0.5, 2, 3]])), [[0, 2, 3], [1, 2, 3]]),
    ((2, 1, np.array([[0.5, 2, 3]]).transpose()), [[0, 2, 3], [1, 2, 3]]),
    ((2, 1, np.matrix([[0.5, 2, 3]])), [[0, 2, 3], [1, 2, 3]]),
    ((2, 1, np.matrix([[0.5, 2, 3]]).transpose()), [[0, 2, 3], [1, 2, 3]]),
]


@pytest.mark.parametrize("args, result", linear_cases)
def test_linear(args, result):
    a = sfs.array.linear(*args)
    assert a.dtype == np.float64
    assert np.all(a == result)


def test_linear_named_args():
    a = sfs.array.linear(N=2, dx=0.5, center=[0.25, 1, 2])
    assert np.all(a == [[0, 1, 2], [0.5, 1, 2]])
