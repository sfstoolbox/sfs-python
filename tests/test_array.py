import numpy as np
from numpy.testing import assert_array_equal
import pytest
import sfs


def vectortypes(*coeffs):
    return [
        list(coeffs),
        tuple(coeffs),
        np.array(coeffs),
        np.array(coeffs).reshape(1, -1),
        np.array(coeffs).reshape(-1, 1),
    ]


def vector_id(vector):
    if isinstance(vector, np.ndarray):
        return 'array, shape=' + repr(vector.shape)
    return type(vector).__name__


@pytest.mark.parametrize('N, spacing, result', [
    (2, 1, sfs.array.SecondarySourceDistribution(
        x=[[0, -0.5, 0], [0, 0.5, 0]],
        n=[[1, 0, 0], [1, 0, 0]],
        a=[1, 1],
    )),
    (3, 1, sfs.array.SecondarySourceDistribution(
        x=[[0, -1, 0], [0, 0, 0], [0, 1, 0]],
        n=[[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        a=[1, 1, 1],
    )),
    (3, 0.5, sfs.array.SecondarySourceDistribution(
        x=[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]],
        n=[[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        a=[0.5, 0.5, 0.5],
    )),
])
def test_linear_with_defaults(N, spacing, result):
    a = sfs.array.linear(N, spacing)
    assert a.x.dtype == np.float64
    assert a.n.dtype == np.float64
    assert a.a.dtype == np.float64
    assert_array_equal(a.x, result.x)
    assert_array_equal(a.n, result.n)
    assert_array_equal(a.a, result.a)


def test_linear_with_named_arguments():
    a = sfs.array.linear(N=2, spacing=0.5)
    assert_array_equal(a.x, [[0, -0.25, 0], [0, 0.25, 0]])
    assert_array_equal(a.n, [[1, 0, 0], [1, 0, 0]])
    assert_array_equal(a.a, [0.5, 0.5])


@pytest.mark.parametrize('center', vectortypes(-1, 0.5, 2), ids=vector_id)
def test_linear_with_center(center):
    a = sfs.array.linear(2, 1, center=center)
    assert_array_equal(a.x, [[-1, 0, 2], [-1, 1, 2]])
    assert_array_equal(a.n, [[1, 0, 0], [1, 0, 0]])
    assert_array_equal(a.a, [1, 1])


@pytest.mark.parametrize('orientation', vectortypes(0, -1, 0), ids=vector_id)
def test_linear_with_center_and_orientation(orientation):
    a = sfs.array.linear(2, 1, center=[0, 1, 2], orientation=orientation)
    assert_array_equal(a.x, [[-0.5, 1, 2], [0.5, 1, 2]])
