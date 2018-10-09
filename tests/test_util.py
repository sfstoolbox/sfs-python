import numpy as np
from numpy.testing import assert_allclose
import pytest
import sfs

cart_sph_data = [
    ((1, 1, 1), (np.pi/4, np.arccos(1/np.sqrt(3)), np.sqrt(3))),
    ((-1, 1, 1), (np.arctan2(1, -1), np.arccos(1/np.sqrt(3)), np.sqrt(3))),
    ((1, -1, 1), (-np.pi/4, np.arccos(1/np.sqrt(3)), np.sqrt(3))),
    ((-1, -1, 1), (np.arctan2(-1, -1), np.arccos(1/np.sqrt(3)), np.sqrt(3))),
    ((1, 1, -1), (np.pi/4, np.arccos(-1/np.sqrt(3)), np.sqrt(3))),
    ((-1, 1, -1), (np.arctan2(1, -1), np.arccos(-1/np.sqrt(3)), np.sqrt(3))),
    ((1, -1, -1), (-np.pi/4, np.arccos(-1/np.sqrt(3)), np.sqrt(3))),
    ((-1, -1, -1), (np.arctan2(-1, -1), np.arccos(-1/np.sqrt(3)), np.sqrt(3))),
]


@pytest.mark.parametrize('coord, polar', cart_sph_data)
def test_cart2sph(coord, polar):
    x, y, z = coord
    a = sfs.util.cart2sph(x, y, z)
    assert_allclose(a, polar)


@pytest.mark.parametrize('coord, polar', cart_sph_data)
def test_sph2cart(coord, polar):
    alpha, beta, r = polar
    b = sfs.util.sph2cart(alpha, beta, r)
    assert_allclose(b, coord)
