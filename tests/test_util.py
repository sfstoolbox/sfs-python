import numpy as np
from numpy.testing import assert_allclose
import pytest
import sfs


cart_sph_data = [
    ((1, 1, 1), (np.pi / 4, np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, 1, 1), (3 / 4 * np.pi, np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((1, -1, 1), (-np.pi / 4, np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, -1, 1), (-3 / 4 * np.pi, np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((1, 1, -1), (np.pi / 4, np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, 1, -1), (3 / 4 * np.pi, np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
    ((1, -1, -1), (-np.pi / 4, np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, -1, -1), (-3 / 4 * np.pi, np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
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


direction_vector_data = [
    ((np.pi / 4, np.pi / 4), (0.5, 0.5, np.sqrt(2) / 2)),
    ((3 * np.pi / 4, 3 * np.pi / 4), (-1 / 2, 1 / 2, -np.sqrt(2) / 2)),
    ((3 * np.pi / 4, -3 * np.pi / 4), (1 / 2, -1 / 2, -np.sqrt(2) / 2)),
    ((np.pi / 4, -np.pi / 4), (-1 / 2, -1 / 2, np.sqrt(2) / 2)),
    ((-np.pi / 4, np.pi / 4), (1 / 2, -1 / 2, np.sqrt(2) / 2)),
    ((-3 * np.pi / 4, 3 * np.pi / 4), (-1 / 2, -1 / 2, -np.sqrt(2) / 2)),
    ((-3 * np.pi / 4, -3 * np.pi / 4), (1 / 2, 1 / 2, -np.sqrt(2) / 2)),
    ((-np.pi / 4, -np.pi / 4), (-1 / 2, 1 / 2, np.sqrt(2) / 2)),
]


@pytest.mark.parametrize('input, vector', direction_vector_data)
def test_direction_vector(input, vector):
    alpha, beta = input
    c = sfs.util.direction_vector(alpha, beta)
    assert_allclose(c, vector)


db_data = [
    (0, -np.inf),
    (0.5, -3.01029995663981),
    (1, 0),
    (2, 3.01029995663981),
    (10, 10),
]


@pytest.mark.parametrize('linear, power_db', db_data)
def test_db_amplitude(linear, power_db):
    d = sfs.util.db(linear)
    assert_allclose(d, power_db * 2)


@pytest.mark.parametrize('linear, power_db', db_data)
def test_db_power(linear, power_db):
    d = sfs.util.db(linear, power=True)
    assert_allclose(d, power_db)
