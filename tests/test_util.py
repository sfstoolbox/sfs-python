import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
import pytest
import sfs

cart_sph_data = [
    ((1, 1, 1), (np.pi / 4, np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, 1, 1), (np.arctan2(1, -1), np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((1, -1, 1), (-np.pi / 4, np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, -1, 1), (np.arctan2(-1, -1), np.arccos(1 / np.sqrt(3)), np.sqrt(3))),
    ((1, 1, -1), (np.pi / 4, np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, 1, -1), (np.arctan2(1, -1), np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
    ((1, -1, -1), (-np.pi / 4, np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
    ((-1, -1, -1), (np.arctan2(-1, -1),
                    np.arccos(-1 / np.sqrt(3)), np.sqrt(3))),
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
    (1, 0),
    (10, 10),
    (10 * 2, (10 + 3.010299956639813)),
    (10 * 10, (10 + 10)),
    (10 * 3, (10 + 4.771212547196624)),
    (10 * 4, (10 + 6.02059991327962)),
    (10 * 0.5, (10 - 3.01029995663981198)),
    (10 * 0.1, (10 - 10)),
    (10 * 0.25, (10 - 6.02059991327962396))
    ]


@pytest.mark.parametrize('linear, decibel', db_data)
def test_db_amplitude(linear, decibel):
    d = sfs.util.db(linear, True)
    assert_allclose(d, decibel)


@pytest.mark.parametrize('linear, decibel', db_data)
def test_db_power(linear, decibel):
    d = sfs.util.db(linear)
    assert_allclose(d, 2 * decibel)


image_sources_for_box_data = [(
    (([1, 1, 1], [5, 4, 3], 1)),
    ((((1,  1, -1),
       (1, -1,  1),
       (-1,  1,  1),
       (1,  1,  1),
       (9,  1,  1),
       (1,  7,  1),
       (1,  1,  5)), ((0, 0, 0, 0, 1, 0),
                      (0, 0, 1, 0, 0, 0),
                      (1, 0, 0, 0, 0, 0),
                      (0, 0, 0, 0, 0, 0),
                      (0, 1, 0, 0, 0, 0),
                      (0, 0, 0, 1, 0, 0),
                      (0, 0, 0, 0, 0, 1)))),
)]


@pytest.mark.parametrize('in_image_source, out_image_source',
                         image_sources_for_box_data)
def test_image_sources_for_box(in_image_source, out_image_source):
    X, L, N = in_image_source
    Xs, wall = out_image_source
    e = sfs.util.image_sources_for_box(X, L, N)
    img, wall_count = e
    assert_allclose(img, Xs)
    assert_array_equal(wall_count, wall)
