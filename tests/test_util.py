import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
import math
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
def testcart2sph(coord, polar):
    x, y, z = coord
    a = sfs.util.cart2sph(x, y, z)
    assert_allclose(a, polar)


@pytest.mark.parametrize('coord, polar', cart_sph_data)
def testsph2cart(coord, polar):
    alpha, beta, r = polar
    b = sfs.util.sph2cart(alpha, beta, r)
    assert_allclose(b, coord)


direction_vector_data = [
    ((np.pi/4, np.pi/4), (0.5, 0.5, np.sqrt(2)/2)),
    ((3*np.pi/4, 3*np.pi/4), (-1/2, 1/2, -np.sqrt(2)/2)),
    ((3*np.pi/4, -3*np.pi/4), (1/2, -1/2, -np.sqrt(2)/2)),
    ((np.pi/4, -np.pi/4), (-1/2, -1/2, np.sqrt(2)/2)),
    ((-np.pi/4, np.pi/4), (1/2, -1/2, np.sqrt(2)/2)),
    ((-3*np.pi/4, 3*np.pi/4), (-1/2, -1/2, -np.sqrt(2)/2)),
    ((-3*np.pi/4, -3*np.pi/4), (1/2, 1/2, -np.sqrt(2)/2)),
    ((-np.pi/4, -np.pi/4), (-1/2, 1/2, np.sqrt(2)/2)),
]


@pytest.mark.parametrize('input, vector', direction_vector_data)
def testdirection_vector(input, vector):
    alpha, beta = input
    c = sfs.util.direction_vector(alpha, beta)
    assert_allclose(c, vector)


db_data = [
    ((0), -math.inf),
    ((1), 0),
    ((10), 2*10),
    ((10*2), 2*(10+3)),
    ((10*10), 2*(10+10)),
    ((10*3), 2*(10+4.7)),
    ((10*4), 2*(10+6)),
    ((10*0.5), 2*(10-3)),
    ((10*0.1), 2*(10-10)),
    ((10*0.25), 2*(10-6)),
    ((0, True), -math.inf),
    ((1, True), 0),
    ((10, True), 10),
    ((10*2, True), 10+3),
    ((10*10, True), 10+10),
    ((10*3, True), 10+4.7),
    ((10*4, True), 10+6),
    ((10*0.5, True), 10-3),
    ((10*0.1, True), 10-10),
    ((10*0.25, True), 10-6),
    ]


@pytest.mark.parametrize('linear, decibel', db_data)
def testdb(linear, decibel):
    try:
        x, is_power = linear
    except:
        x = linear
        is_power = False

    d = sfs.util.db(x, is_power)
    assert_allclose(d, decibel, rtol=1e-2)


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


@pytest.mark.parametrize('in_image_source, out_image_source', image_sources_for_box_data)
def testimage_sources_for_box(in_image_source, out_image_source):
    X, L, N = in_image_source
    Xs, wall = out_image_source
    e = sfs.util.image_sources_for_box(X, L, N)
    img, wall_count = e
    assert_allclose(img, Xs)
    assert_array_equal(wall_count, wall)