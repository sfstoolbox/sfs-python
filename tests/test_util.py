import numpy as np
from numpy.testing import assert_equal
import pytest
import sfs

def test_cart2sph_octant_1():
    a = sfs.util.cart2sph(1,1,1)
    alpha = a[0]  
    beta = a[1]
    r = a[2]   
    assert_equal(r, np.sqrt(3))
    assert_equal(alpha, np.pi / 4)
    assert_equal(beta, np.arccos(1/np.sqrt(3)))
    
def test_cart2sph_octant_2():
    a = sfs.util.cart2sph(-1,1,1)
    alpha = a[0]  
    beta = a[1]
    r = a[2]
    assert_equal(r, np.sqrt(3))
    assert_equal(alpha, np.arctan2(1,-1))
    assert_equal(beta, np.arccos(1/np.sqrt(3)))
    
def test_cart2sph_octant_3():
    a = sfs.util.cart2sph(1,-1,1)
    alpha = a[0]  
    beta = a[1]
    r = a[2] 
    assert_equal(r, np.sqrt(3))
    assert_equal(alpha, -np.pi / 4)
    assert_equal(beta, np.arccos(1/np.sqrt(3)))

def test_cart2sph_octant_4():
    a = sfs.util.cart2sph(-1,-1,1)
    alpha = a[0]  
    beta = a[1]
    r = a[2] 
    assert_equal(r, np.sqrt(3))
    assert_equal(alpha, np.arctan2(-1,-1))
    assert_equal(beta, np.arccos(1/np.sqrt(3)))

def test_cart2sph_octant_5():
    a = sfs.util.cart2sph(1,1,-1)
    alpha = a[0]  
    beta = a[1]
    r = a[2] 
    assert_equal(r, np.sqrt(3))
    assert_equal(alpha, np.pi / 4)
    assert_equal(beta, np.arccos(-1/np.sqrt(3)))

def test_cart2sph_octant_6():
    a = sfs.util.cart2sph(-1,1,-1)
    alpha = a[0]  
    beta = a[1]
    r = a[2] 
    assert_equal(r, np.sqrt(3))
    assert_equal(alpha, np.arctan2(1,-1))
    assert_equal(beta, np.arccos(-1/np.sqrt(3)))

def test_cart2sph_octant_7():
    a = sfs.util.cart2sph(1,-1,-1)
    alpha = a[0]  
    beta = a[1]
    r = a[2] 
    assert_equal(r, np.sqrt(3))
    assert_equal(alpha, -np.pi / 4)
    assert_equal(beta, np.arccos(-1/np.sqrt(3)))

def test_cart2sph_octant_8():
    a = sfs.util.cart2sph(-1,-1,-1)
    alpha = a[0]  
    beta = a[1]
    r = a[2] 
    assert_equal(r, np.sqrt(3))
    assert_equal(alpha, np.arctan2(-1,-1))
    assert_equal(beta,np.arccos(-1/np.sqrt(3)))

def test_sph2cart_octant_1():
    a = sfs.util.sph2cart(np.pi / 4, np.arccos(1/np.sqrt(3)), np.sqrt(3))
    x = a[0]
    y = a[1]
    z = a[2]
    assert_equal(x, 1)
    assert_equal(y, 1)
    assert_equal(z, 1)
    
def test_sph2cart_octant_2():
    a = sfs.util.sph2cart(np.arctan2(1,-1), np.arccos(1/np.sqrt(3)), np.sqrt(3))
    x = round(a[0],10)
    y = a[1]
    z = a[2]
    assert_equal(x, -1 )
    assert_equal(y, 1)
    assert_equal(z, 1)

def test_sph2cart_octant_3():
    a = sfs.util.sph2cart(-np.pi / 4, np.arccos(1/np.sqrt(3)), np.sqrt(3))
    x = a[0]
    y = a[1]
    z = a[2]
    assert_equal(x, 1)
    assert_equal(y, -1)
    assert_equal(z, 1)

def test_sph2cart_octant_4():
    a = sfs.util.sph2cart(np.arctan2(-1,-1), np.arccos(1/np.sqrt(3)), np.sqrt(3))
    x = round(a[0],10)
    y = a[1]
    z = a[2]
    assert_equal(x, -1)
    assert_equal(y, -1)
    assert_equal(z, 1)

def test_sph2cart_octant_5():
    a = sfs.util.sph2cart(np.arctan2(1,1), np.arccos(-1/np.sqrt(3)), np.sqrt(3))
    x = round(a[0],10)
    y = round(a[1],10)
    z = round(a[2],10)
    assert_equal(x, 1)
    assert_equal(y, 1)
    assert_equal(z, -1)

def test_sph2cart_octant_6():
    a = sfs.util.sph2cart(np.arctan2(1,-1), np.arccos(-1/np.sqrt(3)), np.sqrt(3))
    x = round(a[0],10)
    y = round(a[1],10)
    z = round(a[2],10)
    assert_equal(x, -1)
    assert_equal(y, 1)
    assert_equal(z, -1)

def test_sph2cart_octant_7():
    a = sfs.util.sph2cart(np.arctan2(-1,1), np.arccos(-1/np.sqrt(3)), np.sqrt(3))
    x = round(a[0],10)
    y = round(a[1],10)
    z = round(a[2],10)
    assert_equal(x, 1)
    assert_equal(y, -1)
    assert_equal(z, -1)

def test_sph2cart_octant_8():
    a = sfs.util.sph2cart(np.arctan2(-1,-1), np.arccos(-1/np.sqrt(3)), np.sqrt(3))
    x = round(a[0],10)
    y = round(a[1],10)
    z = round(a[2],10)
    assert_equal(x, -1)
    assert_equal(y, -1)
    assert_equal(z, -1)