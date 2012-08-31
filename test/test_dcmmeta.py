"""
Tests for dcmstack.dcmmeta
"""
import sys
from os import path
from glob import glob
from hashlib import sha256
from nose.tools import ok_, eq_, assert_raises
import numpy as np
import dicom
import nibabel as nb

test_dir = path.dirname(__file__)
src_dir = path.normpath(path.join(test_dir, '../src'))
sys.path.insert(0, src_dir)

from dcmstack import dcmmeta

def test_is_constant():
    ok_(dcmmeta.is_constant([0]))
    ok_(dcmmeta.is_constant([0, 0]))
    ok_(dcmmeta.is_constant([0, 0, 1, 1], period=2))
    ok_(dcmmeta.is_constant([0, 0, 1, 1, 2, 2], period=2))
    eq_(dcmmeta.is_constant([0, 1]), False)
    eq_(dcmmeta.is_constant([0, 0, 1, 2], 2), False)
    assert_raises(ValueError, dcmmeta.is_constant, [0, 0, 0], -1)
    assert_raises(ValueError, dcmmeta.is_constant, [0, 0, 0], 1)
    assert_raises(ValueError, dcmmeta.is_constant, [0, 0, 0], 2)
    assert_raises(ValueError, dcmmeta.is_constant, [0, 0, 0], 4)
    
def test_is_repeating():
    ok_(dcmmeta.is_repeating([0, 1, 0, 1], 2))
    ok_(dcmmeta.is_repeating([0, 1, 0, 1, 0, 1], 2))
    eq_(dcmmeta.is_repeating([0, 1, 1, 2], 2), False)
    assert_raises(ValueError, dcmmeta.is_repeating, [0, 1, 0, 1], -1)
    assert_raises(ValueError, dcmmeta.is_repeating, [0, 1, 0, 1], 1)
    assert_raises(ValueError, dcmmeta.is_repeating, [0, 1, 0, 1], 3)
    assert_raises(ValueError, dcmmeta.is_repeating, [0, 1, 0, 1], 4)
    assert_raises(ValueError, dcmmeta.is_repeating, [0, 1, 0, 1], 5)
    