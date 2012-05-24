"""
Tests for dcmstack.dcmstack
"""
import sys
from os import path
from nose.tools import ok_, eq_, assert_raises
import numpy as np

test_dir = path.dirname(__file__)
src_dir = path.normpath(path.join(test_dir, '../src'))
sys.path.insert(0, src_dir)

import dcmstack

def test_key_regex_filter():
        filt = dcmstack.make_key_regex_filter(['test', 'another'], 
                                              ['2', 'another test'])
        ok_(filt('test', 1))
        ok_(filt('test another', 1))
        ok_(filt('another tes', 1))
        ok_(not filt('test2', 1))
        ok_(not filt('2 another', 1))
        ok_(not filt('another test', 1))
        
def test_closest_ortho_pat_axis():
    eq_(dcmstack.closest_ortho_pat_axis((0.9, 0.1, 0.1)), 'lr')
    eq_(dcmstack.closest_ortho_pat_axis((-0.9, 0.1, 0.1)), 'rl')
    eq_(dcmstack.closest_ortho_pat_axis((0.1, 0.9, 0.1)), 'pa')
    eq_(dcmstack.closest_ortho_pat_axis((0.1, -0.9, 0.1)), 'ap')
    eq_(dcmstack.closest_ortho_pat_axis((0.1, 0.1, 0.9)), 'is')
    eq_(dcmstack.closest_ortho_pat_axis((0.1, 0.1, -0.9)), 'si')
    
class TestReorderVoxels(object):
    def setUp(self):
        self.vox_array = np.arange(16).reshape((2, 2, 2, 2))
        self.affine = np.eye(4)
        
    def test_invalid_vox_order(self):
        assert_raises(ValueError, 
                      dcmstack.reorder_voxels,
                      self.vox_array, 
                      self.affine, 
                      'lra', 
                      )
        assert_raises(ValueError, 
                      dcmstack.reorder_voxels,
                      self.vox_array, 
                      self.affine, 
                      'rpil', 
                      )
        assert_raises(ValueError, 
                      dcmstack.reorder_voxels,
                      self.vox_array, 
                      self.affine, 
                      'lrz', 
                      )
        
    def test_invalid_vox_array(self):
        assert_raises(ValueError,
                      dcmstack.reorder_voxels,
                      np.eye(2), 
                      self.affine, 
                      'rpi', 
                     )
        
    def test_invalid_affine(self):
        assert_raises(ValueError,
                      dcmstack.reorder_voxels,
                      self.vox_array, 
                      np.eye(3), 
                      'rpi', 
                     )
                     
    def test_no_op(self):
        vox_order = [dcmstack.closest_ortho_pat_axis(self.affine[:3, idx])[0] 
                     for idx in range(3)
                    ]
        vox_order = ''.join(vox_order)                             
        vox_array, affine, perm = dcmstack.reorder_voxels(self.vox_array, 
                                                          self.affine, 
                                                          vox_order)
        ok_((vox_array == self.vox_array).all())
        ok_((affine == self.affine).all())
        eq_(perm, (0, 1, 2))
        eq_(np.may_share_memory(affine, self.affine), False)
        
def test_dcm_time_to_sec():
    eq_(dcmstack.dcm_time_to_sec('100235.123456'), 36155.123456)
    
