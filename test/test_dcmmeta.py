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
    
def test_get_valid_classes():
    ext = dcmmeta.DcmMetaExtension.make_empty((2, 2, 2), np.eye(4))
    eq_(ext.get_valid_classes(), (('global', 'const'), ('global', 'slices')))
    
    ext.shape = (2, 2, 2, 2)
    eq_(ext.get_valid_classes(), 
        (('global', 'const'), 
         ('global', 'slices'),
         ('time', 'samples'),
         ('time', 'slices')
        )
       )
       
    ext.shape = (2, 2, 2, 1, 2)
    eq_(ext.get_valid_classes(), 
        (('global', 'const'), 
         ('global', 'slices'),
         ('vector', 'samples'),
         ('vector', 'slices')
        )
       )
 
    ext.shape = (2, 2, 2, 2, 2)
    eq_(ext.get_valid_classes(), 
        (('global', 'const'), 
         ('global', 'slices'),
         ('time', 'samples'),
         ('time', 'slices'),
         ('vector', 'samples'),
         ('vector', 'slices')
        )
       )

class TestCheckValid(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2, 3, 4), 
                                                       np.eye(4), 
                                                       2)
    
    def test_empty(self):
        self.ext.check_valid()
                                               
    def test_const(self):
        self.ext.get_class_dict(('global', 'const'))['ConstTest'] = 2
        self.ext.check_valid()
        del self.ext._content['global']['const']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_global_slices(self):
        cls_dict = self.ext.get_class_dict(('global', 'slices'))
        cls_dict['SliceTest'] = [0] * 23
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['SliceTest'] = [0] * 24
        self.ext.check_valid()
        del self.ext._content['global']['slices']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_time_samples(self):
        cls_dict = self.ext.get_class_dict(('time', 'samples'))
        cls_dict['TimeSampleTest'] = [0] * 2
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['TimeSampleTest'] = [0] * 3
        self.ext.check_valid()
        del self.ext._content['time']['samples']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_time_slices(self):
        cls_dict = self.ext.get_class_dict(('time', 'slices'))
        cls_dict['TimeSliceTest'] = [0] 
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['TimeSliceTest'] = [0] * 2
        self.ext.check_valid()
        del self.ext._content['time']['slices']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_vector_samples(self):
        cls_dict = self.ext.get_class_dict(('vector', 'samples'))
        cls_dict['VectorSampleTest'] = [0] * 3
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['VectorSampleTest'] = [0] * 4
        self.ext.check_valid()
        del self.ext._content['vector']['samples']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_vector_slices(self):
        cls_dict = self.ext.get_class_dict(('vector', 'slices'))
        cls_dict['VectorSliceTest'] = [0] * 5
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['VectorSliceTest'] = [0] * 6
        self.ext.check_valid()
        del self.ext._content['vector']['slices']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_invalid_affine(self):
        self.ext._content['dcmmeta_affine'] = np.eye(3).tolist()
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_invalid_slice_dim(self):
        self.ext._content['dcmmeta_slice_dim'] = 3
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        self.ext._content['dcmmeta_slice_dim'] = -1
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_invalid_shape(self):
        self.ext._content['dcmmeta_shape'] = [2, 2]
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        self.ext._content['dcmmeta_shape'] = [2, 2, 1, 1, 1, 2]
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)