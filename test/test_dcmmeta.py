"""
Tests for dcmstack.dcmmeta
"""
import sys
from os import path
from glob import glob
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
        
    def test_multiple_classes(self):
        self.ext.get_class_dict(('global', 'const'))['Test'] = 0 
        self.ext.get_class_dict(('time', 'samples'))['Test'] = [0] * 3
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
def test_dcmmeta_affine():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2), 
                                              np.diag([1, 2, 3, 4]), 
                                              2
                                             )
    ok_(np.allclose(ext.affine, np.diag([1, 2, 3, 4])))
    assert_raises(ValueError, 
                  setattr, 
                  ext, 
                  'affine', 
                  np.eye(3)
                 )
    ext.affine = np.eye(4)
    ok_(np.allclose(ext.affine, np.eye(4)))
    
def test_dcmmeta_slice_dim():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2), np.eye(4), None)
    eq_(ext.slice_dim, None)
    assert_raises(ValueError, 
                  setattr, 
                  ext, 
                  'slice_dim', 
                  3
                 )
    ext.slice_dim = 2
    eq_(ext.slice_dim, 2)
    
def test_dcmmeta_shape():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2), np.eye(4), None)
    eq_(ext.shape, (64, 64, 2))
    assert_raises(ValueError, 
                  setattr, 
                  ext, 
                  'shape', 
                  (64, 64)
                 )
    ext.shape = (128, 128, 64)
    eq_(ext.shape, (128, 128, 64))
    
def test_dcmmeta_version():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2), np.eye(4), None)
    eq_(ext.version, dcmmeta._meta_version)
    ext.version = 1.0
    eq_(ext.version, 1.0)
    
def test_dcmmeta_slice_norm():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2), np.eye(4), 2)
    ok_(np.allclose(ext.slice_normal, [0, 0, 1]))
    ext.slice_dim = 1
    ok_(np.allclose(ext.slice_normal, [0, 1, 0]))
    
def test_dcmmeta_n_slices():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2), np.eye(4), 2)
    eq_(ext.n_slices, 2)
    ext.slice_dim = 1
    eq_(ext.n_slices, 64)
    
class TestGetKeysClassesValues(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2, 3, 4), 
                                                       np.eye(4), 
                                                       2
                                                      )
        self.keys = []
        for classes in self.ext.get_valid_classes():
            key = '%s_%s_test' % classes
            self.keys.append(key)
            self.ext.get_class_dict(classes)[key] = \
                ([0] * self.ext.get_multiplicity(classes))
                
    
    def test_get_keys(self):
        eq_(set(self.keys), set(self.ext.get_keys()))
        
    def test_get_classification(self):
        eq_(self.ext.get_classification('foo'), None)
        for classes in self.ext.get_valid_classes():
            key = '%s_%s_test' % classes
            eq_(self.ext.get_classification(key), classes)
            
    def test_get_class_dict(self):
        for classes in self.ext.get_valid_classes():
            key = '%s_%s_test' % classes
            ok_(key in self.ext.get_class_dict(classes))
            
    def test_get_values(self):
        eq_(self.ext.get_values('foo'), None)
        for classes in self.ext.get_valid_classes():
            key = '%s_%s_test' % classes
            eq_(self.ext.get_values(key), 
                [0] * self.ext.get_multiplicity(classes)
               )
    
    def test_get_vals_and_class(self):
        eq_(self.ext.get_values_and_class('foo'), (None, None))
        for classes in self.ext.get_valid_classes():
            key = '%s_%s_test' % classes
            eq_(self.ext.get_values_and_class(key), 
                ([0] * self.ext.get_multiplicity(classes), classes)
               )
