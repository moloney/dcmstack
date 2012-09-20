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
       
def test_get_mulitplicity_4d():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 7, 11), np.eye(4), 2)
    eq_(ext.get_multiplicity(('global', 'const')), 1)
    eq_(ext.get_multiplicity(('global', 'slices')), 7 * 11)
    eq_(ext.get_multiplicity(('time', 'samples')), 11)
    eq_(ext.get_multiplicity(('time', 'slices')), 7)
    
def test_get_mulitplicity_4d_vec():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 7, 1, 11), np.eye(4), 2)
    eq_(ext.get_multiplicity(('global', 'const')), 1)
    eq_(ext.get_multiplicity(('global', 'slices')), 7 * 11)
    eq_(ext.get_multiplicity(('vector', 'samples')), 11)
    eq_(ext.get_multiplicity(('vector', 'slices')), 7)

def test_get_mulitplicity_5d():
    ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 7, 11, 13), 
                                              np.eye(4), 
                                              2
                                             )
    eq_(ext.get_multiplicity(('global', 'const')), 1)
    eq_(ext.get_multiplicity(('global', 'slices')), 7 * 11 * 13)
    eq_(ext.get_multiplicity(('time', 'samples')), 11 * 13)
    eq_(ext.get_multiplicity(('time', 'slices')), 7)
    eq_(ext.get_multiplicity(('vector', 'samples')), 13)
    eq_(ext.get_multiplicity(('vector', 'slices')), 7 * 11)

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
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['TimeSampleTest'] = [0] * 12
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

class TestFiltering(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 2, 3, 4), 
                                                       np.eye(4), 
                                                       2
                                                      )
        
        for classes in self.ext.get_valid_classes():
            prefix = '%s_%s_test_' % classes
            self.ext.get_class_dict(classes)[prefix + 'foo'] = \
                    ([0] * self.ext.get_multiplicity(classes))
            self.ext.get_class_dict(classes)[prefix + 'foobaz'] = \
                    ([0] * self.ext.get_multiplicity(classes))
    
    def test_filter_all(self):
        self.ext.filter_meta(lambda key, val: 'foo' in key)
        eq_(len(self.ext.get_keys()), 0)
        
    def test_filter_some(self):
        self.ext.filter_meta(lambda key, val: key.endswith('baz'))
        keys = self.ext.get_keys()
        for classes in self.ext.get_valid_classes():
            prefix = '%s_%s_test_' % classes
            ok_(prefix + 'foo' in keys)
            ok_(not prefix + 'foobaz' in keys)
            
    def test_clear_slices(self):
        self.ext.clear_slice_meta()
        for base_cls, sub_cls in self.ext.get_valid_classes():
            if sub_cls == 'slices':
                eq_(len(self.ext.get_class_dict((base_cls, sub_cls))), 0)
    
class TestSimplify(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 3, 5, 7), 
                                                       np.eye(4),
                                                       2
                                                      )
    
    def test_simplify_global_slices(self):
        glob_slc = self.ext.get_class_dict(('global', 'slices'))
        glob_slc['Test1'] = [0] * (3 * 5 * 7)
        glob_slc['Test2'] = []
        for idx in xrange(7):
            glob_slc['Test2'] += [idx] * (3 * 5)
        glob_slc['Test3'] = []
        for idx in xrange(5 * 7):
            glob_slc['Test3'] += [idx] * (3)
        glob_slc['Test4'] = []
        for idx in xrange(7):
            glob_slc['Test4'] += [idx2 for idx2 in xrange(3*5)]
        glob_slc['Test5'] = []
        for idx in xrange(7 * 5):
            glob_slc['Test5'] += [idx2 for idx2 in xrange(3)]
        self.ext.check_valid()
            
        eq_(self.ext._simplify('Test1'), True)
        eq_(self.ext.get_classification('Test1'), ('global', 'const'))
        
        eq_(self.ext._simplify('Test2'), True)
        eq_(self.ext.get_classification('Test2'), ('vector', 'samples'))
        
        eq_(self.ext._simplify('Test3'), True)
        eq_(self.ext.get_classification('Test3'), ('time', 'samples'))
        
        eq_(self.ext._simplify('Test4'), True)
        eq_(self.ext.get_classification('Test4'), ('vector', 'slices'))
        
        eq_(self.ext._simplify('Test5'), True)
        eq_(self.ext.get_classification('Test5'), ('time', 'slices'))
    
    def test_simplify_vector_slices(self):
        vec_slc = self.ext.get_class_dict(('vector', 'slices'))
        vec_slc['Test1'] = [0] * (3 * 5)
        vec_slc['Test2'] = []
        for time_idx in xrange(5):
            vec_slc['Test2'] += [time_idx] * 3
        vec_slc['Test3'] = []
        for time_idx in xrange(5):
            for slc_idx in xrange(3):
                vec_slc['Test3'] += [slc_idx]
        self.ext.check_valid()
        
        eq_(self.ext._simplify('Test1'), True)
        eq_(self.ext.get_classification('Test1'), ('global', 'const'))
        
        eq_(self.ext._simplify('Test2'), True)
        eq_(self.ext.get_classification('Test2'), ('time', 'samples'))
        
        eq_(self.ext._simplify('Test3'), True)
        eq_(self.ext.get_classification('Test3'), ('time', 'slices'))
        
    def test_simplify_time_slices(self):
        time_slc = self.ext.get_class_dict(('time', 'slices'))
        time_slc['Test1'] = [0] * 3
        self.ext.check_valid()
        
        eq_(self.ext._simplify('Test1'), True)
        eq_(self.ext.get_classification('Test1'), ('global', 'const'))
        
    def test_simplify_time_samples(self):
        time_smp = self.ext.get_class_dict(('time', 'samples'))
        time_smp['Test1'] = [0] * (5 * 7)
        time_smp['Test2'] = []
        for vec_idx in xrange(7):
            time_smp['Test2'] += [vec_idx] * 5
        self.ext.check_valid()
        
        eq_(self.ext._simplify('Test1'), True)
        eq_(self.ext.get_values_and_class('Test1'), 
            (0, ('global', 'const')))
        
        eq_(self.ext._simplify('Test2'), True)
        eq_(self.ext.get_classification('Test2'), ('vector', 'samples'))
        
    def test_simplify_vector_samples(self):
        vector_smp = self.ext.get_class_dict(('vector', 'samples'))
        vector_smp['Test1'] = [0] * 7
        self.ext.check_valid()
        
        eq_(self.ext._simplify('Test1'), True)
        eq_(self.ext.get_classification('Test1'), ('global', 'const'))
    
class TestGetSubset(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMetaExtension.make_empty((64, 64, 3, 5, 7), 
                                                       np.eye(4), 
                                                       2
                                                      )
                                                      
        #Add an element to every classification
        for classes in self.ext.get_valid_classes():
            key = '%s_%s_test' % classes
            mult = self.ext.get_multiplicity(classes)
            self.ext.get_class_dict(classes)[key] = range(mult)
        
    def test_slice_subset(self):
        for slc_idx in xrange(self.ext.n_slices):
            sub = self.ext.get_subset(2, slc_idx)
            sub.check_valid()
            
            for classes in self.ext.get_valid_classes():
                key = '%s_%s_test' % classes
                if classes == ('time', 'slices'):
                    eq_(sub.get_values_and_class(key), 
                        (slc_idx, ('global', 'const'))
                       )
                elif classes == ('vector', 'slices'):
                    eq_(sub.get_values_and_class(key), 
                        ((range(slc_idx, (3 * 5), 3) * 7), ('time', 'samples'))
                       )
                elif classes == ('global', 'slices'):
                    eq_(sub.get_values_and_class(key), 
                        (range(slc_idx, (3 * 5 * 7), 3), ('time', 'samples')))
                else:
                    eq_(sub.get_values_and_class(key), 
                        self.ext.get_values_and_class(key)
                       )
                       
    def test_slice_subset_simplify(self):
        vals = []
        for time_idx in xrange(5):
            for slice_idx in xrange(3):
                if slice_idx == 1:
                    vals.append(1)
                else:
                    vals.append(time_idx)
        self.ext.get_class_dict(('vector', 'slices'))['vec_slc_to_const'] = vals
        
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                for slice_idx in xrange(3):
                    if slice_idx == 1:
                        vals.append(1)
                    else:
                        vals.append(vector_idx)
        self.ext.get_class_dict(('global', 'slices'))['glb_slc_to_const'] = \
            vals
        self.ext.check_valid()
        
        sub = self.ext.get_subset(2, 1)
        sub.check_valid()
        eq_(sub.get_values_and_class('vec_slc_to_const'),
            (1, ('global', 'const')))
        eq_(sub.get_values_and_class('glb_slc_to_const'),
            (1, ('global', 'const')))
                       
    def test_time_sample_subset(self):
        for time_idx in xrange(5):
            sub = self.ext.get_subset(3, time_idx)
            sub.check_valid()
            for classes in self.ext.get_valid_classes():
                key = '%s_%s_test' % classes
                if classes[0] == 'time':
                    ok_(not classes in sub.get_valid_classes())
                    if classes[1] == 'samples':
                        eq_(sub.get_values_and_class(key),
                            (range(time_idx, 5 * 7, 5), ('vector', 'samples'))
                           )
                    elif classes[1] == 'slices':
                        eq_(sub.get_values_and_class(key), 
                            (self.ext.get_values(key), ('vector', 'slices')))
                elif classes[0] == 'vector':
                    if classes[1] == 'samples':
                        eq_(sub.get_values_and_class(key),
                            self.ext.get_values_and_class(key)
                           )
                    elif classes[1] == 'slices':
                        start = time_idx * 3
                        end = start + 3
                        eq_(sub.get_values_and_class(key), 
                            (range(start, end), ('vector', 'slices')))
                else:
                    if classes[1] == 'const':
                        eq_(sub.get_values_and_class(key),
                            self.ext.get_values_and_class(key)
                           )
                    elif classes[1] == 'slices':
                        vals = []
                        for vec_idx in xrange(7):
                            start = (vec_idx * (3 * 5)) + (time_idx * 3)
                            end = start + 3
                            vals += range(start, end)
                        eq_(sub.get_values_and_class(key), 
                            (vals, ('global', 'slices')))
            
    def test_time_sample_subset_simplify(self):
        #Test for simplification of time samples that become constant
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                vals.append(time_idx)
        self.ext.get_class_dict(('time', 'samples'))['time_smp_to_const'] = \
            vals
            
        for time_idx in xrange(5):
            sub = self.ext.get_subset(3, time_idx)
            sub.check_valid()
            eq_(sub.get_values_and_class('time_smp_to_const'),
                (time_idx, ('global', 'const')))
        
        #Test for simplification of vector slices that become constant
        vals = []
        for time_idx in xrange(5):
            for slice_idx in xrange(3):
                if time_idx == 1:
                    vals.append(1)
                else:
                    vals.append(slice_idx)
        self.ext.get_class_dict(('vector', 'slices'))['vec_slc_to_const'] = \
            vals
        
        #Test simplification of global slices that become constant
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                for slice_idx in xrange(3):
                    if time_idx == 1:
                        vals.append(1)
                    else:
                        vals.append(vector_idx)
        self.ext.get_class_dict(('global', 'slices'))['glb_slc_to_const'] = \
            vals
            
        sub = self.ext.get_subset(3, 1)
        sub.check_valid()
        eq_(sub.get_values_and_class('vec_slc_to_const'),
            (1, ('global', 'const')))
        eq_(sub.get_values_and_class('glb_slc_to_const'),
            (1, ('global', 'const')))
        
        #Test simplification of global slices that become vector slices or 
        #samples
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                for slice_idx in xrange(3):
                    if time_idx == 1:
                        vals.append(slice_idx)
                    else:
                        vals.append(vector_idx)
        self.ext.get_class_dict(('global', 'slices'))['glb_slc_to_vec'] = \
            vals

        for time_idx in xrange(5):
            sub = self.ext.get_subset(3, time_idx)
            sub.check_valid()
            if time_idx == 1:
                eq_(sub.get_values_and_class('glb_slc_to_vec'),
                    (range(3), ('vector', 'slices')))
            else:
                eq_(sub.get_values_and_class('glb_slc_to_vec'),
                    (range(7), ('vector', 'samples')))            
    
    def test_vector_sample_subset(self):
        for vector_idx in xrange(7):
            sub = self.ext.get_subset(4, vector_idx)
            sub.check_valid()
            for classes in self.ext.get_valid_classes():
                key = '%s_%s_test' % classes
                if classes[0] == 'vector':
                    ok_(not classes in sub.get_valid_classes())
                    if classes[1] == 'samples':
                        eq_(sub.get_values_and_class(key),
                            (vector_idx, ('global', 'const'))
                           )
                    elif classes[1] == 'slices':
                        eq_(sub.get_values_and_class(key), 
                            (range(3 * 5), ('global', 'slices'))
                           )
                elif classes[0] == 'time':
                    if classes[1] == 'samples': #Could be const
                        start = vector_idx * 5
                        end = start + 5
                        eq_(sub.get_values_and_class(key),
                            (range(start, end), classes)
                           )
                    elif classes[1] == 'slices':
                        eq_(sub.get_values_and_class(key),
                            self.ext.get_values_and_class(key))
                else:
                    if classes[1] == 'const':
                        eq_(sub.get_values_and_class(key),
                            self.ext.get_values_and_class(key)
                           )
                    elif classes[1] == 'slices': #Could be const or time samples or time slices
                        start = vector_idx * (3 * 5)
                        end = start + (3 * 5)
                        eq_(sub.get_values_and_class(key),
                            (range(start, end), classes))
                            
    def test_vector_sample_subset_simplify(self):
        
        #Test for simplification of time samples that become constant
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                if vector_idx == 1:
                    vals.append(1)
                else:
                    vals.append(time_idx)
        self.ext.get_class_dict(('time', 'samples'))['time_smp_to_const'] = \
            vals
        sub = self.ext.get_subset(4, 1)
        eq_(sub.get_values_and_class('time_smp_to_const'),
            (1, ('global', 'const')))
            
        #Test for simplification of global slices that become constant, time 
        #samples, or time slices
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                for slice_idx in xrange(3):
                    if vector_idx == 1:
                        vals.append(1)
                    elif vector_idx == 2:
                        vals.append(slice_idx)
                    else:
                        vals.append(time_idx)
        self.ext.get_class_dict(('global', 'slices'))['glb_slc'] = \
            vals
                
        for vector_idx in xrange(7):
            sub = self.ext.get_subset(4, vector_idx)
            sub.check_valid()
            if vector_idx == 1:
                eq_(sub.get_values_and_class('glb_slc'),
                    (1, ('global', 'const')))
            elif vector_idx == 2:
                eq_(sub.get_values_and_class('glb_slc'),
                    (range(3), ('time', 'slices')))
            else:
                eq_(sub.get_values_and_class('glb_slc'),
                    (range(5), ('time', 'samples')))

def test_from_sequence_2d_to_3d():
    ext1 = dcmmeta.DcmMetaExtension.make_empty((2, 2, 1), np.eye(4), 2)
    ext1.get_class_dict(('global', 'const'))['const'] = 1
    ext1.get_class_dict(('global', 'const'))['var'] = 1
    ext1.get_class_dict(('global', 'const'))['missing'] = 1
    ext2 = dcmmeta.DcmMetaExtension.make_empty((2, 2, 1), np.eye(4), 2)
    ext2.get_class_dict(('global', 'const'))['const'] = 1
    ext2.get_class_dict(('global', 'const'))['var'] = 2
    
    merged = dcmmeta.DcmMetaExtension.from_sequence([ext1, ext2], 2)
    eq_(merged.get_values_and_class('const'),
        (1, ('global', 'const')))
    eq_(merged.get_values_and_class('var'),
        ([1, 2], ('global', 'slices')))
    eq_(merged.get_values_and_class('missing'),
        ([1, None], ('global', 'slices')))
        
def test_from_sequence_3d_to_4d():
    for dim_name, dim in (('time', 3), ('vector', 4)):
        ext1 = dcmmeta.DcmMetaExtension.make_empty((2, 2, 2), np.eye(4), 2)
        ext1.get_class_dict(('global', 'const'))['global_const_const'] = 1
        ext1.get_class_dict(('global', 'const'))['global_const_var'] = 1
        ext1.get_class_dict(('global', 'const'))['global_const_missing'] = 1
        ext1.get_class_dict(('global', 'slices'))['global_slices_const'] = [0, 1]
        ext1.get_class_dict(('global', 'slices'))['global_slices_var'] = [0, 1]
        ext1.get_class_dict(('global', 'slices'))['global_slices_missing'] = [0, 1]
        ext2 = dcmmeta.DcmMetaExtension.make_empty((2, 2, 2), np.eye(4), 2)
        ext2.get_class_dict(('global', 'const'))['global_const_const'] = 1
        ext2.get_class_dict(('global', 'const'))['global_const_var'] = 2
        ext2.get_class_dict(('global', 'slices'))['global_slices_const'] = [0, 1]
        ext2.get_class_dict(('global', 'slices'))['global_slices_var'] = [1, 2]
        
        merged = dcmmeta.DcmMetaExtension.from_sequence([ext1, ext2], dim)
        eq_(merged.get_values_and_class('global_const_const'),
            (1, ('global', 'const')))
        eq_(merged.get_values_and_class('global_const_var'),
            ([1, 2], (dim_name, 'samples')))
        eq_(merged.get_values_and_class('global_const_missing'),
            ([1, None], (dim_name, 'samples')))
        eq_(merged.get_values_and_class('global_slices_const'),
            ([0, 1], (dim_name, 'slices')))
        eq_(merged.get_values_and_class('global_slices_var'),
            ([0, 1, 1, 2], ('global', 'slices')))
        eq_(merged.get_values_and_class('global_slices_missing'),
            ([0, 1, None, None], ('global', 'slices')))