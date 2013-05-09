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
    ext = dcmmeta.DcmMeta((2, 2, 2), np.eye(4))
    eq_(ext.get_valid_classes(), ('const', 'per_slice'))
    
    ext.shape = (2, 2, 2, 2)
    eq_(ext.get_valid_classes(), 
        ('const', 'per_volume', 'per_slice')
       )
       
    ext.shape = (2, 2, 2, 1, 2)
    eq_(ext.get_valid_classes(), 
        ('const', 'per_volume', 'per_slice')
       )
 
    ext.shape = (2, 2, 2, 2, 2)
    eq_(ext.get_valid_classes(), 
        ('const',
         'per_sample_4',
         'per_sample_3',
         'per_volume',
         'per_slice',
        )
       )
       
def test_get_n_vals_4d():
    ext = dcmmeta.DcmMeta((64, 64, 7, 11), 
                          np.eye(4), 
                          np.eye(4), 
                          2)
    eq_(ext.get_n_vals('const'), 1)
    eq_(ext.get_n_vals('per_slice'), 7 * 11)
    eq_(ext.get_n_vals('per_volume'), 11)
    
def test_get_n_vals_4d_vec():
    ext = dcmmeta.DcmMeta((64, 64, 7, 1, 11), 
                          np.eye(4), 
                          np.eye(4), 
                          2)
    eq_(ext.get_n_vals('const'), 1)
    eq_(ext.get_n_vals('per_slice'), 7 * 11)
    eq_(ext.get_n_vals('per_volume'), 11)

def test_get_n_vals_5d():
    ext = dcmmeta.DcmMeta((64, 64, 7, 11, 13), 
                          np.eye(4), 
                          np.eye(4),
                          2
                         )
    eq_(ext.get_n_vals('const'), 1)
    eq_(ext.get_n_vals('per_slice'), 7 * 11 * 13)
    eq_(ext.get_n_vals('per_volume'), 11 * 13)
    eq_(ext.get_n_vals('per_sample_3'), 11)
    eq_(ext.get_n_vals('per_sample_4'), 13)

class TestCheckValid(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMeta((64, 64, 2, 3, 4), 
                                   np.eye(4), 
                                   np.eye(4), 
                                   2)
    
    def test_empty(self):
        self.ext.check_valid()
                                               
    def test_const(self):
        self.ext['const']['ConstTest'] = 2
        self.ext.check_valid()
        del self.ext['const']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_per_slice(self):
        cls_dict = self.ext['per_slice']
        cls_dict['SliceTest'] = [0] * 23
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['SliceTest'] = [0] * 24
        self.ext.check_valid()
        del self.ext['per_slice']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_per_volume(self):
        cls_dict = self.ext['per_volume']
        cls_dict['PerVolumeTest'] = [0] * 2
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['PerVolumeTest'] = [0] * 3
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['PerVolumeTest'] = [0] * 12
        self.ext.check_valid()
        del self.ext['per_volume']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_per_sample_3(self):
        cls_dict = self.ext['per_sample_3']
        cls_dict['PerSample3Test'] = [0] * 4
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['PerSample3Test'] = [0] * 3
        self.ext.check_valid()
        del self.ext['per_sample_3']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_per_sample_4(self):
        cls_dict = self.ext['per_sample_4']
        cls_dict['PerSample4Test'] = [0] * 3
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        cls_dict['PerSample4Test'] = [0] * 4
        self.ext.check_valid()
        del self.ext['per_sample_4']
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_invalid_affine(self):
        self.ext['meta']['affine'] = np.eye(3).tolist()
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_invalid_slice_dim(self):
        self.ext['meta']['slice_dim'] = 3
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        self.ext['meta']['slice_dim'] = -1
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_invalid_shape(self):
        self.ext['meta']['shape'] = [2, 2]
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
    def test_multiple_classes(self):
        self.ext['const']['Test'] = 0 
        self.ext['per_sample_3']['Test'] = [0] * 3
        assert_raises(dcmmeta.InvalidExtensionError, self.ext.check_valid)
        
def test_dcmmeta_affine():
    ext = dcmmeta.DcmMeta((64, 64, 2), 
                          np.diag([1, 2, 3, 4]), 
                          np.eye(4), 
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
    ext = dcmmeta.DcmMeta((64, 64, 2), np.eye(4))
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
    ext = dcmmeta.DcmMeta((64, 64, 2), np.eye(4))
    eq_(ext.shape, (64, 64, 2))
    assert_raises(ValueError, 
                  setattr, 
                  ext, 
                  'shape', 
                  (64,)
                 )
    ext.shape = (128, 128, 64)
    eq_(ext.shape, (128, 128, 64))
    
def test_dcmmeta_version():
    ext = dcmmeta.DcmMeta((64, 64, 2), np.eye(4))
    eq_(ext.version, dcmmeta._meta_version)
    ext.version = 1.0
    eq_(ext.version, 1.0)
    
def test_dcmmeta_slice_norm():
    ext = dcmmeta.DcmMeta((64, 64, 2), np.eye(4), np.eye(4), 2)
    ok_(np.allclose(ext.slice_normal, [0, 0, 1]))
    ext.slice_dim = 1
    ok_(np.allclose(ext.slice_normal, [0, 1, 0]))
    
def test_dcmmeta_n_slices():
    ext = dcmmeta.DcmMeta((64, 64, 2), np.eye(4), np.eye(4), 2)
    eq_(ext.n_slices, 2)
    ext.slice_dim = 1
    eq_(ext.n_slices, 64)
    
class TestGetKeysClassesValues(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMeta((64, 64, 2, 3, 4), 
                                   np.eye(4), 
                                   np.eye(4),
                                   2
                                  )
        self.keys = []
        for classification in self.ext.get_valid_classes():
            key = '%s_test' % classification
            self.keys.append(key)
            self.ext[classification][key] = \
                ([0] * self.ext.get_n_vals(classification)) 
    
    def test_get_keys(self):
        eq_(set(self.keys), set(self.ext.get_all_keys()))
        
    def test_get_classification(self):
        eq_(self.ext.get_classification('foo'), None)
        for classification in self.ext.get_valid_classes():
            key = '%s_test' % classification
            eq_(self.ext.get_classification(key), classification)
            
    def test_get_values(self):
        eq_(self.ext.get_values('foo'), None)
        for classification in self.ext.get_valid_classes():
            key = '%s_test' % classification
            eq_(self.ext.get_values(key), 
                [0] * self.ext.get_n_vals(classification)
               )

class TestFiltering(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMeta((64, 64, 2, 3, 4), 
                                   np.eye(4), 
                                   np.eye(4), 
                                   2
                                  )
        
        for classification in self.ext.get_valid_classes():
            prefix = '%s_test_' % classification
            n_vals = self.ext.get_n_vals(classification)
            self.ext[classification][prefix + 'foo'] = ([0] * n_vals)
            self.ext[classification][prefix + 'foobaz'] = ([0] * n_vals)
    
    def test_filter_all(self):
        self.ext.filter_meta(lambda key, val: 'foo' in key)
        eq_(len(self.ext.get_all_keys()), 0)
        
    def test_filter_some(self):
        self.ext.filter_meta(lambda key, val: key.endswith('baz'))
        keys = self.ext.get_all_keys()
        for classification in self.ext.get_valid_classes():
            prefix = '%s_test_' % classification
            ok_(prefix + 'foo' in keys)
            ok_(not prefix + 'foobaz' in keys)
    
class TestSimplify(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMeta((64, 64, 2, 4, 8), 
                                   np.eye(4),
                                   np.eye(4),
                                   2
                                  )
                                  
    def test_simplify_const(self):
        self.ext['const']['Test1'] = None
        self.ext['const']['Test2'] = 'TestData'
        
        eq_(self.ext.simplify('Test1'), True)
        eq_(self.ext.get_classification('Test1'), None)
        
        eq_(self.ext.simplify('Test2'), False)
        eq_(self.ext.get_classification('Test2'), 'const')
        
    def test_simplify_none(self):
        for classification in self.ext.get_valid_classes():
            key = '%s_test' % classification
            if classification == 'const':
                self.ext[classification][key] = None
            else:
                n_vals = self.ext.get_n_vals(classification)
                self.ext[classification][key] = [None] * n_vals
        
        for classification in self.ext.get_valid_classes():
            key = '%s_test' % classification
            eq_(self.ext.simplify(key), True)
            eq_(self.ext.get_classification(key), None)
            eq_(self.ext.get_values(key), None)
    
    def test_simplify_per_slice(self):
        self.ext['per_slice']['Test1'] = [0] * (2 * 4 * 8)
        self.ext['per_slice']['Test2'] = []
        for idx in xrange(8):
            self.ext['per_slice']['Test2'] += [idx] * (2 * 4)
        self.ext['per_slice']['Test3'] = []
        for vec_idx in xrange(8):
            for time_idx in xrange(4):
                self.ext['per_slice']['Test3'] += [time_idx] * 2
        
        self.ext['per_slice']['Test4'] = []
        for vol_idx in xrange(4 * 8):
            self.ext['per_slice']['Test4'] += [vol_idx] * 2
        self.ext.check_valid()
            
        eq_(self.ext.simplify('Test1'), True)
        eq_(self.ext.get_classification('Test1'), 'const')
        
        eq_(self.ext.simplify('Test2'), True)
        eq_(self.ext.get_classification('Test2'), 'per_sample_4')
        
        eq_(self.ext.simplify('Test3'), True)
        eq_(self.ext.get_classification('Test3'), 'per_sample_3')
        
        eq_(self.ext.simplify('Test4'), True)
        eq_(self.ext.get_classification('Test4'), 'per_volume')
        
    def test_simplify_per_volume(self):
        self.ext['per_volume']['Test1'] = [0] * (4 * 8)
        self.ext['per_volume']['Test2'] = []
        for idx in xrange(8):
            self.ext['per_volume']['Test2'] += [idx] * 4
        print self.ext['per_volume']['Test2']
        self.ext['per_volume']['Test3'] = []
        for vec_idx in xrange(8):
            for time_idx in xrange(4):
                self.ext['per_volume']['Test3'] += [time_idx]
        self.ext.check_valid()
                
        eq_(self.ext.simplify('Test1'), True)
        eq_(self.ext.get_classification('Test1'), 'const')
        
        eq_(self.ext.simplify('Test2'), True)
        eq_(self.ext.get_classification('Test2'), 'per_sample_4')
        
        eq_(self.ext.simplify('Test3'), True)
        eq_(self.ext.get_classification('Test3'), 'per_sample_3')
        
    def test_simplify_per_sample(self):
        #Per sample can only become const
        self.ext['per_sample_3']['Test1'] = [0] * 4
        self.ext['per_sample_4']['Test2'] = [0] * 8
        self.ext.check_valid()
        
        eq_(self.ext.simplify('Test1'), True)
        eq_(self.ext.get_classification('Test1'), 'const')
        eq_(self.ext.simplify('Test2'), True)
        eq_(self.ext.get_classification('Test2'), 'const')

def test_simp_sngl_slc_5d():
    ext = dcmmeta.DcmMeta((64, 64, 1, 2, 4), 
                          np.eye(4),
                          np.eye(4),
                          2
                         )
    ext['per_slice']['Test1'] = range(2 * 4)
    ext['per_slice']['Test2'] = []
    for idx in xrange(4):
        ext['per_slice']['Test2'] += [idx] * 2
    ext['per_slice']['Test3'] = []
    for vec_idx in xrange(4):
        for time_idx in xrange(2):
            ext['per_slice']['Test3'] += [time_idx]
    ext.check_valid()
    
    eq_(ext.simplify('Test1'), True)
    eq_(ext.get_classification('Test1'), 'per_volume')
    eq_(ext.get_values('Test1'), range(8))
    
    eq_(ext.simplify('Test2'), True)
    eq_(ext.get_classification('Test2'), 'per_sample_4')
    eq_(ext.get_values('Test2'), range(4))
    
    eq_(ext.simplify('Test3'), True)
    eq_(ext.get_classification('Test3'), 'per_sample_3')
    eq_(ext.get_values('Test3'), range(2))

class TestGetSubset(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMeta((64, 64, 3, 5, 7), 
                                   np.eye(4), 
                                   np.eye(4), 
                                   2
                                  )
                                                      
        #Add an element to every classification
        for classification in self.ext.get_valid_classes():
            key = '%s_test' % classification
            n_vals = self.ext.get_n_vals(classification)
            self.ext[classification][key] = range(n_vals)
        
    def test_slice_subset(self):
        for slc_idx in xrange(self.ext.n_slices):
            sub = self.ext.get_subset(2, slc_idx)
            sub.check_valid()
            
            for classification in self.ext.get_valid_classes():
                key = '%s_test' % classification
                if classification == 'per_slice':
                    eq_(sub.get_classification(key), 'per_volume')
                    eq_(sub.get_values(key), range(slc_idx, (3 * 5 * 7), 3))
                else:
                    eq_(sub.get_classification(key), 
                        self.ext.get_classification(key)
                       )
                    eq_(sub.get_values(key), self.ext.get_values(key))
                       
    def test_slice_subset_simplify(self):
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                for slice_idx in xrange(3):
                    if slice_idx == 1:
                        vals.append(1)
                    else:
                        vals.append(vector_idx)
        self.ext['per_slice']['per_slice_to_const'] = vals
        self.ext.check_valid()
        
        sub = self.ext.get_subset(2, 1)
        sub.check_valid()
        eq_(sub.get_classification('per_slice_to_const'), 'const')
        eq_(sub.get_values('per_slice_to_const'), 1)
                       
    def test_sample_3_subset(self):
        for time_idx in xrange(5):
            sub = self.ext.get_subset(3, time_idx)
            sub.check_valid()
            for classification in self.ext.get_valid_classes():
                key = '%s_test' % classification
                if classification == 'per_sample_3':
                    ok_(not classification in sub.get_valid_classes())
                    eq_(sub.get_classification(key), 'const')
                    eq_(sub.get_values(key), time_idx)
                elif classification == 'per_sample_4':
                    ok_(not classification in sub.get_valid_classes())
                    eq_(sub.get_classification(key), 'per_volume')
                    eq_(sub.get_values(key), self.ext.get_values(key))
                elif classification == 'per_volume':
                    eq_(sub.get_classification(key), 'per_volume')
                    eq_(sub.get_values(key), range(time_idx, 5*7, 5))
                elif classification == 'const':
                    eq_(sub.get_classification(key), 'const')
                    eq_(sub.get_values(key), self.ext.get_values(key))
                else:
                    eq_(sub.get_classification(key), 'per_slice')
                    slc_result = []
                    for start_idx in range(time_idx*3, 3*5*7, 3*5):
                        slc_result += range(start_idx, start_idx+3)
                    eq_(sub.get_values(key), slc_result)
            
    def test_sample_3_subset_simplify(self):
        #Test for simplification of per_volume meta that becomes constant
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                if time_idx == 0:
                    vals.append(0)
                else:
                    vals.append(time_idx * vector_idx)
        self.ext['per_volume']['per_vol_to_const'] = vals
            
        sub = self.ext.get_subset(3, 0)
        sub.check_valid()
        eq_(sub.get_classification('per_vol_to_const'), 'const')
        eq_(sub.get_values('per_vol_to_const'), 0)

        #Test simplification of global slices that become constant
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                for slice_idx in xrange(3):
                    if time_idx == 1:
                        vals.append(1)
                    else:
                        vals.append(slice_idx * time_idx * vector_idx)
        self.ext['per_slice']['per_slc_to_const'] = vals
            
        sub = self.ext.get_subset(3, 1)
        sub.check_valid()
        eq_(sub.get_classification('per_slc_to_const'), 'const')
        eq_(sub.get_values('per_slc_to_const'), 1)
    
    def test_sample_4_subset(self):
        for vector_idx in xrange(7):
            sub = self.ext.get_subset(4, vector_idx)
            sub.check_valid()
            for classification in self.ext.get_valid_classes():
                key = '%s_test' % classification
                if classification == 'per_sample_4':
                    ok_(not classification in sub.get_valid_classes())
                    eq_(sub.get_classification(key), 'const')
                    eq_(sub.get_values(key), vector_idx)
                elif classification == 'per_sample_3':
                    ok_(not classification in sub.get_valid_classes())
                    eq_(sub.get_classification(key), 'per_volume')
                    eq_(sub.get_values(key), self.ext.get_values(key))
                elif classification == 'per_volume':
                    eq_(sub.get_classification(key), 'per_volume')
                    eq_(sub.get_values(key), 
                        range(vector_idx * 5, (vector_idx * 5) + 5))
                elif classification == 'const':
                    eq_(sub.get_classification(key), 'const')
                    eq_(sub.get_values(key), self.ext.get_values(key))
                else:
                    eq_(sub.get_classification(key), 'per_slice')
                    eq_(sub.get_values(key), 
                        range(vector_idx * 15, (vector_idx * 15) + 15))
                            
    def test_sample_4_subset_simplify(self):
        #Test for simplification of per_volume meta that becomes constant
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                if vector_idx == 0:
                    vals.append(0)
                else:
                    vals.append(time_idx * vector_idx)
        self.ext['per_volume']['per_vol_to_const'] = vals
            
        sub = self.ext.get_subset(4, 0)
        sub.check_valid()
        eq_(sub.get_classification('per_vol_to_const'), 'const')
        eq_(sub.get_values('per_vol_to_const'), 0)

        #Test simplification of global slices that become constant
        vals = []
        for vector_idx in xrange(7):
            for time_idx in xrange(5):
                for slice_idx in xrange(3):
                    if vector_idx == 1:
                        vals.append(1)
                    else:
                        vals.append(slice_idx * time_idx * vector_idx)
        self.ext['per_slice']['per_slc_to_const'] = vals
            
        sub = self.ext.get_subset(4, 1)
        sub.check_valid()
        eq_(sub.get_classification('per_slc_to_const'), 'const')
        eq_(sub.get_values('per_slc_to_const'), 1)

class TestChangeClass(object):
    def setUp(self):
        self.ext = dcmmeta.DcmMeta((2, 2, 3, 5, 7), 
                                   np.eye(4), 
                                   np.eye(4), 
                                   2
                                  )
        #Add an element to every classification
        for classification in self.ext.get_valid_classes():
            key = '%s_test' % classification
            n_vals = self.ext.get_n_vals(classification)
            if n_vals == 1:
                vals = 0
            else:
                vals = range(n_vals)
            self.ext[classification][key] = vals
    
    def test_change_none(self):
        eq_(self.ext._get_changed_class('None_test', 'const'),
            None)
        eq_(self.ext._get_changed_class('None_test', 'per_slice'),
            [None] * (3 * 5 * 7))
        eq_(self.ext._get_changed_class('None_test', 'per_volume'),
            [None] * (5 * 7))
        eq_(self.ext._get_changed_class('None_test', 'per_sample_4'),
            [None] * 7)
        eq_(self.ext._get_changed_class('None_test', 'per_sample_3'),
            [None] * 5)
        
    def test_change_const(self):
        eq_(self.ext._get_changed_class('const_test', 'per_slice'),
            [0] * (3 * 5 * 7))
        eq_(self.ext._get_changed_class('const_test', 'per_volume'),
            [0] * (5 * 7))
        eq_(self.ext._get_changed_class('const_test', 'per_sample_4'),
            [0] * 7)
        eq_(self.ext._get_changed_class('const_test', 'per_sample_3'),
            [0] * 5)
    
    def test_change_per_sample_4(self):
        vals = []
        for vector_idx in xrange(7):
            vals += [vector_idx] * 15
        eq_(self.ext._get_changed_class('per_sample_4_test', 'per_slice'),
            vals)
        vals = []
        for vector_idx in xrange(7):
            vals += [vector_idx] * 5
        eq_(self.ext._get_changed_class('per_sample_4_test', 'per_volume'),
            vals)
        
    def test_change_per_sample_3(self):
        vals = []
        for time_idx in xrange(5):
            vals += [time_idx] * 21
        eq_(self.ext._get_changed_class('per_sample_3_test', 'per_slice'),
            vals)
        vals = []
        for time_idx in xrange(5):
            vals += [time_idx] * 7
        eq_(self.ext._get_changed_class('per_sample_3_test', 'per_volume'),
            vals)
            
    def test_change_per_volume(self):
        vals = []
        for vol_idx in xrange(5*7):
            vals += [vol_idx] * 3
        eq_(self.ext._get_changed_class('per_volume_test', 'per_slice'),
            vals)

def test_from_sequence_2d_to_3d():
    ext1 = dcmmeta.DcmMeta((2, 2, 1), np.eye(4), np.eye(4), 2)
    ext1['const']['const_test'] = 1
    ext1['const']['var_test'] = 1
    ext1['const']['missing_test1'] = 1
    ext2 = dcmmeta.DcmMeta((2, 2, 1), np.eye(4), np.eye(4), 2)
    ext2['const']['const_test'] = 1
    ext2['const']['var_test'] = 2
    ext2['const']['missing_test2'] = 1
    
    merged = dcmmeta.DcmMeta.from_sequence([ext1, ext2], 2)
    merged.check_valid()
    eq_(merged.get_classification('const_test'), 'const')
    eq_(merged.get_values('const_test'), 1)
    eq_(merged.get_classification('var_test'), 'per_slice')
    eq_(merged.get_values('var_test'), [1,2])
    eq_(merged.get_classification('missing_test1'), 'per_slice')
    eq_(merged.get_values('missing_test1'), [1, None])
    eq_(merged.get_classification('missing_test2'), 'per_slice')
    eq_(merged.get_values('missing_test2'), [None, 1])
        
def test_from_sequence_3d_to_4d():
    for dim in (3, 4):
        ext1 = dcmmeta.DcmMeta((2, 2, 2), np.eye(4), np.eye(4), 2)
        ext1['const']['const_const'] = 1
        ext1['const']['const_var'] = 1
        ext1['const']['const_missing'] = 1
        ext1['per_slice']['per_slice_const'] = [0, 1]
        ext1['per_slice']['per_slice_var'] = [0, 1]
        ext1['per_slice']['per_slice_missing'] = [0, 1]
        ext2 = dcmmeta.DcmMeta((2, 2, 2), np.eye(4), np.eye(4), 2)
        ext2['const']['const_const'] = 1
        ext2['const']['const_var'] = 2
        ext2['per_slice']['per_slice_const'] = [0, 1]
        ext2['per_slice']['per_slice_var'] = [1, 2]
        
        merged = dcmmeta.DcmMeta.from_sequence([ext1, ext2], dim)
        merged.check_valid()
        eq_(merged.get_classification('const_const'), 'const')
        eq_(merged.get_values('const_const'), 1)
        eq_(merged.get_classification('const_var'), 'per_volume')
        eq_(merged.get_values('const_var'), [1, 2])
        eq_(merged.get_classification('const_missing'), 'per_volume')
        eq_(merged.get_values('const_missing'), [1, None])
        eq_(merged.get_classification('per_slice_const'), 'per_slice')
        eq_(merged.get_values('per_slice_const'), [0, 1, 0, 1])
        eq_(merged.get_classification('per_slice_var'), 'per_slice')
        eq_(merged.get_values('per_slice_var'), [0, 1, 1, 2])
        eq_(merged.get_classification('per_slice_missing'), 'per_slice')
        eq_(merged.get_values('per_slice_missing'), [0, 1, None, None])

def test_from_sequence_4d_to_5d():
    ext1 = dcmmeta.DcmMeta((2, 2, 2, 2), np.eye(4), np.eye(4), 2)
    ext1['const']['const_const'] = 1
    ext1['const']['const_var'] = 1
    ext1['const']['const_missing'] = 1
    ext1['per_slice']['per_slice_const'] = [0, 1, 2, 3]
    ext1['per_slice']['per_slice_var'] = [0, 1, 2, 3]
    ext1['per_slice']['per_slice_missing'] = [0, 1, 2, 3]
    ext1['per_volume']['per_volume_const'] = [0, 1]
    ext1['per_volume']['per_volume_var'] = [0, 1]
    ext1['per_volume']['per_volume_missing'] = [0, 1]
    
    ext2 = dcmmeta.DcmMeta((2, 2, 2, 2), np.eye(4), np.eye(4), 2)
    ext2['const']['const_const'] = 1
    ext2['const']['const_var'] = 2
    ext2['per_slice']['per_slice_const'] = [0, 1, 2, 3]
    ext2['per_slice']['per_slice_var'] = [1, 2, 3, 4]
    ext2['per_volume']['per_volume_const'] = [0, 1]
    ext2['per_volume']['per_volume_var'] = [1, 2]
    
    merged = dcmmeta.DcmMeta.from_sequence([ext1, ext2], 4)
    merged.check_valid()
    eq_(merged.get_classification('const_const'), 'const')
    eq_(merged.get_values('const_const'), 1)
    eq_(merged.get_classification('const_var'), 'per_sample_4')
    eq_(merged.get_values('const_var'), [1, 2])
    eq_(merged.get_classification('const_missing'), 'per_sample_4')
    eq_(merged.get_values('const_missing'), [1, None])
    eq_(merged.get_classification('per_slice_const'), 'per_slice')
    eq_(merged.get_values('per_slice_const'), [0, 1, 2, 3, 0, 1, 2, 3])
    eq_(merged.get_classification('per_slice_var'), 'per_slice')
    eq_(merged.get_values('per_slice_var'), [0, 1, 2, 3, 1, 2, 3, 4])
    eq_(merged.get_classification('per_slice_missing'), 'per_slice')
    eq_(merged.get_values('per_slice_missing'), 
        [0, 1, 2, 3, None, None, None, None])
    eq_(merged.get_classification('per_volume_const'), 'per_sample_3')
    eq_(merged.get_values('per_volume_const'), [0, 1])
    eq_(merged.get_classification('per_volume_var'), 'per_volume')
    eq_(merged.get_values('per_volume_var'), [0, 1, 1, 2])
    eq_(merged.get_classification('per_volume_missing'), 'per_volume')
    eq_(merged.get_values('per_volume_missing'), [0, 1, None, None])

def test_from_sequence_no_slc():
    ext1 = dcmmeta.DcmMeta((2, 2, 2), np.eye(4))
    ext2 = dcmmeta.DcmMeta((2, 2, 2), np.eye(4))
    merged = dcmmeta.DcmMeta.from_sequence([ext1, ext2], 4)
   
def test_nifti_wrapper_init():
    nii = nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
    assert_raises(dcmmeta.MissingExtensionError, 
                  dcmmeta.NiftiWrapper,
                  nii)
    hdr = nii.get_header()
    meta = dcmmeta.DcmMeta((5, 5, 5), np.eye(4))
    ext = dcmmeta.DcmMetaExtension.from_dcmmeta(meta)
    hdr.extensions.append(ext)
    nw = dcmmeta.NiftiWrapper(nii)
    eq_(nw.meta_ext, meta)
    
    nii2 = nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
    nw2 = dcmmeta.NiftiWrapper(nii, True)
    eq_(nw.meta_ext, nw2.meta_ext)
    
class TestMetaValid(object):
    def setUp(self):
        nii = nb.Nifti1Image(np.zeros((5, 5, 5, 7, 9)), np.eye(4))
        hdr = nii.get_header()
        hdr.set_dim_info(None, None, 2)
        self.nw = dcmmeta.NiftiWrapper(nii, True)
        
    def test_initial_setup(self):
        for classification in self.nw.meta_ext.get_valid_classes():
            ok_(self.nw.meta_valid(classification))
        
    def test_sample_3_changed(self):
        self.nw.meta_ext.shape = (5, 5, 5, 3, 9)
        for classification in self.nw.meta_ext.get_valid_classes():
            if classification in ('per_slice', 'per_volume', 'per_sample_3'):
                eq_(self.nw.meta_valid(classification), False)
            else:
                ok_(self.nw.meta_valid(classification))
    
    def test_sample_4_changed(self):
        self.nw.meta_ext.shape = (5, 5, 5, 7, 3)
        for classification in self.nw.meta_ext.get_valid_classes():
            if classification in ('per_slice', 'per_volume', 'per_sample_4'):
                eq_(self.nw.meta_valid(classification), False)
            else:
                print classification
                ok_(self.nw.meta_valid(classification))
          
    def test_slice_dim_changed(self):
        self.nw.meta_ext.slice_dim = 0
        for classification in self.nw.meta_ext.get_valid_classes():
            if classification == 'per_slice':
                eq_(self.nw.meta_valid(classification), False)
            else:
                ok_(self.nw.meta_valid(classification))
        
    def test_slice_dir_changed(self):
        aff = self.nw.nii_img.get_affine()
        aff[:] = np.c_[aff[2, :],
                       aff[1, :],
                       aff[0, :],
                       aff[3, :],
                      ]
        
        for classification in self.nw.meta_ext.get_valid_classes():
            if classification == 'per_slice':
                eq_(self.nw.meta_valid(classification), False)
            else:
                ok_(self.nw.meta_valid(classification))
        
class TestGetMeta(object):
    def setUp(self):
        nii = nb.Nifti1Image(np.zeros((5, 5, 5, 7, 9)), np.eye(4))
        hdr = nii.get_header()
        hdr.set_dim_info(None, None, 2)
        self.nw = dcmmeta.NiftiWrapper(nii, True)
        
        #Add an element to every classification
        for classification in self.nw.meta_ext.get_valid_classes():
            key = '%s_test' % classification
            n_vals = self.nw.meta_ext.get_n_vals(classification)
            if n_vals == 1:
                vals = 0
            else:
                vals = range(n_vals)
            self.nw.meta_ext[classification][key] = vals
            
    def test_invalid_index(self):
        assert_raises(IndexError, 
                      self.nw.get_meta('per_volume_test'), 
                      (0, 0, 0, 0))
        assert_raises(IndexError, 
                      self.nw.get_meta('per_volume_test'), 
                      (0, 0, 0, 0, 0, 0))
        assert_raises(IndexError, 
                      self.nw.get_meta('per_volume_test'), 
                      (6, 0, 0, 0, 0))
        assert_raises(IndexError, 
                      self.nw.get_meta('per_volume_test'), 
                      (-1, 0, 0, 0, 0))
            
    def test_get_item(self):
        for classification in self.nw.meta_ext.get_valid_classes():
            key = '%s_test' % classification
            if classification == 'const':
                eq_(self.nw[key], 0)
            else:
                assert_raises(KeyError,
                              self.nw.__getitem__,
                              key)
        
    def test_get_const(self):
        eq_(self.nw.get_meta('const_test'), 0)
        eq_(self.nw.get_meta('const_test', (0, 0, 0, 0, 0)), 0)
        eq_(self.nw.get_meta('const_test', (0, 0, 3, 4, 5)), 0)
        
    def test_get_per_slice(self):
        eq_(self.nw.get_meta('per_slice_test'), None)
        eq_(self.nw.get_meta('per_slice_test', None, 0), 0)
        for vector_idx in xrange(9):
            for time_idx in xrange(7):
                for slice_idx in xrange(5):
                    idx = (0, 0, slice_idx, time_idx, vector_idx)
                    eq_(self.nw.get_meta('per_slice_test', idx),
                        slice_idx + (time_idx * 5) + (vector_idx * 7 * 5)
                       )
                       
    def test_per_volume(self):
        eq_(self.nw.get_meta('per_volume_test'), None)
        eq_(self.nw.get_meta('per_volume_test', None, 0), 0)
        for vector_idx in xrange(9):
            for time_idx in xrange(7):
                for slice_idx in xrange(5):
                    idx = (0, 0, slice_idx, time_idx, vector_idx)
                    eq_(self.nw.get_meta('per_volume_test', idx),
                        time_idx + (vector_idx * 7)
                       )
                       
    def test_get_sample_3(self):
        eq_(self.nw.get_meta('per_sample_3_test'), None)
        eq_(self.nw.get_meta('per_sample_3_test', None, 0), 0)
        for vector_idx in xrange(9):
            for time_idx in xrange(7):
                for slice_idx in xrange(5):
                    idx = (0, 0, slice_idx, time_idx, vector_idx)
                    eq_(self.nw.get_meta('per_sample_3_test', idx),
                        time_idx
                       )
                       
    def get_get_sample_4(self):
        eq_(self.nw.get_meta('per_sample_4_test'), None)
        eq_(self.nw.get_meta('per_sample_4_test', None, 0), 0)
        for vector_idx in xrange(9):
            for time_idx in xrange(7):
                for slice_idx in xrange(5):
                    idx = (0, 0, slice_idx, time_idx, vector_idx)
                    eq_(self.nw.get_meta('per_sample_4_test', idx),
                        vector_idx
                       )
 
class TestSplit(object):
    def setUp(self):
        self.arr = np.arange(3 * 3 * 3 * 5 * 7).reshape(3, 3, 3, 5, 7)
        nii = nb.Nifti1Image(self.arr, np.diag([1.1, 1.1, 1.1, 1.0]))
        hdr = nii.get_header()
        hdr.set_dim_info(None, None, 2)
        self.nw = dcmmeta.NiftiWrapper(nii, True)
        
        #Add an element to every classification
        for classification in self.nw.meta_ext.get_valid_classes():
            key = '%s_test' % classification
            n_vals = self.nw.meta_ext.get_n_vals(classification)
            if n_vals == 1:
                vals = 0
            else:
                vals = range(n_vals)
            self.nw.meta_ext[classification][key] = vals
            
    def test_split_slice(self):
        for split_idx, nw_split in enumerate(self.nw.split(2)):
            eq_(nw_split.nii_img.shape, (3, 3, 1, 5, 7))
            ok_(np.allclose(nw_split.nii_img.get_affine(), 
                            np.c_[[1.1, 0.0, 0.0, 0.0],
                                  [0.0, 1.1, 0.0, 0.0],
                                  [0.0, 0.0, 1.1, 0.0],
                                  [0.0, 0.0, 1.1*split_idx, 1.0]
                                 ]
                           )
               )
            ok_(np.all(nw_split.nii_img.get_data() == 
                       self.arr[:, :, split_idx:split_idx+1, :, :])
               )
            
    def test_split_time(self):
        for split_idx, nw_split in enumerate(self.nw.split(3)):
            eq_(nw_split.nii_img.shape, (3, 3, 3, 1, 7))
            ok_(np.allclose(nw_split.nii_img.get_affine(), 
                            np.diag([1.1, 1.1, 1.1, 1.0])))
            ok_(np.all(nw_split.nii_img.get_data() == 
                       self.arr[:, :, :, split_idx:split_idx+1, :])
               )
               
    def test_split_vector(self):
        for split_idx, nw_split in enumerate(self.nw.split(4)):
            eq_(nw_split.nii_img.shape, (3, 3, 3, 5))
            ok_(np.allclose(nw_split.nii_img.get_affine(), 
                            np.diag([1.1, 1.1, 1.1, 1.0])))
            ok_(np.all(nw_split.nii_img.get_data() == 
                       self.arr[:, :, :, :, split_idx])
               )
               
def test_split_keep_spatial():
    arr = np.arange(3 * 3 * 3).reshape(3, 3, 3) 
    nii = nb.Nifti1Image(arr, np.eye(4))
    nw = dcmmeta.NiftiWrapper(nii, True)
    
    for split_idx, nw_split in enumerate(nw.split(2)):
        eq_(nw_split.nii_img.shape, (3, 3, 1))
    

def test_from_dicom():
    data_dir = path.join(test_dir, 
                         'data', 
                         'dcmstack', 
                         '2D_16Echo_qT2')
    src_fn = path.join(data_dir, 'TE_40_SlcPos_-33.707626341697.dcm')
    src_dcm = dicom.read_file(src_fn)
    src_dw = nb.nicom.dicomwrappers.wrapper_from_data(src_dcm)
    meta = {'EchoTime': 40}
    nw = dcmmeta.NiftiWrapper.from_dicom(src_dcm, meta)
    hdr = nw.nii_img.get_header()
    eq_(nw.nii_img.get_shape(), (192, 192, 1))
    ok_(np.allclose(np.dot(np.diag([-1., -1., 1., 1.]), src_dw.get_affine()), 
                    nw.nii_img.get_affine())
       )
    eq_(hdr.get_xyzt_units(), ('mm', 'sec'))
    eq_(hdr.get_dim_info(), (0, 1, 2))
    eq_(nw.meta_ext.get_classification('EchoTime'), 'const')
    eq_(nw.meta_ext.get_values('EchoTime'), 40)
       
def test_from_2d_slice_to_3d():
    slice_nws = []
    for idx in xrange(3):
        arr = np.arange(idx * (4 * 4), (idx + 1) * (4 * 4)).reshape(4, 4, 1)
        aff = np.diag((1.1, 1.1, 1.1, 1.0))
        aff[:3, 3] += [0.0, 0.0, idx * 0.5]
        nii = nb.Nifti1Image(arr, aff)
        hdr = nii.get_header()
        hdr.set_dim_info(0, 1, 2)
        hdr.set_xyzt_units('mm', 'sec')
        nw = dcmmeta.NiftiWrapper(nii, True)
        nw.meta_ext['const']['EchoTime'] = 40
        nw.meta_ext['const']['SliceLocation'] = idx
        slice_nws.append(nw)
        
    merged = dcmmeta.NiftiWrapper.from_sequence(slice_nws, 2)
    eq_(merged.nii_img.shape, (4, 4, 3))
    ok_(np.allclose(merged.nii_img.get_affine(), 
                    np.diag((1.1, 1.1, 0.5, 1.0)))
       )
    eq_(merged.meta_ext.get_classification('EchoTime'), 'const')
    eq_(merged.meta_ext.get_values('EchoTime'), 40)
    eq_(merged.meta_ext.get_classification('SliceLocation'), 'per_slice')
    eq_(merged.meta_ext.get_values('SliceLocation'), range(3))
    merged_hdr = merged.nii_img.get_header()
    eq_(merged_hdr.get_dim_info(), (0, 1, 2))
    eq_(merged_hdr.get_xyzt_units(), ('mm', 'sec'))
    merged_data = merged.nii_img.get_data()
    for idx in xrange(3):
        ok_(np.all(merged_data[:, :, idx] == 
                   np.arange(idx * (4 * 4), (idx + 1) * (4 * 4)).reshape(4, 4))
           )
               
def test_from_3d_to_4d():
    time_nws = []
    for idx in xrange(3):
        arr = np.arange(idx * (4 * 4 * 4), 
                        (idx + 1) * (4 * 4 * 4)
                       ).reshape(4, 4, 4)
        nii = nb.Nifti1Image(arr, np.diag((1.1, 1.1, 1.1, 1.0)))
        hdr = nii.get_header()
        hdr.set_dim_info(0, 1, 2)
        hdr.set_xyzt_units('mm', 'sec')
        nw = dcmmeta.NiftiWrapper(nii, True)
        nw.meta_ext['const']['PatientID'] = 'Test'
        nw.meta_ext['const']['EchoTime'] = idx
        nw.meta_ext['per_slice']['SliceLocation'] = range(4)
        nw.meta_ext['per_slice']['AcquisitionTime'] = \
            range(idx * 4, (idx + 1) * 4)
        time_nws.append(nw)
        
    merged = dcmmeta.NiftiWrapper.from_sequence(time_nws, 3)
    eq_(merged.nii_img.shape, (4, 4, 4, 3))
    ok_(np.allclose(merged.nii_img.get_affine(), 
                    np.diag((1.1, 1.1, 1.1, 1.0)))
       )
    eq_(merged.meta_ext.get_classification('PatientID'), 'const')
    eq_(merged.meta_ext.get_values('PatientID'), 'Test')
    eq_(merged.meta_ext.get_classification('EchoTime'), 'per_volume')
    eq_(merged.meta_ext.get_values('EchoTime'), range(3))
    eq_(merged.meta_ext.get_classification('SliceLocation'), 'per_slice')
    eq_(merged.meta_ext.get_values('SliceLocation'), 
        [val for vol in range(3) for val in range(4)])
    eq_(merged.meta_ext.get_classification('AcquisitionTime'), 'per_slice')
    eq_(merged.meta_ext.get_values('AcquisitionTime'), range(4 * 3))
    merged_hdr = merged.nii_img.get_header()
    eq_(merged_hdr.get_dim_info(), (0, 1, 2))
    eq_(merged_hdr.get_xyzt_units(), ('mm', 'sec'))
    merged_data = merged.nii_img.get_data()
    for idx in xrange(3):
        ok_(np.all(merged_data[:, :, :, idx] == 
                   np.arange(idx * (4 * 4 * 4), 
                             (idx + 1) * (4 * 4 * 4)).reshape(4, 4, 4))
           )
           
def test_from_3d_to_5d():
    vector_nws = []
    for idx in xrange(3):
        arr = np.arange(idx * (4 * 4 * 4), 
                        (idx + 1) * (4 * 4 * 4)
                       ).reshape(4, 4, 4)
        nii = nb.Nifti1Image(arr, np.diag((1.1, 1.1, 1.1, 1.0)))
        hdr = nii.get_header()
        hdr.set_dim_info(0, 1, 2)
        hdr.set_xyzt_units('mm', 'sec')
        nw = dcmmeta.NiftiWrapper(nii, True)
        nw.meta_ext['const']['PatientID'] = 'Test'
        nw.meta_ext['const']['EchoTime'] = idx
        nw.meta_ext['per_slice']['SliceLocation'] = range(4)
        nw.meta_ext['per_slice']['AcquisitionTime'] = \
            range(idx * 4, (idx + 1) * 4)
        vector_nws.append(nw)
        
    merged = dcmmeta.NiftiWrapper.from_sequence(vector_nws, 4)
    eq_(merged.nii_img.shape, (4, 4, 4, 1, 3))
    ok_(np.allclose(merged.nii_img.get_affine(), 
                    np.diag((1.1, 1.1, 1.1, 1.0)))
       )
    eq_(merged.meta_ext.get_classification('PatientID'), 'const')
    eq_(merged.meta_ext.get_values('PatientID'), 'Test')
    eq_(merged.meta_ext.get_classification('EchoTime'), 'per_volume')
    eq_(merged.meta_ext.get_values('EchoTime'), range(3))
    eq_(merged.meta_ext.get_classification('SliceLocation'), 'per_slice')
    eq_(merged.meta_ext.get_values('SliceLocation'), 
        [val for vol in range(3) for val in range(4)])
    eq_(merged.meta_ext.get_classification('AcquisitionTime'), 'per_slice')
    eq_(merged.meta_ext.get_values('AcquisitionTime'), range(4 * 3))
    merged_hdr = merged.nii_img.get_header()
    eq_(merged_hdr.get_dim_info(), (0, 1, 2))
    eq_(merged_hdr.get_xyzt_units(), ('mm', 'sec'))
    merged_data = merged.nii_img.get_data()
    for idx in xrange(3):
        ok_(np.all(merged_data[:, :, :, 0, idx] == 
                   np.arange(idx * (4 * 4 * 4), 
                             (idx + 1) * (4 * 4 * 4)).reshape(4, 4, 4))
           )
           
def test_merge_inconsistent_hdr():
    #Test that inconsistent header data does not make it into the merged
    #result
    time_nws = []
    for idx in xrange(3):
        arr = np.arange(idx * (4 * 4 * 4), 
                        (idx + 1) * (4 * 4 * 4)
                       ).reshape(4, 4, 4)
        nii = nb.Nifti1Image(arr, np.diag((1.1, 1.1, 1.1, 1.0)))
        hdr = nii.get_header()
        if idx == 1:
            hdr.set_dim_info(1, 0, 2)
            hdr.set_xyzt_units('mm', None)
        else:
            hdr.set_dim_info(0, 1, 2)
            hdr.set_xyzt_units('mm', 'sec')
        nw = dcmmeta.NiftiWrapper(nii, True)
        time_nws.append(nw)
    
    merged = dcmmeta.NiftiWrapper.from_sequence(time_nws)
    merged_hdr = merged.nii_img.get_header()
    eq_(merged_hdr.get_dim_info(), (None, None, 2))
    eq_(merged_hdr.get_xyzt_units(), ('mm', 'unknown'))
        
def test_merge_with_slc_and_without():
    #Test merging two data sets where one has per slice meta and other does not
    input_nws = []
    for idx in xrange(3):
        arr = np.arange(idx * (4 * 4 * 4), 
                        (idx + 1) * (4 * 4 * 4)
                       ).reshape(4, 4, 4)
        nii = nb.Nifti1Image(arr, np.diag((1.1, 1.1, 1.1, 1.0)))
        hdr = nii.get_header()
        if idx == 0:
            hdr.set_dim_info(0, 1, 2)
        hdr.set_xyzt_units('mm', 'sec')
        nw = dcmmeta.NiftiWrapper(nii, True)
        nw.meta_ext['const']['PatientID'] = 'Test'
        nw.meta_ext['const']['EchoTime'] = idx
        if idx == 0:
            nw.meta_ext['per_slice']['SliceLocation'] = range(4)
        input_nws.append(nw)
        
    merged = dcmmeta.NiftiWrapper.from_sequence(input_nws)
    eq_(merged.nii_img.shape, (4, 4, 4, 3))
    ok_(np.allclose(merged.nii_img.get_affine(), 
                    np.diag((1.1, 1.1, 1.1, 1.0)))
       )
    eq_(merged.meta_ext.get_classification('PatientID'), 'const')
    eq_(merged.meta_ext.get_values('PatientID'), 'Test')
    eq_(merged.meta_ext.get_classification('EchoTime'), 'per_volume')
    eq_(merged.meta_ext.get_values('EchoTime'), range(3))
    eq_(merged.meta_ext.get_classification('SliceLocation'), 'per_slice')
    eq_(merged.meta_ext.get_values('SliceLocation'), 
        (range(4) + ([None] * 8)))
    merged_hdr = merged.nii_img.get_header()
    eq_(merged_hdr.get_xyzt_units(), ('mm', 'sec'))