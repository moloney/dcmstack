"""
Tests for dcmstack.dcmstack
"""
from __future__ import absolute_import, print_function

import sys
import warnings
from copy import deepcopy
from glob import glob
from hashlib import sha256
from os import path

import numpy as np
from nose.tools import ok_, eq_, assert_raises

from . import test_dir, src_dir

try:
    import pydicom
    from pydicom.datadict import keyword_dict, dictionary_VR
    from pydicom.uid import ExplicitVRLittleEndian
except ImportError:
    import dicom as pydicom
    from dicom.datadict import keyword_dict
    from dicom.datadict import dictionaryVR as dictionary_VR
    from dicom.UID import ExplicitVRLittleEndian
import nibabel as nb
from nibabel.orientations import aff2axcodes

import dcmstack

_def_file_meta = pydicom.dataset.Dataset()
_def_file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

def_dicom_attrs = {'file_meta' : _def_file_meta,
                   'ImagePositionPatient' : [0.0, 0.0, 0.0],
                   'ImageOrientationPatient' : [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                   'PixelSpacing' : [1.0, 1.0],
                   'SliceThickness' : 1.0,
                   'Rows' : 16,
                   'Columns' : 16,
                   'BitsAllocated' : 16,
                   'BitsStored' : 16,
                   'PixelRepresentation' : 0,
                   'SamplesPerPixel' : 1,
                   'PhotometricInterpretation': 'MONOCHROME2',
                  }

def make_dicom(attrs=None, pix_val=1):
    '''Build a mock DICOM dataset for testing purposes'''
    ds = pydicom.dataset.Dataset()
    ds.is_little_endian = True
    if attrs is None:
        attrs = {}
    for attr_name, attr_val in attrs.items():
        setattr(ds, attr_name, deepcopy(attr_val))
    for attr_name, attr_val in def_dicom_attrs.items():
        if not hasattr(ds, attr_name):
            setattr(ds, attr_name, deepcopy(attr_val))
    if not hasattr(ds, 'PixelData'):
        if ds.PixelRepresentation == 0:
            arr_dtype = np.uint16
        else:
            arr_dtype = np.int16
        arr = np.empty((ds.Rows, ds.Columns), dtype=arr_dtype)
        arr[:, :] = pix_val
        ds.PixelData = arr.tostring()
    return ds

def test_key_regex_filter():
        filt = dcmstack.make_key_regex_filter(['test', 'another'],
                                              ['2', 'another test'])
        ok_(filt('test', 1))
        ok_(filt('test another', 1))
        ok_(filt('another tes', 1))
        ok_(not filt('test2', 1))
        ok_(not filt('2 another', 1))
        ok_(not filt('another test', 1))

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
        vox_order = ''.join(aff2axcodes(self.affine))
        (vox_array,
         affine,
         aff_trans,
         ornt_trans) = dcmstack.reorder_voxels(self.vox_array,
                                               self.affine,
                                               vox_order)
        ok_((vox_array == self.vox_array).all())
        ok_((affine == self.affine).all())
        ok_((aff_trans == np.eye(4)).all())
        ok_(np.allclose(ornt_trans, [[0, 1], [1, 1], [2, 1]]))
        eq_(np.may_share_memory(affine, self.affine), False)

    def test_reorder(self):
        (vox_array,
         affine,
         aff_trans,
         ornt_trans) = dcmstack.reorder_voxels(self.vox_array,
                                               self.affine,
                                               'PRS')
        ok_(np.all(vox_array == np.array([[[[4, 5],
                                            [6, 7]],
                                           [[12, 13],
                                            [14,15]]
                                          ],
                                          [[[0, 1],
                                            [2, 3]],
                                           [[8, 9],
                                            [10, 11]]
                                          ]
                                         ]
                                        )
                  )
           )
        ok_(np.allclose(affine,
                        np.array([[0,1,0,0],
                                  [-1,0,0,1],
                                  [0,0,1,0],
                                  [0,0,0,1]])
                       )
           )

    def test_aniso_reorder(self):
        self.vox_array = self.vox_array.reshape(2, 4, 2)
        self.affine = np.eye(4)
        (vox_array,
         affine,
         aff_trans,
         ornt_trans) = dcmstack.reorder_voxels(self.vox_array,
                                               self.affine,
                                               'PLS')
        ok_(np.allclose(affine,
                        np.array([[0,-1,0,1],
                                  [-1,0,0,3],
                                  [0,0,1,0],
                                  [0,0,0,1]])
                       )
           )

def test_dcm_time_to_sec():
    eq_(dcmstack.dcm_time_to_sec('100235.123456'), 36155.123456)
    eq_(dcmstack.dcm_time_to_sec('100235'), 36155)
    eq_(dcmstack.dcm_time_to_sec('1002'), 36120)
    eq_(dcmstack.dcm_time_to_sec('10'), 36000)

    #Allow older NEMA style values
    eq_(dcmstack.dcm_time_to_sec('10:02:35.123456'), 36155.123456)
    eq_(dcmstack.dcm_time_to_sec('10:02:35'), 36155)
    eq_(dcmstack.dcm_time_to_sec('10:02'), 36120)
    eq_(dcmstack.dcm_time_to_sec('10'), 36000)

class TestDicomOrdering(object):
    def setUp(self):
        self.ds = {'EchoTime' : 2}

    def test_missing_key(self):
        ordering = dcmstack.DicomOrdering('blah')
        eq_(ordering.get_ordinate(self.ds), None)

    def test_non_abs(self):
        ordering = dcmstack.DicomOrdering('EchoTime')
        eq_(ordering.get_ordinate(self.ds), self.ds['EchoTime'])

    def test_abs(self):
        abs_order = [1,2,3]
        ordering = dcmstack.DicomOrdering('EchoTime', abs_ordering=abs_order)
        eq_(ordering.get_ordinate(self.ds),
            abs_order.index(self.ds['EchoTime']))

    def test_abs_as_str(self):
        abs_order = ['1','2','3']
        ordering = dcmstack.DicomOrdering('EchoTime',
                                          abs_ordering=abs_order,
                                          abs_as_str=True)
        eq_(ordering.get_ordinate(self.ds),
            abs_order.index(str(self.ds['EchoTime'])))

    def test_abs_missing(self):
        abs_order = [1,3]
        ordering = dcmstack.DicomOrdering('EchoTime', abs_ordering=abs_order)
        assert_raises(ValueError,
                      ordering.get_ordinate,
                      self.ds
                     )

def test_image_collision():
    dcm_path = path.join(test_dir,
                         'data',
                         'dcmstack',
                         '2D_16Echo_qT2',
                         'TE_20_SlcPos_-33.707626341697.dcm')
    dcm = pydicom.read_file(dcm_path)
    stack = dcmstack.DicomStack('EchoTime')
    stack.add_dcm(dcm)
    assert_raises(dcmstack.ImageCollisionError,
                  stack.add_dcm,
                  dcm)

class TestIncongruentImage(object):
    def setUp(self):
        dcm_path = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2',
                             'TE_20_SlcPos_-33.707626341697.dcm')
        self.dcm = pydicom.read_file(dcm_path)

        self.stack = dcmstack.DicomStack()
        self.stack.add_dcm(self.dcm)
        self.dcm = pydicom.read_file(dcm_path)

    def _chk_raises(self):
        assert_raises(dcmstack.IncongruentImageError,
                      self.stack.add_dcm,
                      self.dcm)

    def test_rows(self):
        self.dcm.Rows += 1
        self._chk_raises()

    def test_columns(self):
        self.dcm.Columns += 1
        self._chk_raises()

    def test_pix_space(self):
        self.dcm.PixelSpacing[0] *= 2
        self._chk_raises()

    def test_close_pix_space(self):
        self.dcm.PixelSpacing[0] += 1e-7
        # Shouldn't raise
        self.stack.add_dcm(self.dcm)

    def test_orientation(self):
        self.dcm.ImageOrientationPatient = \
            [0.5 * elem
             for elem in self.dcm.ImageOrientationPatient
            ]
        self._chk_raises()

    def test_close_orientation(self):
        self.dcm.ImageOrientationPatient = \
            [elem + 1e-7
             for elem in self.dcm.ImageOrientationPatient
            ]
        # Shouldn't raise
        self.stack.add_dcm(self.dcm)


class TestInvalidStack(object):
    def setUp(self):
        data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [pydicom.read_file(path.join(data_dir, fn))
                       for fn in ('TE_20_SlcPos_-33.707626341697.dcm',
                                  'TE_20_SlcPos_-23.207628249046.dcm',
                                  'TE_40_SlcPos_-33.707626341697.dcm',
                                  'TE_60_SlcPos_-23.207628249046.dcm',
                                  'TE_20_SlcPos_-2.2076272953718.dcm'
                                  )
                      ]

    def _chk(self):
        assert_raises(dcmstack.InvalidStackError,
                      self.stack.get_shape)
        assert_raises(dcmstack.InvalidStackError,
                      self.stack.get_affine)
        assert_raises(dcmstack.InvalidStackError,
                      self.stack.get_data)
        assert_raises(dcmstack.InvalidStackError,
                      self.stack.to_nifti)

    def test_empty(self):
        self.stack = dcmstack.DicomStack()
        self._chk()

    def test_only_dummy(self):
        self.stack = dcmstack.DicomStack(allow_dummies=True)
        del self.inputs[0].Rows
        del self.inputs[0].Columns
        del self.inputs[1].Rows
        del self.inputs[1].Columns
        self.stack.add_dcm(self.inputs[0])
        self.stack.add_dcm(self.inputs[1])
        self._chk()

    def test_missing_slice(self):
        self.stack = dcmstack.DicomStack()
        self.stack.add_dcm(self.inputs[0])
        self.stack.add_dcm(self.inputs[1])
        self.stack.add_dcm(self.inputs[4])
        self._chk()

    def test_wrong_number_of_files(self):
        self.stack = dcmstack.DicomStack(time_order='EchoTime')
        self.stack.add_dcm(self.inputs[0])
        self.stack.add_dcm(self.inputs[1])
        self.stack.add_dcm(self.inputs[2])
        self._chk()

    def test_vector_var_over_vol(self):
        self.stack = dcmstack.DicomStack(vector_order='EchoTime')
        self.stack.add_dcm(self.inputs[0])
        self.stack.add_dcm(self.inputs[1])
        self.stack.add_dcm(self.inputs[2])
        self.stack.add_dcm(self.inputs[3])
        self._chk()

class TestGetShape(object):
    def setUp(self):
        data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [pydicom.read_file(path.join(data_dir, fn))
                       for fn in ('TE_40_SlcPos_-33.707626341697.dcm',
                                  'TE_40_SlcPos_-23.207628249046.dcm',
                                  'TE_60_SlcPos_-33.707626341697.dcm',
                                  'TE_60_SlcPos_-23.207628249046.dcm',
                                  )
                      ]

    def test_single_slice(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        shape = stack.shape
        eq_(shape, (192, 192, 1))

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        shape = stack.shape
        eq_(shape, (192, 192, 2))

    def test_four_dim(self):
        stack = dcmstack.DicomStack(time_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        shape = stack.shape
        eq_(shape, (192, 192, 2, 2))

    def test_five_dim(self):
        stack = dcmstack.DicomStack(vector_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        shape = stack.shape
        eq_(shape, (192, 192, 2, 1, 2))

    def test_allow_dummy(self):
        del self.inputs[0].Rows
        del self.inputs[0].Columns
        stack = dcmstack.DicomStack(allow_dummies=True)
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        shape = stack.shape
        eq_(shape, (192, 192, 2))

class TestGuessDim(object):
    def setUp(self):
        data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [pydicom.read_file(path.join(data_dir, fn))
                       for fn in ('TE_40_SlcPos_-33.707626341697.dcm',
                                  'TE_40_SlcPos_-23.207628249046.dcm',
                                  'TE_60_SlcPos_-33.707626341697.dcm',
                                  'TE_60_SlcPos_-23.207628249046.dcm',
                                  )
                      ]
        for in_dcm in self.inputs:
            for key in dcmstack.DicomStack.sort_guesses:
                if hasattr(in_dcm, key):
                    delattr(in_dcm, key)

    def _get_vr_ord(self, key, ordinate):
        tag = keyword_dict[key]
        vr = dictionary_VR(tag)
        if vr == 'TM':
            return '%06d.000000' % ordinate
        else:
            return ordinate

    def test_single_guess(self):
        #Test situations where there is only one possible correct guess
        for key in dcmstack.DicomStack.sort_guesses:
            stack = dcmstack.DicomStack()
            for idx, in_dcm in enumerate(self.inputs):
                setattr(in_dcm, key, self._get_vr_ord(key, idx))
                stack.add_dcm(in_dcm)
            eq_(stack.shape, (192, 192, 2, 2))
            for in_dcm in self.inputs:
                delattr(in_dcm, key)

    def test_wrong_guess_first(self):
        #Test situations where the initial guesses are wrong
        stack = dcmstack.DicomStack()
        for key in dcmstack.DicomStack.sort_guesses[:-1]:
            for in_dcm in self.inputs:
                setattr(in_dcm, key, self._get_vr_ord(key, 0))
        for idx, in_dcm in enumerate(self.inputs):
            setattr(in_dcm,
                    dcmstack.DicomStack.sort_guesses[-1],
                    self._get_vr_ord(key, idx) )
            stack.add_dcm(in_dcm)
        eq_(stack.shape, (192, 192, 2, 2))

class TestGetData(object):
    def setUp(self):
        data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [pydicom.read_file(path.join(data_dir, fn))
                       for fn in ('TE_40_SlcPos_-33.707626341697.dcm',
                                  'TE_40_SlcPos_-23.207628249046.dcm',
                                  'TE_60_SlcPos_-33.707626341697.dcm',
                                  'TE_60_SlcPos_-23.207628249046.dcm',
                                  )
                      ]

    def test_single_slice(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        data = stack.get_data()
        eq_(data.shape, stack.shape)
        eq_(sha256(data).hexdigest(),
            '15cfa107ca73810a1c97f1c1872a7a4a05808ba6147e039cef3f63fa08735f5d')

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        data = stack.get_data()
        eq_(data.shape, stack.shape)
        eq_(sha256(data).hexdigest(),
            'ab5225fdbedceeea3442b2c9387e1abcbf398c71f525e0017251849c3cfbf49c')

    def test_four_dim(self):
        stack = dcmstack.DicomStack(time_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        data = stack.get_data()
        eq_(data.shape, stack.shape)
        eq_(sha256(data).hexdigest(),
            'bb3639a6ece13dc9a11d65f1b09ab3ccaed63b22dcf0f96fb5d3dd8805cc7b8a')

    def test_five_dim(self):
        stack = dcmstack.DicomStack(vector_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        data = stack.get_data()
        eq_(data.shape, stack.shape)
        eq_(sha256(data).hexdigest(),
            'bb3639a6ece13dc9a11d65f1b09ab3ccaed63b22dcf0f96fb5d3dd8805cc7b8a')

    def test_allow_dummy(self):
        del self.inputs[0].Rows
        del self.inputs[0].Columns
        stack = dcmstack.DicomStack(allow_dummies=True)
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        data = stack.get_data()
        eq_(data.shape, stack.shape)
        ok_(np.all(data[:, :, -1] == np.iinfo(np.int16).max))
        eq_(sha256(data).hexdigest(),
            '7d85fbcb60a5021a45df3975613dcb7ac731830e0a268590cc798dc39897c04b')

class TestGetAffine(object):
    def setUp(self):
        self.data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [pydicom.read_file(path.join(self.data_dir, fn))
                       for fn in ('TE_20_SlcPos_-33.707626341697.dcm',
                                  'TE_20_SlcPos_-23.207628249046.dcm'
                                 )
                      ]

    def test_single_slice(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        affine = stack.affine
        ref = np.load(path.join(self.data_dir, 'single_slice_aff.npy'))
        ok_(np.allclose(affine, ref))

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        affine = stack.affine
        ref = np.load(path.join(self.data_dir, 'single_vol_aff.npy'))
        ok_(np.allclose(affine, ref))

class TestToNifti(object):

    eq_keys = ['sizeof_hdr',
               'data_type',
               'extents',
               'dim_info',
               'dim',
               'intent_p1',
               'intent_p2',
               'intent_p3',
               'intent_code',
               'datatype',
               'bitpix',
               'slice_start',
               'pixdim',
               'scl_slope',
               'scl_inter',
               'slice_end',
               'slice_code',
               'xyzt_units',
              ]

    close_keys = ['cal_max',
                  'cal_min',
                  'slice_duration',
                  'toffset',
                  'glmax',
                  'glmin',
                  'qform_code',
                  'sform_code',
                  'quatern_b',
                  'quatern_c',
                  'quatern_d',
                  'qoffset_x',
                  'qoffset_y',
                  'qoffset_z',
                  'srow_x',
                  'srow_y',
                  'srow_z',
                 ]

    def setUp(self):
        self.data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [pydicom.read_file(path.join(self.data_dir, fn))
                       for fn in ('TE_20_SlcPos_-33.707626341697.dcm',
                                  'TE_20_SlcPos_-23.207628249046.dcm',
                                  'TE_40_SlcPos_-33.707626341697.dcm',
                                  'TE_40_SlcPos_-23.207628249046.dcm',
                                 )
                      ]

    def _build_nii(self, name):
        kwargs = {}
        if name.endswith('_meta'):
            kwargs['embed_meta'] = True
            name = name[:-5]
        if name.endswith('_SAR'):
            kwargs['voxel_order'] = 'SAR'
            name = name[:-4]
        if name == 'two_time_vol':
            stack = dcmstack.DicomStack(time_order='EchoTime')
        elif name == 'two_vector_vol':
            stack = dcmstack.DicomStack(vector_order='EchoTime')
        else:
            stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        if name == 'single_slice':
            return stack.to_nifti(**kwargs)
        stack.add_dcm(self.inputs[1])
        if name == 'single_vol':
            return stack.to_nifti(**kwargs)
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        if name in ('two_time_vol', 'two_vector_vol'):
            return stack.to_nifti(**kwargs)
        assert False # Unknown name

    def _chk(self, nii, ref_base_fn):
        hdr = nii.header
        ref_nii = nb.load(path.join(self.data_dir, ref_base_fn) + '.nii.gz')
        ref_hdr = ref_nii.header

        for key in self.eq_keys:
            print("Testing key %s" % key)
            v1 = hdr[key]
            v2 = ref_hdr[key]
            try:
                np.testing.assert_equal(v1, v2)
            except AssertionError:
                if key == 'slice_code':
                    warnings.warn(
                        "Random failure due to a slim test volume and absent "
                        "information on slice ordering.  "
                        "See https://github.com/nipy/nibabel/pull/647"
                    )
                    continue
                raise

        for key in self.close_keys:
            print("Testing key %s" % key)
            ok_(np.allclose(hdr[key], ref_hdr[key]))

    def test_single_slice(self):
        for tst in ('single_slice', 'single_slice_meta'):
            nii = self._build_nii(tst)
            self._chk(nii, tst)

    def test_single_vol(self):
        for tst in ('single_vol', 'single_vol_meta'):
            nii = self._build_nii(tst)
            self._chk(nii, tst)

    def test_slice_dim_reorient(self):
        for tst in ('single_vol_SAR', 'single_vol_SAR_meta'):
            nii = self._build_nii(tst)
            self._chk(nii, tst)

    def test_two_time_vol(self):
        for tst in ('two_time_vol', 'two_time_vol_meta'):
            nii = self._build_nii(tst)
            self._chk(nii, tst)

    def test_two_vector_vol(self):
        for tst in ('two_vector_vol', 'two_vector_vol_meta'):
            nii = self._build_nii(tst)
            self._chk(nii, tst)

    def test_allow_dummies(self):
        del self.inputs[0].Rows
        del self.inputs[0].Columns
        stack = dcmstack.DicomStack(allow_dummies=True)
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        nii = stack.to_nifti()
        data = nii.get_data()
        ok_(np.all(data[:, :, 0] == np.iinfo(np.int16).max))


class TestParseAndGroup(object):
    def setUp(self):
        self.data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.in_paths = [path.join(self.data_dir, fn)
                         for fn in ('TE_20_SlcPos_-33.707626341697.dcm',
                                    'TE_20_SlcPos_-23.207628249046.dcm',
                                    'TE_40_SlcPos_-33.707626341697.dcm',
                                    'TE_40_SlcPos_-23.207628249046.dcm',
                                   )
                        ]

    def test_default(self):
        res = dcmstack.parse_and_group(self.in_paths)
        eq_(len(res), 1)
        ds = pydicom.read_file(self.in_paths[0])
        group_key = list(res.keys())[0]
        for attr_idx, attr in enumerate(dcmstack.default_group_keys):
            if attr in dcmstack.default_close_keys:
                ok_(np.allclose(group_key[attr_idx], getattr(ds, attr)))
            else:
                eq_(group_key[attr_idx], getattr(ds, attr))


class TestParseAndStack(object):
    def setUp(self):
        self.data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.in_paths = [path.join(self.data_dir, fn)
                         for fn in ('TE_20_SlcPos_-33.707626341697.dcm',
                                    'TE_20_SlcPos_-23.207628249046.dcm',
                                    'TE_40_SlcPos_-33.707626341697.dcm',
                                    'TE_40_SlcPos_-23.207628249046.dcm',
                                   )
                        ]

    def test_default(self):
        res = dcmstack.parse_and_stack(self.in_paths)
        eq_(len(res), 1)
        ds = pydicom.read_file(self.in_paths[0])
        group_key = list(res.keys())[0]
        for attr_idx, attr in enumerate(dcmstack.default_group_keys):
            if attr in dcmstack.default_close_keys:
                ok_(np.allclose(group_key[attr_idx], getattr(ds, attr)))
            else:
                eq_(group_key[attr_idx], getattr(ds, attr))
        stack = list(res.values())[0]
        ok_(isinstance(stack, dcmstack.DicomStack))
        stack_data = stack.get_data()
        eq_(stack_data.ndim, 4)


def test_fsl_hack():
    ds = make_dicom({'BitsStored': 14, }, 2**14 - 1)
    stack = dcmstack.DicomStack()
    stack.add_dcm(ds)
    data = stack.get_data()    
    eq_(np.max(data), (2**14 - 1))
    eq_(data.dtype, np.int16)


def test_pix_overflow():
    ds = make_dicom(pix_val=(2**16 - 1))
    stack = dcmstack.DicomStack()
    stack.add_dcm(ds)
    data = stack.get_data()    
    eq_(np.max(data), (2**16 - 1))
    eq_(data.dtype, np.uint16)
