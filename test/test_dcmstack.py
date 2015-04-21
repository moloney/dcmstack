"""
Tests for dcmstack.dcmstack
"""
import sys
from os import path
from glob import glob
from hashlib import sha256
from nose.tools import ok_, eq_, assert_raises
from copy import deepcopy
import numpy as np
import dicom
from dicom import datadict
import nibabel as nb
from nibabel.orientations import aff2axcodes

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
    dcm = dicom.read_file(dcm_path)
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
        self.dcm = dicom.read_file(dcm_path)

        self.stack = dcmstack.DicomStack()
        self.stack.add_dcm(self.dcm)
        self.dcm = dicom.read_file(dcm_path)

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
        self.inputs = [dicom.read_file(path.join(data_dir, fn))
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
        self.inputs = [dicom.read_file(path.join(data_dir, fn))
                       for fn in ('TE_40_SlcPos_-33.707626341697.dcm',
                                  'TE_40_SlcPos_-23.207628249046.dcm',
                                  'TE_60_SlcPos_-33.707626341697.dcm',
                                  'TE_60_SlcPos_-23.207628249046.dcm',
                                  )
                      ]

    def test_single_slice(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        shape = stack.get_shape()
        eq_(shape, (192, 192, 1))

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        shape = stack.get_shape()
        eq_(shape, (192, 192, 2))

    def test_four_dim(self):
        stack = dcmstack.DicomStack(time_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        shape = stack.get_shape()
        eq_(shape, (192, 192, 2, 2))

    def test_five_dim(self):
        stack = dcmstack.DicomStack(vector_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        shape = stack.get_shape()
        eq_(shape, (192, 192, 2, 1, 2))

    def test_allow_dummy(self):
        del self.inputs[0].Rows
        del self.inputs[0].Columns
        stack = dcmstack.DicomStack(allow_dummies=True)
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        shape = stack.get_shape()
        eq_(shape, (192, 192, 2))

class TestGuessDim(object):
    def setUp(self):
        data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [dicom.read_file(path.join(data_dir, fn))
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
        tag = datadict.keyword_dict[key]
        vr = datadict.dictionaryVR(tag)
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
            eq_(stack.get_shape(), (192, 192, 2, 2))
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
        eq_(stack.get_shape(), (192, 192, 2, 2))

class TestGetData(object):
    def setUp(self):
        data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [dicom.read_file(path.join(data_dir, fn))
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
        eq_(data.shape, stack.get_shape())
        eq_(sha256(data).hexdigest(),
            '15cfa107ca73810a1c97f1c1872a7a4a05808ba6147e039cef3f63fa08735f5d')

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        data = stack.get_data()
        eq_(data.shape, stack.get_shape())
        eq_(sha256(data).hexdigest(),
            'ab5225fdbedceeea3442b2c9387e1abcbf398c71f525e0017251849c3cfbf49c')

    def test_four_dim(self):
        stack = dcmstack.DicomStack(time_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        data = stack.get_data()
        eq_(data.shape, stack.get_shape())
        eq_(sha256(data).hexdigest(),
            'bb3639a6ece13dc9a11d65f1b09ab3ccaed63b22dcf0f96fb5d3dd8805cc7b8a')

    def test_five_dim(self):
        stack = dcmstack.DicomStack(vector_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        data = stack.get_data()
        eq_(data.shape, stack.get_shape())
        eq_(sha256(data).hexdigest(),
            'bb3639a6ece13dc9a11d65f1b09ab3ccaed63b22dcf0f96fb5d3dd8805cc7b8a')

    def test_allow_dummy(self):
        del self.inputs[0].Rows
        del self.inputs[0].Columns
        stack = dcmstack.DicomStack(allow_dummies=True)
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        data = stack.get_data()
        eq_(data.shape, stack.get_shape())
        ok_(np.all(data[:, :, -1] == np.iinfo(np.int16).max))
        eq_(sha256(data).hexdigest(),
            '7d85fbcb60a5021a45df3975613dcb7ac731830e0a268590cc798dc39897c04b')

class TestGetAffine(object):
    def setUp(self):
        self.data_dir = path.join(test_dir,
                             'data',
                             'dcmstack',
                             '2D_16Echo_qT2')
        self.inputs = [dicom.read_file(path.join(self.data_dir, fn))
                       for fn in ('TE_20_SlcPos_-33.707626341697.dcm',
                                  'TE_20_SlcPos_-23.207628249046.dcm'
                                 )
                      ]

    def test_single_slice(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        affine = stack.get_affine()
        ref = np.load(path.join(self.data_dir, 'single_slice_aff.npy'))
        ok_(np.allclose(affine, ref))

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        affine = stack.get_affine()
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
        self.inputs = [dicom.read_file(path.join(self.data_dir, fn))
                       for fn in ('TE_20_SlcPos_-33.707626341697.dcm',
                                  'TE_20_SlcPos_-23.207628249046.dcm',
                                  'TE_40_SlcPos_-33.707626341697.dcm',
                                  'TE_40_SlcPos_-23.207628249046.dcm',
                                 )
                      ]

    def _chk(self, nii, ref_base_fn):
        hdr = nii.get_header()
        ref_nii = nb.load(path.join(self.data_dir, ref_base_fn) + '.nii.gz')
        ref_hdr = ref_nii.get_header()

        for key in self.eq_keys:
            np.testing.assert_equal(hdr[key], ref_hdr[key])

        for key in self.close_keys:
            ok_(np.allclose(hdr[key], ref_hdr[key]))

    def test_single_slice(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        nii = stack.to_nifti()
        self._chk(nii, 'single_slice')

    def test_single_vol(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        nii = stack.to_nifti()
        self._chk(nii, 'single_vol')

    def test_slice_dim_reorient(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        nii = stack.to_nifti(voxel_order='SAR')
        self._chk(nii, 'single_vol_SAR')

    def test_two_time_vol(self):
        stack = dcmstack.DicomStack(time_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        nii = stack.to_nifti()
        self._chk(nii, 'two_time_vol')

    def test_two_vector_vol(self):
        stack = dcmstack.DicomStack(vector_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        nii = stack.to_nifti()
        self._chk(nii, 'two_vector_vol')

    def test_allow_dummies(self):
        del self.inputs[0].Rows
        del self.inputs[0].Columns
        stack = dcmstack.DicomStack(allow_dummies=True)
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        nii = stack.to_nifti()
        data = nii.get_data()
        ok_(np.all(data[:, :, 0] == np.iinfo(np.int16).max))

