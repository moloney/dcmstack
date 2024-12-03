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
import pytest

from . import test_dir, src_dir

try:
    import pydicom
    import pydicom.dataset
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

if hasattr(pydicom.dataset, "FileMetaDataset"):
    _def_file_meta = pydicom.dataset.FileMetaDataset()
else:
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
    if attrs is not None:
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
        ds.PixelData = arr.tobytes()
    return ds

def test_key_regex_filter():
        filt = dcmstack.make_key_regex_filter(['test', 'another'],
                                              ['2', 'another test'])
        assert(filt('test', 1))
        assert(filt('test another', 1))
        assert(filt('another tes', 1))
        assert(not filt('test2', 1))
        assert(not filt('2 another', 1))
        assert(not filt('another test', 1))

class TestReorderVoxels(object):
    def setup_method(self, method):
        self.vox_array = np.arange(16).reshape((2, 2, 2, 2))
        self.affine = np.eye(4)

    def test_invalid_vox_order(self):
        with pytest.raises(ValueError):
            dcmstack.reorder_voxels(self.vox_array, self.affine, 'lra')
        with pytest.raises(ValueError):
            dcmstack.reorder_voxels(self.vox_array, self.affine, 'rpil')
        with pytest.raises(ValueError):
            dcmstack.reorder_voxels(self.vox_array, self.affine, 'lrz')

    def test_invalid_vox_array(self):
        with pytest.raises(ValueError):
            dcmstack.reorder_voxels(np.eye(2), self.affine, 'rpi')

    def test_invalid_affine(self):
        with pytest.raises(ValueError):
            dcmstack.reorder_voxels(self.vox_array, np.eye(3), 'rpi')

    def test_no_op(self):
        vox_order = ''.join(aff2axcodes(self.affine))
        (vox_array,
         affine,
         aff_trans,
         ornt_trans) = dcmstack.reorder_voxels(self.vox_array,
                                               self.affine,
                                               vox_order)
        assert((vox_array == self.vox_array).all())
        assert((affine == self.affine).all())
        assert((aff_trans == np.eye(4)).all())
        assert(np.allclose(ornt_trans, [[0, 1], [1, 1], [2, 1]]))
        assert np.may_share_memory(affine, self.affine) == False

    def test_reorder(self):
        (vox_array,
         affine,
         aff_trans,
         ornt_trans) = dcmstack.reorder_voxels(self.vox_array,
                                               self.affine,
                                               'PRS')
        assert(np.all(vox_array == np.array([[[[4, 5],
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
        assert(np.allclose(affine,
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
        assert(np.allclose(affine,
                        np.array([[0,-1,0,1],
                                  [-1,0,0,3],
                                  [0,0,1,0],
                                  [0,0,0,1]])
                       )
           )


class TestDicomOrdering(object):
    class _MockNiiWrp:
        def __init__(self, meta_dict):
            self._meta_dict = meta_dict
        
        def get_meta(self, key):
            return self._meta_dict.get(key)
        
    def setup_method(self, method):
        self.meta = {'EchoTime' : 2}
        self.nw = TestDicomOrdering._MockNiiWrp(self.meta)

    def test_missing_key(self):
        ordering = dcmstack.DicomOrdering('blah')
        assert ordering.get_ordinate(self.nw) == None

    def test_non_abs(self):
        ordering = dcmstack.DicomOrdering('EchoTime')
        assert ordering.get_ordinate(self.nw) == self.meta['EchoTime']

    def test_abs(self):
        abs_order = [1,2,3]
        ordering = dcmstack.DicomOrdering('EchoTime', abs_ordering=abs_order)
        assert ordering.get_ordinate(self.nw) == abs_order.index(self.meta['EchoTime'])

    def test_abs_as_str(self):
        abs_order = ['1','2','3']
        ordering = dcmstack.DicomOrdering('EchoTime',
                                          abs_ordering=abs_order,
                                          abs_as_str=True)
        assert ordering.get_ordinate(self.nw) == abs_order.index(str(self.meta['EchoTime']))

    def test_abs_missing(self):
        abs_order = [1,3]
        ordering = dcmstack.DicomOrdering('EchoTime', abs_ordering=abs_order)
        with pytest.raises(ValueError):
            ordering.get_ordinate(self.nw)

def test_image_collision():
    dcm_path = path.join(test_dir,
                         'data',
                         'dcmstack',
                         '2D_16Echo_qT2',
                         'TE_20_SlcPos_-33.707626341697.dcm')
    dcm = pydicom.read_file(dcm_path)
    stack = dcmstack.DicomStack('EchoTime')
    stack.add_dcm(dcm)
    with pytest.raises(dcmstack.ImageCollisionError):
        stack.add_dcm(dcm)


class TestIncongruentImage(object):
    def setup_method(self, method):
        self.dcm = make_dicom()
        self.stack = dcmstack.DicomStack()
        self.stack.add_dcm(self.dcm)

    def _chk_raises(self, ds):
        with pytest.raises(dcmstack.IncongruentImageError):
            self.stack.add_dcm(ds)

    def test_rows(self):
        ds = make_dicom({"Rows": self.dcm.Rows + 1})
        self._chk_raises(ds)

    def test_columns(self):
        ds = make_dicom({"Columns": self.dcm.Columns + 1})
        self._chk_raises(ds)

    def test_pix_space(self):
        ds = make_dicom(
            {"PixelSpacing": [self.dcm.PixelSpacing[0] * 2, self.dcm.PixelSpacing[1]]}
        )
        self._chk_raises(ds)

    def test_close_pix_space(self):
        ds = make_dicom(
            {"PixelSpacing": [self.dcm.PixelSpacing[0] + 1e-7, self.dcm.PixelSpacing[1]]}
        )
        # Shouldn't raise
        self.stack.add_dcm(ds)

    def test_orientation(self):
        ds = make_dicom({"ImageOrientationPatient": [1., 0., 0., 0., 0., 1.]})
        self._chk_raises(ds)

    def test_close_orientation(self):
        ds = make_dicom({"ImageOrientationPatient": [1. - 1e-8, 0., 0., 0., 1., 0.]})
        # Shouldn't raise
        self.stack.add_dcm(ds)


class TestInvalidStack(object):
    def setup_method(self, method):
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
        with pytest.raises(dcmstack.InvalidStackError):
            self.stack.get_shape()
        with pytest.raises(dcmstack.InvalidStackError):
                      self.stack.get_affine()
        with pytest.raises(dcmstack.InvalidStackError):
                      self.stack.get_data()
        with pytest.raises(dcmstack.InvalidStackError):
                      self.stack.to_nifti()

    def test_empty(self):
        self.stack = dcmstack.DicomStack()
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
    def setup_method(self, method):
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
        assert shape == (192, 192, 1)

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        shape = stack.shape
        assert shape == (192, 192, 2)

    def test_four_dim(self):
        stack = dcmstack.DicomStack(time_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        shape = stack.shape
        assert shape == (192, 192, 2, 2)

    def test_five_dim(self):
        stack = dcmstack.DicomStack(vector_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        shape = stack.shape
        assert shape == (192, 192, 2, 1, 2)

class TestGuessDim(object):
    def setup_method(self, method):
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
            for key in dcmstack.DicomStack.SORT_GUESSES:
                if hasattr(in_dcm, key):
                    delattr(in_dcm, key)

    def _get_vr_ord(self, key, ordinate):
        try:
            tag = keyword_dict[key]
        except KeyError:
            return ordinate
        vr = dictionary_VR(tag)
        if vr == 'TM':
            return '%06d.000000' % ordinate
        else:
            return ordinate

    def test_single_guess(self):
        #Test situations where there is only one possible correct guess
        for key in dcmstack.DicomStack.SORT_GUESSES:
            stack = dcmstack.DicomStack()
            for idx, in_dcm in enumerate(self.inputs):
                setattr(in_dcm, key, self._get_vr_ord(key, idx))
                stack.add_dcm(in_dcm)
            assert stack.shape == (192, 192, 2, 2)
            for in_dcm in self.inputs:
                delattr(in_dcm, key)

    def test_wrong_guess_first(self):
        #Test situations where the initial guesses are wrong
        stack = dcmstack.DicomStack()
        for key in dcmstack.DicomStack.SORT_GUESSES[:-1]:
            for in_dcm in self.inputs:
                setattr(in_dcm, key, self._get_vr_ord(key, 0))
        for idx, in_dcm in enumerate(self.inputs):
            setattr(in_dcm,
                    dcmstack.DicomStack.SORT_GUESSES[-1],
                    self._get_vr_ord(key, idx) )
            stack.add_dcm(in_dcm)
        assert stack.shape == (192, 192, 2, 2)

class TestGetData(object):
    def setup_method(self, method):
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
        assert data.shape == stack.shape
        assert sha256(data).hexdigest() == \
            '15cfa107ca73810a1c97f1c1872a7a4a05808ba6147e039cef3f63fa08735f5d'

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        data = stack.get_data()
        assert data.shape == stack.shape
        assert sha256(data).hexdigest() == \
            'ab5225fdbedceeea3442b2c9387e1abcbf398c71f525e0017251849c3cfbf49c'

    def test_four_dim(self):
        stack = dcmstack.DicomStack(time_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        data = stack.get_data()
        assert data.shape == stack.shape
        assert sha256(data).hexdigest() == \
            'bb3639a6ece13dc9a11d65f1b09ab3ccaed63b22dcf0f96fb5d3dd8805cc7b8a'

    def test_five_dim(self):
        stack = dcmstack.DicomStack(vector_order='EchoTime')
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        stack.add_dcm(self.inputs[2])
        stack.add_dcm(self.inputs[3])
        data = stack.get_data()
        assert data.shape == stack.shape
        assert sha256(data).hexdigest() == \
            'bb3639a6ece13dc9a11d65f1b09ab3ccaed63b22dcf0f96fb5d3dd8805cc7b8a'
    
    def test_data_scaling(self):
        inputs = deepcopy(self.inputs)
        for input in inputs:
            input.RescaleSlope = 10.0
        stack = dcmstack.DicomStack(vector_order='EchoTime')
        stack.add_dcm(inputs[0])
        stack.add_dcm(inputs[1])
        stack.add_dcm(inputs[2])
        stack.add_dcm(inputs[3])
        data = stack.get_data()
        unscl_data = stack.get_data(scaled=False)
        assert data.dtype == np.float32
        assert unscl_data.dtype == np.int16
        assert np.allclose(data, unscl_data*10.0)


class TestGetAffine(object):
    def setup_method(self, method):
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
        assert(np.allclose(affine, ref))

    def test_three_dim(self):
        stack = dcmstack.DicomStack()
        stack.add_dcm(self.inputs[0])
        stack.add_dcm(self.inputs[1])
        affine = stack.affine
        ref = np.load(path.join(self.data_dir, 'single_vol_aff.npy'))
        assert(np.allclose(affine, ref))

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

    def setup_method(self, method):
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
            assert(np.allclose(hdr[key], ref_hdr[key]))

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


class TestParseAndGroup(object):
    def setup_method(self, method):
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
        assert len(res) == 1
        ds = pydicom.read_file(self.in_paths[0])
        group_key = list(res.keys())[0]
        for attr_idx, attr in enumerate(dcmstack.DEFAULT_GROUP_KEYS):
            if attr in dcmstack.DEFAULT_CLOSE_KEYS:
                assert(np.allclose(group_key[attr_idx], getattr(ds, attr)))
            else:
                assert group_key[attr_idx] == getattr(ds, attr)


class TestParseAndStack(object):
    def setup_method(self, method):
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
        assert len(res) == 1
        ds = pydicom.read_file(self.in_paths[0])
        group_key = list(res.keys())[0]
        for attr_idx, attr in enumerate(dcmstack.DEFAULT_GROUP_KEYS):
            if attr in dcmstack.DEFAULT_CLOSE_KEYS:
                assert(np.allclose(group_key[attr_idx], getattr(ds, attr)))
            else:
                assert group_key[attr_idx] == getattr(ds, attr)
        stack = list(res.values())[0]
        assert(isinstance(stack, dcmstack.DicomStack))
        stack_data = stack.get_data()
        assert stack_data.ndim == 4


def test_fsl_hack():
    ds = make_dicom({'BitsStored': 14, }, 2**14 - 1)
    stack = dcmstack.DicomStack()
    stack.add_dcm(ds)
    data = stack.get_data()    
    assert np.max(data) == (2**14 - 1)
    assert data.dtype == np.int16


def test_pix_overflow():
    ds = make_dicom(pix_val=(2**16 - 1))
    stack = dcmstack.DicomStack()
    stack.add_dcm(ds)
    data = stack.get_data()    
    assert np.max(data) == (2**16 - 1)
    assert data.dtype == np.uint16
