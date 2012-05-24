"""
Stack DICOM datasets into volumes. The contents of this module are imported 
into the package namespace.
"""
import warnings, re, dicom
from copy import deepcopy
import nibabel as nb
from nibabel.nifti1 import Nifti1Extensions
from nibabel.spatialimages import HeaderDataError
import numpy as np
from .dcmmeta import DcmMetaExtension, NiftiWrapper

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from .extract import default_extractor

def make_key_regex_filter(exclude_res, force_include_res=None):
    '''Make a meta data filter using regular expressions.

    Parameters
    ----------
    exclude_res : sequence
        Sequence of regular expression strings. Any meta data where the key 
        matches one of these expressions will be excluded, unless it matches 
        one of the `force_include_res`.
    force_include_res : sequence
        Sequence of regular expression strings. Any meta data where the key 
        matches one of these expressions will be included.
        
    Returns
    -------
    A callable which can be passed to `DicomStack` as the `meta_filter`.
    '''
    exclude_re = re.compile('|'.join(['(?:' + regex + ')' 
                                      for regex in exclude_res])
                           )
    include_re = None
    if force_include_res:
        include_re = re.compile('|'.join(['(?:' + regex + ')' 
                                          for regex in force_include_res])
                               )

    def key_regex_filter(key, value):
        return (exclude_re.search(key) and 
                not (include_re and include_re.search(key)))
    return key_regex_filter

default_key_excl_res = ['Patient',
                        'Physician',
                        'Operator',
                        'Date', 
                        'Birth',
                        'Address',
                        'Institution',
                        'SiteName',
                        'Age',
                        'Comment',
                        'Phone',
                        'Telephone',
                        'Insurance',
                        'Religious',
                        'Language',
                        'Military',
                        'MedicalRecord',
                        'Ethnic',
                        'Occupation',
                        'Unknown',
                        'PrivateTagData',
                       ]
'''A list of regexes passed to `make_key_regex_filter` as `exclude_res` to 
create the `default_meta_filter`.'''
                        
default_key_incl_res = ['ImageOrientationPatient',
                        'ImagePositionPatient',
                       ]
'''A list of regexes passed to `make_key_regex_filter` as `force_include_res` 
to create the `default_meta_filter`.'''
                        
default_meta_filter = make_key_regex_filter(default_key_excl_res,
                                            default_key_incl_res)
'''Default meta_filter for `DicomStack`.'''

def closest_ortho_pat_axis(direction):
    '''
    Determine the closest orthographic patient axis for the given direction.
    
    Parameters
    ----------
    direction : array
        A direction vector with three dimensions in Nifti patient space (RAS).
        
    Returns
    -------
    A two character code coresponding to the orientation in terms of the 
    closest orthogonal patient axis (eg. 'lr' would mean from (l)eft to 
    (r)ight).
    '''
    if (abs(direction[0]) >= abs(direction[1]) and 
        abs(direction[0]) >= abs(direction[2])):
        if direction[0] < 0.0:
            return 'rl'
        else:
            return 'lr'
    elif (abs(direction[1]) >= abs(direction[0]) and 
          abs(direction[1]) >= abs(direction[2])):
        if direction[1] < 0.0:
            return 'ap'
        else:
            return 'pa'
    elif (abs(direction[2]) >= abs(direction[0]) and 
          abs(direction[2]) >= abs(direction[1])):
        if direction[2] < 0.0:
            return 'si'
        else:
            return 'is'

def reorder_voxels(vox_array, affine, voxel_order):
    '''Reorder the given voxel array and corresponding affine. 
    
    Parameters
    ----------
    vox_array : array
        The array of voxel data
    
    affine : array
        The affine for mapping voxel indices to Nifti patient space
    
    voxel_order : str
        A three character code specifing the desired starting point for rows, 
        columns, and slices in terms of the orthogonal axes of patient space: 
        (l)eft, (r)ight, (a)nterior, (p)osterior, (s)uperior, and (i)nferior.

    Returns
    -------
    out_vox : array
        An updated view of vox_array.
        
    out_aff : array
        A new array with the updated affine
        
    perm : tuple
        A tuple with the permuted dimension indices
    '''
    #Take a copy of affine so that we return a new array even when nothing is 
    #changed
    affine = affine.copy()
    
    #Check if voxel_order is valid
    voxel_order = voxel_order.lower()
    if len(voxel_order) != 3:
        raise ValueError('The voxel_order must contain three characters')
    dcm_axes = ['lr', 'ap', 'si']
    for char in voxel_order:
        if not char in 'lrapis':
            raise ValueError('The characters in voxel_order must be one '
                             'of: l,r,a,p,i,s')
        for idx, axis in enumerate(dcm_axes):
            if char in axis:
                del dcm_axes[idx]
    if len(dcm_axes) != 0:
        raise ValueError('No character in voxel_order corresponding to '
                         'axes: %s' % dcm_axes)
    
    #Check the vox_array and affine have correct shape/size
    if len(vox_array.shape) < 3:
        raise ValueError('The vox_array must be at least three dimensional')
    if affine.shape != (4, 4):
        raise ValueError('The affine must be 4x4')
    
    #Pull the current index directions from the affine
    index_dirs = [affine[:3, 0].copy(),
                  affine[:3, 1].copy(),
                  affine[:3, 2].copy()
                 ]
    for i in range(3):
        index_dirs[i] /= np.sqrt(np.dot(index_dirs[i], index_dirs[i]))
    
    #Track some information about the three spatial axes
    axes = [{'ortho_axis' : closest_ortho_pat_axis(index_dirs[0]),
             'num_samples' : vox_array.shape[0],
             'orig_idx' : 0,
            },
            {'ortho_axis' : closest_ortho_pat_axis(index_dirs[1]),
             'num_samples' : vox_array.shape[1],
             'orig_idx' : 1
            },
            {'ortho_axis' : closest_ortho_pat_axis(index_dirs[2]),
             'num_samples' : vox_array.shape[2],
             'orig_idx' : 2,
            }
           ]
    
    #Reorder data as specifed by voxel_order, updating the affine
    slice_lst = []
    for dest_index, axis_char in enumerate(voxel_order):
        for orig_index, axis in enumerate(axes):
            if axis_char in axis['ortho_axis']:
                if dest_index != orig_index:
                    vox_array = vox_array.swapaxes(dest_index, orig_index)
                    swap_trans = np.eye(4)
                    swap_trans[:, orig_index], swap_trans[:, dest_index] = \
                        (swap_trans[:, dest_index].copy(), 
                         swap_trans[:, orig_index].copy())
                    affine = np.dot(affine, swap_trans)
                    axes[orig_index], axes[dest_index] = \
                        (axes[dest_index], axes[orig_index])
                if axis_char == axis['ortho_axis'][1]:
                    vox_array = vox_array[slice_lst + 
                                          [slice(None, None, -1), Ellipsis]]
                    rev_mat = np.eye(4)
                    rev_mat[dest_index, dest_index] = -1
                    rev_mat[dest_index,3] = (axis['num_samples']-1)
                    affine = np.dot(affine, rev_mat)
                    axis['ortho_axis'] = (axis['ortho_axis'][1] + 
                                          axis['ortho_axis'][0])
                break
        slice_lst.append(slice(None))

    permutation = (axes[0]['orig_idx'], 
                   axes[1]['orig_idx'], 
                   axes[2]['orig_idx'])

    return (vox_array, affine, permutation)

def dcm_time_to_sec(time_str):
    '''Convert a DICOM time value (value representation of 'TM') to the number 
    of seconds past midnight.
    
    Parameters
    ----------
    time_str : str
        The DICOM time value string
        
    Returns
    -------
    A floating point representing the number of seconds past midnight
    '''
    return ((int(time_str[:2]) * 3600) + 
            (int(time_str[2:4]) * 60) + 
            float(time_str[4:]))

class IncongruentImageError(Exception):
    '''An exception denoting that a DICOM with incorrect size or orientation 
    was passed to `DicomStack.add_dcm`.'''
    
    def __init__(self, msg):
        self.msg = msg
        
    def __str__(self):
        return 'The image is not congruent to the existing stack: %s' % self.msg

class ImageCollisionError(Exception):
    '''An exception denoting that a DICOM which collides with one already in 
    the stack was passed to a `DicomStack.add_dcm`.'''
    def __str__(self):
        return 'The image collides with one already in the stack'
        
class InvalidStackError(Exception):
    '''An exception denoting that a `DicomStack` is not currently valid'''
    def __init__(self, msg):
        self.msg = msg
    
    def __str__(self):
        return 'The DICOM stack is not valid: %s' % self.msg

class DicomOrdering(object):
    '''Object defining an ordering for a set of dicom datasets.'''
    
    def __init__(self, key, abs_ordering=None, abs_as_str=False):
        '''Create a DicomOrdering with the given DICOM element keyword. 
        
        Parameters
        ----------
        key : str
            The DICOM keyword to use for ordering the datasets
            
        abs_ordering : sequence
            A sequence specifying the absolute order for values corresponding 
            to the `key`. Instead of ordering by the value associated with the 
            `key`, the index of the value in this sequence will be used.
        
        abs_as_str : bool
            If true, the values will be converted to strings before looking up 
            the index in `abs_ordering`.
            
        '''
        self.key = key
        self.abs_ordering = abs_ordering
        self.abs_as_str = abs_as_str
        
    def get_ordinate(self, ds):
        '''Get the ordinate for the given DICOM data set.
        
        Parameters
        ----------
        ds : dict like
            The DICOM data set we want the ordinate of. Should allow 
            dict like access where DICOM keywords return the corresponing 
            value.
            
        Returns
        -------
        An ordinate for the data set. If `abs_ordering` is None then this will 
        just be the value for the keyword `key`. Otherwise it will be an 
        integer.        
        '''
        try:
            val = ds[self.key]
        except KeyError:
            return None
            
        if self.abs_ordering:
            if self.abs_as_str:
                val = str(val)
            return self.abs_ordering.index(val)
        
        return val

def _make_dummy(reference, meta):
    '''Make a "dummy" NiftiWrapper (no valid pixel data).'''
    #Create the dummy data array filled with largest representable value
    data = np.empty_like(reference.nii_img.get_data())
    data[...] = np.iinfo(data.dtype).max
    
    #Create the nifti image and set header data
    nii_img = nb.nifti1.Nifti1Image(data, None)
    hdr = nii_img.get_header()
    aff = reference.nii_img.get_header().get_best_affine()
    hdr.set_qform(aff, 'scanner')
    nii_img._affine = hdr.get_best_affine()
    hdr.set_xyzt_units('mm', 'sec')
    dim_info = {'freq' : None, 
                'phase' : None, 
                'slice' : 2
               }
    if 'InplanePhaseEncodingDirection' in meta:
        if meta['InplanePhaseEncodingDirection'] == 'ROW':
            dim_info['phase'] = 1
            dim_info['freq'] = 0
        else:
            dim_info['phase'] = 0
            dim_info['freq'] = 1
    hdr.set_dim_info(**dim_info)
    
    #Embed the meta data extension
    result = NiftiWrapper(nii_img, make_empty=True)
    result.meta_ext.get_class_dict(('global', 'const')).update(meta)
    
    return result

class DicomStack(object):
    '''Defines a method for stacking together DICOM data sets into a multi 
    dimensional volume. 
    
    Tailored towards creating NiftiImage output, but can also just create numpy 
    arrays. Can summarize all of the meta data from the input DICOM data sets 
    into a Nifti header extension (see `dcmmeta.DcmMetaExtension`).
    '''
    
    def __init__(self, time_order='AcquisitionTime', vector_order=None, 
                 allow_dummies=False, meta_filter=None):
        '''Initialize a DicomStack object. 
        
        Parameters
        ----------
        time_order : str or DicomOrdering
            The DICOM keyword or DicomOrdering object specifying how to order 
            the DICOM data sets along the time dimension.

        vector_order : str or DicomOrdering
            The DICOM keyword or DicomOrdering object specifying how to order 
            the DICOM data sets along the vector dimension.
        
        allow_dummies : bool
            If True then data sets without pixel data can be added to the stack.
            The "dummy" voxels will have the maximum representable value for 
            the datatype.
        
        meta_filter : callable
            A callable that takes a meta data key and value, and returns True if 
            that meta data element should be excluded from the DcmMeta extension.
        '''
        if isinstance(time_order, str):
            self._time_order = DicomOrdering(time_order)
        else:
            self._time_order = time_order
        if isinstance(vector_order, str):
            self._vector_order = DicomOrdering(vector_order)
        else:
            self._vector_order = vector_order
        
        if meta_filter is None:
            self._meta_filter = default_meta_filter
        else:
            self._meta_filter = meta_filter
        
        self._allow_dummies = allow_dummies
        
        #Sets all the state variables to their defaults
        self.clear()
        
    def _chk_equal(self, keys, meta1, meta2):
        for key in keys:
            if meta1[key] != meta2[key]:
                raise IncongruentImageError("%s does not match" % key)

    def _chk_congruent(self, meta):
        is_dummy = not 'Rows' in meta or not 'Columns' in meta
        if is_dummy and not self._allow_dummies:
            raise IncongruentImageError('Missing Rows/Columns')
            
        if not self._ref_input is None:
            self._chk_equal(('PixelSpacing', 
                             'ImageOrientationPatient'), 
                             meta, 
                             self._ref_input
                            )
            if not is_dummy:
                self._chk_equal(('Rows', 'Columns'), meta, self._ref_input)
        elif len(self._dummies) != 0:
            self._chk_equal(('PixelSpacing', 
                             'ImageOrientationPatient'), 
                            meta, 
                            self._dummies[0][0]
                           )
        return is_dummy
            
    
    def add_dcm(self, dcm, meta=None):
        '''Add a pydicom dataset to the stack. 
        
        Parameters
        ----------
        dcm : dicom.dataset.Dataset
            The data set being added to the stack
            
        meta : dict
            The extracted meta data for the DICOM data set `dcm`. If None 
            extract.default_extractor will be used.
        
        Raises
        ------
        IncongruentImageError
            The provided `dcm` does not match the orientation or dimensions of 
            those already in the stack.
            
        ImageCollisionError
            The provided `dcm` has the same slice location and time/vector 
            values.

        '''
        
        if meta is None:
            meta = default_extractor(dcm)

        is_dummy = self._chk_congruent(meta)
        
        self._phase_enc_dirs.add(meta.get('InPlanePhaseEncodingDirection'))
        self._repetition_times.add(meta.get('RepetitionTime'))
        
        #Pull the info used for sorting
        if 'CsaImage.SliceNormalVector' in meta:
            slice_dir = np.array(meta['CsaImage.SliceNormalVector'])
        else:
            slice_dir = np.cross(meta['ImageOrientationPatient'][:3],
                                 meta['ImageOrientationPatient'][3:],
                                )
        slice_pos = np.dot(slice_dir, 
                           np.array(meta['ImagePositionPatient']))
        self._slice_pos_vals.add(slice_pos)
        time_val = None
        if self._time_order:
            time_val = self._time_order.get_ordinate(meta)
        self._time_vals.add(time_val)
        vector_val = None
        if self._vector_order:
            vector_val = self._vector_order.get_ordinate(meta)
        self._vector_vals.add(vector_val)
        
        #Create a tuple with the sorting values
        sorting_tuple = (vector_val, time_val, slice_pos)
        
        #Raise exception if image collides with another already in the stack
        if sorting_tuple in self._sorting_tuples:
            raise ImageCollisionError()
        self._sorting_tuples.add(sorting_tuple)
        
        #Create a NiftiWrapper for this input if possible
        nii_wrp = None
        if not is_dummy:
            nii_wrp = NiftiWrapper.from_dicom(dcm, meta)
            if self._ref_input is None:
                #We don't have a reference input yet, use this one
                self._ref_input = nii_wrp
                #Convert any dummies that we have stashed previously
                for dummy_meta, dummy_tuple in self._dummies:
                    dummy_wrp = _make_dummy(self._ref_input, dummy_meta)
                    self._files_info.append((dummy_wrp, dummy_tuple))
        else:
            if self._ref_input is None:
                #We don't have a reference input, so stash the dummy for now
                self._dummies.append((meta, sorting_tuple))
            else:
                #Convert dummy using the reference input
                nii_wrp = _make_dummy(self._ref_input, meta)
        
        #If we made a NiftiWrapper add it to the stack
        if not nii_wrp is None:
            self._files_info.append((nii_wrp, sorting_tuple))
        
        #Set the dirty flags
        self._shape_dirty = True 
        self._meta_dirty = True
        
    def clear(self):
        '''Remove any DICOM datasets from the stack.'''
        self._slice_pos_vals = set()
        self._time_vals = set()
        self._vector_vals = set()
        self._sorting_tuples = set()
        
        self._phase_enc_dirs = set()
        self._repetition_times = set()
        
        self._dummies = []
        self._ref_input = None
        
        self._shape_dirty = True
        self._shape = None
    
        self._meta_dirty = True
        self._meta = None
        
        self._files_info = []
        
        
    def get_shape(self):
        '''Get the shape of the stack.
        
        Returns
        -------
        A tuple of integers giving the size of the dimensions of the stack.
        
        Raises
        ------
        InvalidStackError
            The stack is incomplete or invalid.
        '''
        #If the dirty flag is not set, return the cached value
        if not self._shape_dirty:
            return self._shape

        #We need at least one non-dummy file in the stack
        if len(self._files_info) == 0:
            raise InvalidStackError("No (non-dummy) files in the stack")
        
        #Figure out number of files and slices per volume
        files_per_vol = len(self._slice_pos_vals)
        
        #Simple check for an incomplete stack
        if len(self._files_info) % files_per_vol != 0:
            raise InvalidStackError("Number of files is not an even multiple "
                                    "of the number of unique slice positions.")
        num_volumes = len(self._files_info) / files_per_vol
        
        #Sort the files
        self._files_info.sort(key=lambda x: x[1])
        if files_per_vol > 1:
            for vol_idx in range(num_volumes):
                start_slice = vol_idx * files_per_vol
                end_slice = start_slice + files_per_vol
                self._files_info[start_slice:end_slice] = \
                    sorted(self._files_info[start_slice:end_slice], 
                           key=lambda x: x[1][-1])
        
        #Figure out the number of vector components and time points
        num_vec_comps = len(self._vector_vals)
        if num_vec_comps > num_volumes:
            raise InvalidStackError("Vector variable varies within volumes")
        if num_volumes % num_vec_comps != 0:
            raise InvalidStackError("Number of volumes not an even multiple "
                                    "of the number of vector components.")
        num_time_points = num_volumes / num_vec_comps
       
        #Do a more thorough check for completeness
        slice_positions = sorted(list(self._slice_pos_vals))
        for vec_idx in xrange(num_vec_comps):
            file_idx = vec_idx*num_time_points*files_per_vol
            curr_vec_val = self._files_info[file_idx][1][0]
            for time_idx in xrange(num_time_points):
                for slice_idx in xrange(files_per_vol):
                    file_idx = (vec_idx*num_time_points*files_per_vol + 
                                time_idx*files_per_vol + slice_idx)
                    file_info = self._files_info[file_idx]
                    if file_info[1][0] != curr_vec_val:
                        raise InvalidStackError("Not enough images with the " + 
                                                "vector value of " + 
                                                str(curr_vec_val))
                    if (file_info[1][2] != slice_positions[slice_idx]):
                        if (file_info[1][2] == slice_positions[slice_idx-1]):
                            error_msg = ["Duplicate slice position"]
                        else:
                            error_msg = ["Missing slice position"]
                        error_msg.append(" at slice index %d" % slice_idx)
                        if num_time_points > 1:
                            error_msg.append(' in time point %d' % time_idx)
                        if num_vec_comps > 1:
                            error_msg.append(' for vector component %s' % 
                                             str(curr_vec_val))
                        raise InvalidStackError(''.join(error_msg))
        
        #Stack appears to be valid, build the shape tuple
        file_shape = self._files_info[0][0].nii_img.get_shape()
        vol_shape = list(file_shape)
        if files_per_vol > 1:
            vol_shape[2] = files_per_vol 
        shape = vol_shape+ [num_time_points, num_vec_comps]
        if shape[4] == 1:
            shape = shape[:-1]
            if shape[3] == 1:
                shape = shape[:-1]
        self._shape = tuple(shape)
            
        self._shape_dirty = False
        return self._shape

    def get_data(self):
        '''Get an array of the voxel values.
        
        Returns
        -------
        A numpy array filled with values from the DICOM data sets' pixels.
        
        Raises
        ------
        InvalidStackError
            The stack is incomplete or invalid.
        '''
        #Create a numpy array for storing the voxel data
        stack_shape = self.get_shape()
        stack_shape = tuple(list(stack_shape) + ((5 - len(stack_shape)) * [1]))
        vox_array = np.empty(stack_shape, np.int16)        
        
        #Fill the array with data
        n_vols = 1
        if len(stack_shape) > 3:
            n_vols *= stack_shape[3]
        if len(stack_shape) > 4:
            n_vols *= stack_shape[4]
        files_per_vol = len(self._files_info) / n_vols
        file_shape = self._files_info[0][0].nii_img.get_shape()
        for vec_idx in range(stack_shape[4]):
            for time_idx in range(stack_shape[3]):
                if files_per_vol == 1 and file_shape[2] != 1:
                    file_idx = vec_idx*(stack_shape[3]) + time_idx
                    vox_array[:, :, :, time_idx, vec_idx] = \
                        self._files_info[file_idx][0].nii_img.get_data()
                else:
                    for slice_idx in range(files_per_vol):
                        file_idx = (vec_idx*(stack_shape[3]*stack_shape[2]) + 
                                    time_idx*(stack_shape[2]) + slice_idx)
                        vox_array[:, :, slice_idx, time_idx, vec_idx] = \
                            self._files_info[file_idx][0].nii_img.get_data()[:, :, 0]
        
        #Trim unused time/vector dimensions
        if stack_shape[4] == 1:
            vox_array = vox_array[...,0]
            if stack_shape[3] == 1:
                vox_array = vox_array[...,0]
        
        return vox_array
        
    def get_affine(self):
        '''Get the affine transform for mapping row/column/slice indices 
        to Nifti (RAS) patient space.
        
        Returns
        -------
        A 4x4 numpy array containing the affine transform.
        
        Raises
        ------
        InvalidStackError
            The stack is incomplete or invalid.
        '''
        #Figure out the number of three (or two) dimensional volumes
        shape = self.get_shape()
        n_vols = 1
        if len(shape) > 3:
            n_vols *= shape[3]
        if len(shape) > 4:
            n_vols *= shape[4]
        
        #Figure out the number of files in each volume
        files_per_vol = len(self._files_info) / n_vols
        
        #Pull the DICOM Patient Space affine from the first input
        dps_aff = self._files_info[0][0].nii_img.get_header().get_best_affine()
        
        #If there is more than one file per volume, we need to fix slice scaling
        if files_per_vol > 1:
            first_offset = dps_aff[:3, 3]
            second_offset = self._files_info[1][0].nii_img.get_header().get_best_affine()[:3,3]
            scaled_slc_dir = second_offset - first_offset
            dps_aff[:3, 2] = scaled_slc_dir
       
        #The Nifti patient space flips the x and y directions
        nps_aff = np.dot(np.diag([-1., -1., 1., 1.]), dps_aff)
        
        return nps_aff
        
    def to_nifti(self, voxel_order='rpi', embed_meta=False):
        '''Returns a NiftiImage with the data and affine from the stack.
        
        Parameters
        ----------
        voxel_order : str
            A three character string repsenting the voxel order in patient 
            space (see the function `reorder_voxels`).
            
        embed_meta : bool
            If true a dcmmeta.DcmMetaExtension will be embedded in the Nifti 
            header.
            
        Returns
        -------
        A nibabel.nifti1.Nifti1Image created with the stack's data and affine. 
        '''
        #Get the voxel data and affine   
        data = self.get_data()
        affine = self.get_affine()
        
        #Figure out the number of three (or two) dimensional volumes
        n_vols = 1
        if len(data.shape) > 3:
            n_vols *= data.shape[3]
        if len(data.shape) > 4:
            n_vols *= data.shape[4]
        
        files_per_vol = len(self._files_info) / n_vols 
        
        #Reorder the voxel data if requested
        permutation = (0, 1, 2)
        orig_slice_dir = affine[:3,2]
        if voxel_order:
            data, affine, permutation = reorder_voxels(data, 
                                                       affine, 
                                                       voxel_order)
                                                           
        #Reverse file order in each volume's files if we flipped slice order
        #This will keep the slice times and meta data order correct
        if files_per_vol > 1:
            new_slice_dir = affine[:3, permutation.index(2)]
            if np.allclose(-orig_slice_dir, new_slice_dir):
                self._shape_dirty = True
                for vol_idx in xrange(n_vols):
                    start = vol_idx * files_per_vol
                    stop = start + files_per_vol
                    self._files_info[start:stop] = [self._files_info[idx] 
                                                    for idx in xrange(stop - 1, 
                                                                      start - 1, 
                                                                      -1)
                                                   ]
            else:
                assert np.allclose(orig_slice_dir, new_slice_dir)
        
        #Create the nifti image using the data array
        nifti_image = nb.Nifti1Image(data, None)
        nifti_header = nifti_image.get_header()
        
        #Stick the affine in the q_form with 'scanner' code
        nifti_header.set_qform(affine, 'scanner')
        
        #Set the units and dimension info
        nifti_header.set_xyzt_units('mm', 'msec')
        if len(self._repetition_times) == 1 and not None in self._repetition_times:
            nifti_header['pixdim'][4] = list(self._repetition_times)[0]
        slice_dim = permutation.index(2)
        dim_info = {'freq' : None, 'phase' : None, 'slice' : slice_dim}
        if len(self._phase_enc_dirs) == 1 and not None in self._phase_enc_dirs:
            phase_dir = list(self._phase_enc_dirs)[0]
            if phase_dir == 'ROW':
                dim_info['phase'] = permutation.index(1)
                dim_info['freq'] = permutation.index(0)
            else:
                dim_info['phase'] = permutation.index(0)
                dim_info['freq'] = permutation.index(1)
        nifti_header.set_dim_info(**dim_info)
        n_slices = data.shape[slice_dim]
        
        #Set the slice timing header info
        has_acq_time = (self._files_info[0][0].get_meta('AcquisitionTime') != 
                        None)
        if files_per_vol > 1 and has_acq_time:
            #Pull out the relative slice times for the first volume
            slice_times = np.array([dcm_time_to_sec(file_info[0]['AcquisitionTime']) 
                                    for file_info in self._files_info[:n_slices]]
                                  )
            slice_times -= np.min(slice_times)
            
            #If there is more than one volume, check if times are consistent
            is_consistent = True
            for vol_idx in xrange(1, n_vols):
                start_slice = vol_idx * n_slices
                end_slice = start_slice + n_slices
                slices_info = self._files_info[start_slice:end_slice]
                vol_slc_times = \
                    np.array([dcm_time_to_sec(file_info[0]['AcquisitionTime']) 
                              for file_info in slices_info]
                            )
                vol_slc_times -= np.min(vol_slc_times)
                if not np.allclose(slice_times, vol_slc_times):
                    is_consistent = False
                    break
                
            #If the times are consistent and not all zero, try setting the slice 
            #times (sets the slice duration and code if possible).
            if is_consistent and not np.allclose(slice_times, 0.0):
                try:
                    nifti_header.set_slice_times(slice_times)
                except HeaderDataError:
                    pass
                
        #Embed the meta data extension if requested
        if embed_meta:
            #Build meta data for each volume if needed
            vol_meta = []
            if files_per_vol > 1:
                for vol_idx in xrange(n_vols):
                    start_slice = vol_idx * n_slices
                    end_slice = start_slice + n_slices
                    exts = [file_info[0].meta_ext
                            for file_info in self._files_info[start_slice:end_slice]]
                    meta = DcmMetaExtension.from_sequence(exts, 2)
                    vol_meta.append(meta)
            else:
                vol_meta = [file_info[0].meta_ext 
                            for file_info in self._files_info]
                            
            #Build meta data for each time point / vector component
            if len(data.shape) == 5:
                if data.shape[3] != 1:
                    vec_meta = []
                    for vec_idx in xrange(data.shape[4]):
                        start_idx = vec_idx * data.shape[3]
                        end_idx = start_idx + data.shape[3]
                        meta = DcmMetaExtension.from_sequence(\
                            vol_meta[start_idx:end_idx], 3)
                        vec_meta.append(meta)
                else:
                    vec_meta = vol_meta
                        
                meta_ext = DcmMetaExtension.from_sequence(vec_meta, 4)
            elif len(data.shape) == 4:
                meta_ext = DcmMetaExtension.from_sequence(vol_meta, 3)
            else:
                meta_ext = vol_meta[0]
                if meta_ext is file_info[0].meta_ext:
                    meta_ext = deepcopy(meta_ext)
                    
            meta_ext.set_shape(data.shape)
            meta_ext.set_slice_dim(slice_dim)
            meta_ext.set_affine(nifti_header.get_best_affine())
                    
            #Filter and embed the meta data
            meta_ext.filter_meta(self._meta_filter)
            nifti_header.extensions = Nifti1Extensions([meta_ext])

        nifti_image.update_header()
        return nifti_image
        
    def to_nifti_wrapper(self, voxel_order=''):
        return NiftiWrapper(self.to_nifti(voxel_order, True))
        
def parse_and_stack(src_paths, key_format='%(SeriesNumber)03d-%(ProtocolName)s', 
                    opt_key_suffix='TE_%(EchoTime).3f', extractor=None, 
                    force=False, warn_on_except=False, **stack_args):
    '''Parse the given dicom files into a dictionary containing one or more 
    DicomStack objects.
    
    Parameters
    ----------
    src_paths : sequence
        A list of paths to the source DICOM files.
        
    key_format : str
        A python format string used to create the dictionary keys in the result.
        The string is formatted with the DICOM meta data and any files with the 
        same result will be added to the same DicomStack.
        
    opt_key_suffix : str
        A python format string whose result is appended to the dictionary keys 
        created by key_format if (and only if) it differs between the source 
        DICOM files.
        
    extractor : callable
        Should take a dicom.dataset.Dataset and return a dictionary of the 
        extracted meta data. 
        
    force : bool
        Force reading source files even if they do not appear to be DICOM.
        
    warn_on_except : bool
        Convert exceptions into warnings, possibly allowing some results to be 
        returned.
    
    stack_args : kwargs
        Keyword arguments to pass to the DicomStack constructor.
    '''
    if extractor is None:
        extractor = default_extractor
        
    results = {}
    suffixes = {}
    for dcm_path in src_paths:
        try:
            dcm = dicom.read_file(dcm_path, force=force)
            meta = extractor(dcm)
            base_key = key_format % meta
            opt_suffix = opt_key_suffix % meta
            stack_key = base_key
            if base_key in suffixes:
                #We have already found more than one suffix, so use it
                if len(suffixes[base_key]) != 1:
                    stack_key = '%s-%s' % (base_key, opt_suffix)
                #We have found the first different suffix
                elif not opt_suffix in suffixes[base_key]:
                    #Change key for existing stack
                    existing_suffix = list(suffixes[base_key])[0]
                    new_key = '%s-%s' % (base_key, existing_suffix)
                    results[new_key] = results[base_key]
                    del results[base_key]
                    
                    #Use the suffix for this stack and add it to suffixes
                    stack_key = '%s-%s' % (base_key, opt_suffix)
                    suffixes[base_key].add(opt_suffix)
            else:
                suffixes[base_key] = set([opt_suffix])

            if not stack_key in results:
                results[stack_key] = DicomStack(**stack_args)
            results[stack_key].add_dcm(dcm, meta)
        except Exception, e:
            if warn_on_except:
                warnings.warn('Error adding file %s to stack: %s' % 
                              (dcm_path, str(e)))
            else:
                raise
    
    return results
