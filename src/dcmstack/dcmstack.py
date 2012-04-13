"""
Stack DICOM datasets into volumes.

@author: moloney
"""
import warnings, re, dicom
import nibabel as nb
from nibabel.nifti1 import Nifti1Extensions
from nibabel.spatialimages import HeaderDataError
import numpy as np
from collections import OrderedDict
from .dcmmeta import DcmMetaExtension, NiftiWrapper, _meta_version

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from .extract import default_extractor

def make_key_regex_filter(exclude_res, force_include_res=None):
    '''Make a regex filter that will exlude meta items where the key matches any
    of the regexes in exclude_res, unless it matches one of the regexes in 
    force_include_res.'''
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
'''A list of regexes passed to make_key_regex_filter as exclude_res to create 
the default_meta_filter.'''
                        
default_key_incl_res = ['ImageOrientationPatient',
                        'ImagePositionPatient',
                       ]                        
                        
default_meta_filter = make_key_regex_filter(default_key_excl_res,
                                            default_key_incl_res)
'''Default meta_filter for DicomStack.to_nifti method.'''

def closest_ortho_pat_axis(direction):
    '''Take a vector of three dimensions in Nifti patient space and return a 
    two character code coresponding to the orientation in terms of the closest 
    orthogonal axis. (eg. 'lr' would mean from (l)eft to (r)ight)'''
    if (abs(direction[0]) >= abs(direction[1]) and 
        abs(direction[0]) >= abs(direction[2])):
        if direction[0] < 0.0:
            return 'lr'
        else:
            return 'rl'
    elif (abs(direction[1]) >= abs(direction[0]) and 
          abs(direction[1]) >= abs(direction[2])):
        if direction[1] < 0.0:
            return 'pa'
        else:
            return 'ap'
    elif (abs(direction[2]) >= abs(direction[0]) and 
          abs(direction[2]) >= abs(direction[1])):
        if direction[2] < 0.0:
            return 'si'
        else:
            return 'is'

def reorder_voxels(vox_array, affine, voxel_order):
    '''Reorder the given voxel array and update the affine based on the 
    argument voxel_order. The affine should transform voxel indices to DICOM 
    patient space. Returns a tuple containing the updated voxel array, affine, 
    and a tuple of the permuted dimensions.
    
    The parameter voxel_order must be an empty string or a three character 
    code specifing the desired starting point for rows, columns, and slices 
    in terms of the orthogonal axes of patient space: (l)eft, (r)ight, 
    (a)nterior, (p)osterior, (s)uperior, and (i)nferior.'''
    
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
    '''Take a string corresponding to a DICOM time value (value representation 
    of TM) and convert it to a float representing the number of seconds past 
    midnight.'''
    return ((int(time_str[:2]) * 3600) + 
            (int(time_str[2:4]) * 60) + 
            float(time_str[4:]))

class IncongruentImageError(Exception):
    def __init__(self, msg):
        self.msg = msg
        
    def __str__(self):
        return 'The image is not congruent to the existing stack: %s' % self.msg

class ImageCollisionError(Exception):
    def __str__(self):
        return 'The image collides with one already in the stack'
        
class InvalidStackError(Exception):
    def __init__(self, msg):
        self.msg = msg
    
    def __str__(self):
        return 'The DICOM stack is not valid: %s' % self.msg

class DicomOrdering(object):
    '''Object defining an ordering for a set of dicom datasets.'''
    def __init__(self, tag, abs_ordering=None, abs_as_str=False):
        '''Create an ordering based on the DICOM attribute 'tag'. A list 
        specifying the absolute order for the values corresponding to that tag
        can be supplied as 'abs_ordering'. If abs_as_str is true, the value will
        be converted to a string before searching in abs_ordering.'''
        self.tag = tag
        self.abs_ordering = abs_ordering
        self.abs_as_str = abs_as_str
        
    def get_ordinate(self, dcm):
        try:
            val = dcm[self.tag]
        except KeyError:
            return None
            
        if self.abs_ordering:
            if self.abs_as_str:
                val = str(val)
            return self.abs_ordering.index(val)
        
        return val

def make_dummy(reference, meta):
    #Create the dummy data array filled with largest representable value
    data = np.empty_like(reference.nii_img.get_data())
    data[...] = np.iinfo(data.dtype).max
    
    #Create the nifti image and set header data
    nii_img = nb.nifti1.Nifti1Image(data, None)
    hdr = nii_img.get_header()
    hdr.set_qform(reference.nii_img.get_affine(), 'scanner')
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
    '''
    Convert a collection of DICOM data sets into 2D or 3D images with optional 
    time and vector dimensions. Can also summarize the meta data from the 
    individual files and produce a Nifti1Image with the meta data embeded.
    '''
    
    def __init__(self, time_order='AcquisitionTime', vector_order=None, 
                 allow_dummies=False, meta_filter=None, voxel_order='rpi'):
        '''Initialize a DicomStack object. 
        
        The method of ordering the time and vector dimensions can be specified 
        with 'time_order' and 'vector_order' by passing the DICOM keyword or a 
        DicomOrdering object to 'time_order' and 'vector_order' respectively. 
        
        If 'allow_dummies' is True then images without pixel data can be used. 
        The "dummy" slices will be replaced with the maximum representable 
        value for the datatype.
        
        If 'meta_filter' is provided, it must be a callable that takes a meta 
        data key and value, and returns True if that meta data element should 
        be excluded. If meta_filter is None it will default to the module's 
        'default_meta_filter'.
        
        The 'voxel_order' is a three character string repsenting the data 
        layout in patient space.        
        '''
        if isinstance(time_order, str):
            self._time_order = DicomOrdering(time_order)
        else:
            self._time_order = time_order
        if isinstance(vector_order, str):
            self._vector_order = DicomOrdering(vector_order)
        else:
            self._vector_order = vector_order
        self._slice_pos_vals = set()
        self._time_vals = set()
        self._vector_vals = set()
        self._sorting_tuples = set()
        
        self._phase_enc_dirs = set()
        self._repetition_times = set()
        
        if meta_filter is None:
            self._meta_filter = default_meta_filter
        else:
            self._meta_filter = meta_filter
        
        self._allow_dummies = allow_dummies
        self._dummies = []
        self._ref_input = None
        
        #We cache the shape and meta data to avoid duplicate processing
        self._shape_dirty = True
        self._shape = None
    
        self._meta_dirty = True
        self._meta = None
        
        self._files_info = []
        
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
            self._chk_consitency(('PixelSpacing', 
                                  'ImageOrientationPatient'), 
                                 meta, 
                                 self._dummies[0]
                                )
        return is_dummy
            
    
    def add_dcm(self, dcm, meta=None):
        '''Add a pydicom dataset 'dcm' to the stack. 
        
        Optionally provide a dict 'meta' with the extracted meta data for the
        DICOM data set. If None extract.default_extractor will be used.
        
        The first DICOM added defines the orientation and the size of the 
        dimensions. Each following DICOM added must match these paramaters or an 
        IncongruentImageError will be raised. 
        
        If the image has the same slice location or (if the time_order and 
        vector_order are not None) time/vector values, then an 
        ImageCollisionError will be raised.'''
        
        if meta is None:
            meta = default_extractor(dcm)

        is_dummy = self._chk_congruent(meta)
        
        self._phase_enc_dirs.add(meta.get('InplanePhaseEncodingDirection'))
        self._repetition_times.add(meta.get('RepetitionTime'))
        
        #Pull the info used for sorting
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
                    dummy_wrp = make_dummy(self._ref_input, dummy_meta)
                    self._files_info.append((dummy_wrp, dummy_tuple))
        else:
            if self._ref_input is None:
                #We don't have a reference input, so stash the dummy for now
                self._dummies.append((meta, sorting_tuple))
            else:
                #Convert dummy using the reference input
                nii_wrp = make_dummy(self._ref_input, meta)
        
        #If we made a NiftiWrapper add it to the stack
        if not nii_wrp is None:
            self._files_info.append((nii_wrp, sorting_tuple))
        
        #Set the dirty flags
        self._shape_dirty = True 
        self._meta_dirty = True
        
    def get_shape(self):
        '''Provided the stack is complete and valid, return it's shape. 
        Otherwise an InvalidStackError is raised.'''
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
        '''Return a numpy array of voxel values. 
        
        If the DICOM files that are in the stack do not make a complete valid 
        stack then an InvalidStackError will be raised.         
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
                    file_idx = (vec_idx*(stack_shape[3]*stack_shape[2]) + 
                                time_idx*(stack_shape[2]))
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
        '''Return the affine transform for mapping row/column/slice indices 
        to Nifti (RAS) patient space.'''
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
        dps_aff = self._files_info[0][0].nii_img.get_affine()
        
        #If there is more than one file per volume, we need to fix slice scaling
        if files_per_vol > 1:
            first_offset = dps_aff[:3, 3]
            second_offset = self._files_info[1][0].nii_img.get_affine()[:3,3]
            scaled_slc_dir = second_offset - first_offset
            dps_aff[:3, 2] = scaled_slc_dir
            
        #The Nifti patient space flips the x and y directions
        nps_aff = np.dot(np.diag([-1., -1., 1., 1.]), dps_aff)
        
        return nps_aff
        
    def to_nifti(self, voxel_order='rpi', embed_meta=False):
        '''Combines the slices into a single Nifti1Image
        
        The 'voxel_order' specifies the data layout in Nifti patient space.
        
        If 'embed_meta' is true a dcmmeta.DcmMetaExtension will be embedded in 
        the result.        
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
        nifti_image._affine = nifti_header.get_best_affine()
        
        #Set the units and dimension info
        nifti_header.set_xyzt_units('mm', 'msec')
        if len(self._repetition_times) == 1 and not None in self._repetition_times:
            nifti_header['pixdim'][4] = self._repetition_times.pop()
        dim_info = {'freq' : None, 'phase' : None, 'slice' : permutation.index(2)}
        if len(self._phase_enc_dirs) == 1 and not None in self._phase_enc_dirs:
            phase_dir = self._phase_enc_dirs.pop()
            if phase_dir == 'ROW':
                dim_info['phase'] = permutation.index(1)
                dim_info['freq'] = permutation.index(0)
            else:
                dim_info['phase'] = permutation.index(0)
                dim_info['freq'] = permutation.index(1)
        nifti_header.set_dim_info(**dim_info)
        
        #Set the slice timing header info
        has_acq_time = (self._files_info[0][0].get_meta('AcquisitionTime') != 
                        None)
        if files_per_vol > 1 and has_acq_time:
            #Pull out the relative slice times for the first volume
            n_slices = data.shape[dim_info['slice']]
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
            meta_ext = self._get_dcm_meta_ext(affine, 
                                              data.shape,
                                              permutation.index(2))
            
            meta_ext.filter_meta(self._meta_filter)
            nifti_header.extensions = Nifti1Extensions([meta_ext])

        nifti_image.update_header()
        return nifti_image
        
    def to_nifti_wrapper(self, voxel_order=''):
        return NiftiWrapper(self.to_nifti(voxel_order, True))
        
def parse_and_stack(src_paths, key_format='%(SeriesNumber)03d-%(ProtocolName)s', 
                    opt_key_suffix='TE_%(EchoTime).3f',
                    time_order=None, vector_order=None, allow_dummies=False, 
                    extractor=None, meta_filter=None, force=False, 
                    warn_on_except=False):
    '''
    
    Create a dictionary mapping strings generated by 'key_format' to 'DicomStack' 
    objects. For each path in the iterable 'src_dicoms', create an 
    ExtractedDcmWrapper and add it to the stack coresponding to the string 
    generated by formatting 'key_format' with the wrapper meta data. 
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
                results[stack_key] = DicomStack(time_order, 
                                                vector_order, 
                                                allow_dummies, 
                                                meta_filter)
            results[stack_key].add_dcm(dcm, meta)
        except Exception, e:
            if warn_on_except:
                warnings.warn('Error adding file %s to stack: %s' % 
                              (dcm_path, str(e)))
            else:
                raise
    
    return results
