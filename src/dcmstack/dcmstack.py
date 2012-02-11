"""
Stack DICOM datasets into volumes.

@author: moloney
"""
import warnings
import dicom
import nibabel as nb
from nibabel.nifti1 import Nifti1Extensions
from nibabel.spatialimages import HeaderDataError
import numpy as np
from collections import OrderedDict
from .dcmmeta import DcmMetaExtension, NiftiWrapper

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from .extract import ExtractedDcmWrapper

_meta_version = 0.5

def closest_ortho_pat_axis(direction):
    '''Take a vector of three dimensions in DICOM patient space and return a 
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
    '''Reorder the given voxel array  and update the affine based on the 
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
    index_dirs = [affine[:-1,0].copy(),
                  affine[:-1,1].copy(),
                  affine[:-1,2].copy()
                 ]
    for i in range(3):
        index_dirs[i] /= np.dot(index_dirs[i], index_dirs[i])
    
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
        val = dcm[self.tag]
        if self.abs_ordering:
            if self.abs_as_str:
                val = str(val)
            return self.abs_ordering.index(val)
        else:
            return val

class DicomStack(object):
    '''
    Keeps track of a collection of DICOM data sets that make up 2D or 3D images 
    with optional time and vector dimensions. Can also summarize the meta data
    from the individual slices.
    '''     
    
    def __init__(self, time_order='AcquisitionTime', vector_order=None, 
                 allow_dummies=False):
        '''Initialize a DicomStack, optionally providing DicomOrdering objects 
        for ordering the time and vector dimensions. If the time order is None
        then 'AcquisitionTime' will be used.
        '''
        if isinstance(time_order, str):
            self._time_order = DicomOrdering(time_order)
        else:
            self._time_order = time_order
        if isinstance(vector_order, str):
            self._vector_order = DicomOrdering(vector_order)
        else:
            self._vector_order = vector_order
        self._allow_dummies = allow_dummies
        self._num_dummies = 0
        
        self._slice_pos_vals = set()
        self._time_vals = set()
        self._vector_vals = set()
        self._sorting_tuples = set()
        
        self._phase_enc_dir = None
        
        #We cache the shape and meta data to avoid duplicate processing
        self._shape_dirty = True
        self._shape = None
    
        self._meta_dirty = True
        self._meta = None
        
        self._files_info = []
    
    def add_dcm(self, dcm):
        '''Add a wrapped dicom object to the stack. The first DICOM added
        defines the orientation and the size of the row and column dimensions. 
        Each following DICOM added must match these paramaters or an
        IncongruentImageError will be raised. If the image has the same slice
        location or (if the time_order and vector_order are not None) 
        time/vector values, then an ImageCollisionError will be raised.'''

        #If we haven't seen a non-dummy, try pulling out the pixel array dims
        is_dummy = False
        if len(self._files_info) <= self._num_dummies:
            try:
                self._num_rows = dcm['Rows']
                self._num_cols = dcm['Columns']
            except KeyError, e:
                if not self._allow_dummies:
                    raise e
                is_dummy = True
        
        #If this is the first dicom, store its orientation and spacing info
        if len(self._files_info) == 0:
            self._row_idx_spacing = dcm['PixelSpacing'][1]
            self._col_idx_spacing = dcm['PixelSpacing'][0]
            self._row_idx_dir = np.array(dcm['ImageOrientationPatient'][3:])
            self._col_idx_dir = np.array(dcm['ImageOrientationPatient'][:3])
            self._image_orientation = dcm['ImageOrientationPatient']
            self._slice_dir = np.cross(self._row_idx_dir, self._col_idx_dir)
            if 'InplanePhaseEncodingDirection' in dcm:
                self._phase_enc_dir = dcm['InplanePhaseEncodingDirection']
        else: #Otherwise check for consistency
            try:
                if (dcm['Rows'] != self._num_rows or 
                   dcm['Columns'] != self._num_cols):
                    raise IncongruentImageError("Dimensions do not match")
            except KeyError, e:
                if not self._allow_dummies:
                    raise e
                is_dummy = True
            
            if (dcm['PixelSpacing'][0] != self._col_idx_spacing or
                dcm['PixelSpacing'][1] != self._row_idx_spacing):
                raise IncongruentImageError("Pixel spacings do not match")
            if dcm['ImageOrientationPatient'] != self._image_orientation:
                raise IncongruentImageError("Orientations do not match")
            if (self._phase_enc_dir and 
               (not 'InplanePhaseEncodingDirection' in dcm or
               dcm['InplanePhaseEncodingDirection'] != self._phase_enc_dir)):
                self._phase_enc_dir = None
        
        if is_dummy:
            self._num_dummies += 1
        
        #Pull the info used for sorting
        slice_pos = np.dot(self._slice_dir, 
                           np.array(dcm['ImagePositionPatient']))
        self._slice_pos_vals.add(slice_pos)
        time_val = None
        if self._time_order:
            time_val = self._time_order.get_ordinate(dcm)
        self._time_vals.add(time_val)
        vector_val = None
        if self._vector_order:
            vector_val = self._vector_order.get_ordinate(dcm)
        self._vector_vals.add(vector_val)
        
        #Create a tuple with the sorting values
        sorting_tuple = (vector_val, time_val, slice_pos)
        
        #Raise exception if image collides with another already in the stack
        if sorting_tuple in self._sorting_tuples:
            raise ImageCollisionError()
        
        #Add it to the stack
        self._sorting_tuples.add(sorting_tuple)
        self._files_info.append((dcm, sorting_tuple, is_dummy))
        
        #Set the dirty flags
        self._shape_dirty = True 
        self._meta_dirty = True
        
    def get_shape(self):
        '''Provided the stack is complete and valid, return it's shape. 
        Otherwise an InvalidStackError is raised.'''
        #If the dirty flag is not set, return the cached value
        if not self._shape_dirty:
            return self._shape

        #We need at least one file in the stack
        if len(self._files_info) == 0:
            raise InvalidStackError("No files in the stack")
        
        #We need at least one file to not be a dummy
        if self._num_dummies == len(self._files_info):
            raise InvalidStackError("All files in the stack are dummies")
        
        #Simple check for an incomplete stack
        slices_per_vol = len(self._slice_pos_vals)     
        if len(self._files_info) % slices_per_vol != 0:
            raise InvalidStackError("Number of files is not an even multiple "
                                    "of the number of unique slice positions.")
        
        #Perform an initial sort by vector/time/position tuple
        self._files_info.sort(key=lambda x: x[1])
        
        #We still need to sort each volume by its slice position
        num_vec_comps = len(self._vector_vals)
        num_time_points = ((len(self._files_info) / num_vec_comps) / 
                           slices_per_vol)
        num_volumes = num_vec_comps * num_time_points
        for vol_idx in range(num_volumes):
            start_slice = vol_idx * slices_per_vol
            end_slice = start_slice + slices_per_vol
            self._files_info[start_slice:end_slice] = \
                sorted(self._files_info[start_slice:end_slice], 
                       key=lambda x: x[1][-1])
        
        #Do a more thorough check for completeness
        slice_positions = sorted(list(self._slice_pos_vals))
        for vec_idx in xrange(num_vec_comps):
            file_idx = vec_idx*num_time_points*slices_per_vol
            curr_vec_val = self._files_info[file_idx][1][0]
            for time_idx in xrange(num_time_points):
                for slice_idx in xrange(slices_per_vol):
                    file_idx = (vec_idx*num_time_points*slices_per_vol + 
                                time_idx*slices_per_vol + slice_idx)
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
        shape = [self._num_rows,
                 self._num_cols,
                 slices_per_vol,
                 num_time_points,
                 num_vec_comps]
        if shape[4] == 1:
            shape = shape[:-1]
            if shape[3] == 1:
                shape = shape[:-1]
        self._shape = tuple(shape)
            
        self._shape_dirty = False
        return self._shape

    def get_array_and_affine(self):
        '''Return a tuple containing two numpy arrays: the array of voxel values
        and the corresponding affine transformation. The affine maps the row, 
        column, and slice indices to the voxel position in DICOM patient space. 
        
        If the DICOM files that are in the stack do not make a complete valid 
        stack then an InvalidStackError will be raised.         
        '''
        
        #Create a numpy array for storing the voxel data
        stack_shape = self.get_shape()
        
        #Pad the shape out with ones to simplify filling the stack
        stack_shape = tuple(list(stack_shape) + ((5 - len(stack_shape)) * [1]))
        vox_array = np.zeros(stack_shape, np.int16)
        
        #Fill the array with data
        for vec_idx in range(stack_shape[4]):
            for time_idx in range(stack_shape[3]):
                for slice_idx in range(stack_shape[2]):
                    file_idx = (vec_idx*(stack_shape[3]*stack_shape[2]) + 
                                time_idx*(stack_shape[2]) + slice_idx)
                    if self._files_info[file_idx][2]:
                        vox_array[:, :, slice_idx, time_idx, vec_idx] = \
                            np.iinfo(vox_array.dtype).max
                    else:
                        vox_array[:, :, slice_idx, time_idx, vec_idx] = \
                            self._files_info[file_idx][0].get_pixel_array()
        
        #Trim unused time/vector dimensions
        if stack_shape[4] == 1:
            vox_array = vox_array[...,0]
            if stack_shape[3] == 1:
                vox_array = vox_array[...,0]
        
        #Determine the slice scaling
        if stack_shape[2] == 1:
            scaled_slice_dir = self._slice_dir
        else:
            first_pos = np.array(self._files_info[0][0]['ImagePositionPatient'])
            last_pos = np.array(self._files_info[-1][0]['ImagePositionPatient'])
            slice_disp = last_pos - first_pos
            scaled_slice_dir = slice_disp / (stack_shape[2] - 1)
        
        #Determine the translation offset
        offset = np.array(self._files_info[0][0]['ImagePositionPatient'])
        
        #Scale the row, column index directions
        scaled_row_idx_dir = self._row_idx_dir*self._row_idx_spacing
        scaled_col_idx_dir = self._col_idx_dir*self._col_idx_spacing
        
        #Determine the affine mapping indices to DICOM Patient Space (DPS)
        affine = np.c_[np.append(scaled_row_idx_dir, [0]),
                       np.append(scaled_col_idx_dir, [0]),
                       np.append(scaled_slice_dir, [0]),
                       np.append(offset, [1])]
        
        return vox_array, affine
        
    def _get_meta_lvl_samples(self, meta, shape, lvl_idx):
        #Figure out the number of samples and slices per sample for this level        
        num_samples = shape[lvl_idx]
        slices_per_sample = shape[2]
        for i in xrange(3, lvl_idx):
            slices_per_sample *= shape[i]
        
        result = OrderedDict()
        #Iterate through the keys and value lists in the global 'slices' dict
        for key, vals in meta['global']['slices'].iteritems():
            #For each sample at the current level
            for sample_idx in xrange(num_samples):
                #Figure out the first value for this sample
                start_slice_idx = sample_idx * slices_per_sample
                curr_val = vals[start_slice_idx]
                #See if the value varies across slices in the sample
                is_const = True
                for slice_idx in xrange(slices_per_sample):
                    if curr_val != vals[start_slice_idx + slice_idx]:
                        is_const = False
                        break
                #If it does not vary across slices in the sample
                else:
                    #If the key is already in the results, append the value
                    if key in result:
                        result[key].append(curr_val)
                    #Otherwise add a new list to the results dict
                    else:
                        result[key] = [curr_val]
                
                #If the value is not constant across any sample then make sure
                #it is not in the results and move on to the next key
                if not is_const:
                    if key in result:
                        del result[key]
                    break
                
        #Delete any keys in the results from the global slices dict
        for key in result:
            del meta['global']['slices'][key]
            
        return result

    def _get_meta_lvl_slices(self, meta, shape, lvl_idx):
        #Figure out the number of samples and slices per sample for this level        
        num_samples = shape[lvl_idx]
        slices_per_sample = shape[2]
        for i in xrange(3, lvl_idx):
            slices_per_sample *= shape[i]
        
        result = OrderedDict()
        #Iterate through the keys and value lists in the global 'slices' dict
        for key, vals in meta['global']['slices'].iteritems():
            #Take the slice values for the first sample            
            curr_vals = vals[:slices_per_sample]
            #Check if the values repeat for all other samples
            for sample_idx in xrange(1, num_samples):
                start_slice_idx = sample_idx * slices_per_sample
                end_slice_idx = (sample_idx + 1) * slices_per_sample
                if curr_vals != vals[start_slice_idx:end_slice_idx]:
                    break
            else:
                result[key] = curr_vals
                
        #Delete any keys in the result dict from the global slices dict
        for key in result:
            del meta['global']['slices'][key]
                
        return result

    def _get_dcm_meta_ext(self, affine, shape, slice_dim):
        '''Return a dcmmeta.DcmMetaExtension the meta data for the stack. The meta 
        data in the dicom wrappers' "ext_meta" attribute is used for each slice.
        Data that is constant across the entire stack goes into 
        meta['global']['const'] while data that varies across the entire stack
        goes into meta['global']['slices'] (where each value is a list of the 
        values corresponding to each slice). If there is a time dimension, 
        values that are contant for each time sample go into 
        meta['time']['samples'] (where each value is a list of the same legnth 
        as the time dimension) and values that are repeating for the same slice
        across all time samples goes into meta['time']['slices'] (where the 
        values are lists of the same length as the number of slices in a time 
        point). If there is a vector dimesion it is handled in the same manner
        as the time dimension.'''
        if not self._meta_dirty:
            return self._meta

        #Initialize our meta dict with the 'global' level 
        meta = OrderedDict()
        meta['global'] = OrderedDict()

        #Pull out just the dicom datasets from our _files_info
        datasets = [file_info[0] for file_info in self._files_info]        
        
        #Initially add all key/value pairs to 'global' const dict
        meta['global']['const'] = datasets[0].ext_meta
        meta['global']['slices'] = OrderedDict()
        
        #Iterate through all keys in all files, updating the 'global' level
        meta_lvl = meta['global']
        for idx, ds in enumerate(datasets[1:]):
            for key, value in ds.ext_meta.iteritems():
                #If the key is currently in the const dict
                if key in meta_lvl['const']:
                    #If the value is varying, move it to 'slices'
                    if value != meta_lvl['const'][key]:
                        #Each value in 'slices' is a list of values 
                        meta_lvl['slices'][key] = ([meta_lvl['const'][key]] * 
                                                   (idx + 1))
                        meta_lvl['slices'][key].append(value)                      
                        del meta_lvl['const'][key]
                #If it is already in slices dict, append the value
                elif key in meta_lvl['slices']:
                    meta_lvl['slices'][key].append(value)
                #Otherwise we haven't seen this key before, use None for 
                #previous slices
                else:
                    meta_lvl['slices'][key] = [None] * (idx + 1)
                    meta_lvl['slices'][key].append(value)
           
            #Handle keys missing from this slice (using None as the value)
            for key, val in meta_lvl['const'].iteritems():
                if not key in ds.ext_meta:
                    meta_lvl['slices'][key] = [val] * (idx + 1)
                    meta_lvl['slices'][key].append(None)
            for key, vals in meta_lvl['slices'].iteritems():
                if key in meta_lvl['const']:
                    del meta_lvl['const'][key]
                if len(vals) != idx + 2:
                    vals.append(None)
        
        #Find values that are constant for each time/vector sample
        lvl_names = (None, None, None, 'time', 'vector')
        for lvl_idx in xrange(len(shape)-1, 2, -1):
            meta[lvl_names[lvl_idx]] = OrderedDict()
            meta[lvl_names[lvl_idx]]['samples'] = \
                self._get_meta_lvl_samples(meta, shape, lvl_idx)
                
        #Find values that repeat for the same slices in different time/vector 
        #samples
        for lvl_idx in xrange(3, len(shape)):
            meta[lvl_names[lvl_idx]]['slices'] = \
                self._get_meta_lvl_slices(meta, shape, lvl_idx)
            
        #Add in the meta meta data...
        meta['dcmmeta_version'] = _meta_version
        meta['dcmmeta_affine'] = [list(aff_row) for aff_row in list(affine)]
        meta['dcmmeta_shape'] = shape
        meta['dcmmeta_slice_dim'] = slice_dim
            
        self._meta = DcmMetaExtension.from_runtime_repr(meta)
        self._meta_dirty = False
        return self._meta

    def to_nifti(self, voxel_order='rpi', embed_meta=False):
        '''Calls 'get_array_and_affine', updates the affine to map to Nifti
        patient space, and then creates a nibabel.Nifti1Image. If 'embed_meta' 
        is True, create a meta_nii.DcmMetaExtension and include it in the Nifti 
        header extensions.'''
            
        #Get the voxel data and affine mapping to DICOM patient space (DPS)        
        data, dps_affine = self.get_array_and_affine()
        
        #Figure out the number of three (or two) dimensional volumes
        n_vols = 1
        if len(data.shape) > 3:
            n_vols *= data.shape[3]
        if len(data.shape) > 4:
            n_vols *= data.shape[3]
        
        #Reorder the voxel data if requested
        permutation = (0, 1, 2)
        orig_slice_dir = dps_affine[:3,2]
        if voxel_order:
            data, dps_affine, permutation = reorder_voxels(data, 
                                                           dps_affine, 
                                                           voxel_order)
                                                           
        #Reverse file order in each volume's files if we flipped slice order
        #This will keep the slice times and meta data order correct
        self._shape_dirty = True
        new_slice_dir = dps_affine[:3, permutation.index(2)]
        if np.allclose(-orig_slice_dir, new_slice_dir):
            for vol_idx in xrange(n_vols):
                start = vol_idx * data.shape[2]
                stop = start + data.shape[2]
                self._files_info[start:stop] = [self._files_info[idx] 
                                                for idx in xrange(stop - 1, 
                                                                  start - 1, 
                                                                  -1)
                                               ]
        
        #The NIFTI Patient Space (NPS) flips the x and y directions
        nps_affine = np.dot(np.diag([-1., -1., 1., 1.]), dps_affine)
        
        #Create the nifti image using the data array
        nifti_image = nb.Nifti1Image(data, None)
        nifti_header = nifti_image.get_header()
        
        #Stick the affine in the q_form with 'scanner' code
        nifti_header.set_qform(nps_affine, 'scanner')
        
        #Set the units and dimension info
        nifti_header = nifti_image.get_header()
        nifti_header.set_xyzt_units('mm', 'sec')
        dim_info = {'freq' : None, 'phase' : None, 'slice' : permutation.index(2)}
        if self._phase_enc_dir:
            if self._phase_enc_dir == 'ROW':
                dim_info['phase'] = permutation.index(1)
                dim_info['freq'] = permutation.index(0)
            else:
                dim_info['phase'] = permutation.index(0)
                dim_info['freq'] = permutation.index(1)
        nifti_header.set_dim_info(**dim_info)
        
        #Pull out the relative slice times for the first volume
        if 'AcquisitionTime' in self._files_info[0][0]:
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
            nifti_header.extensions = \
                Nifti1Extensions([self._get_dcm_meta_ext(nps_affine,
                                                         data.shape,
                                                         permutation.index(2))
                                 ]
                                )

        nifti_image.update_header()
        return nifti_image
        
    def to_nifti_wrapper(self, voxel_order=''):
        return NiftiWrapper(self.to_nifti(voxel_order, True))
        
def parse_and_stack(src_paths, key_format='%(SeriesNumber)03d-%(ProtocolName)s', 
                    time_order=None, vector_order=None, allow_dummies=False, 
                    extractor=None, meta_filter=None, force=False, 
                    warn_on_except=False):
    '''
    Create a dictionary mapping strings generated by 'key_format' to 'DcmStack' 
    objects. For each path in the iterable 'src_dicoms', create an 
    ExtractedDcmWrapper and add it to the stack coresponding to the string 
    generated by formatting 'key_format' with the wrapper meta data. 
    '''    
    results = {}
    for dcm_path in src_paths:
        try:
            dcm = dicom.read_file(dcm_path, force=force)
            wrp_dcm = ExtractedDcmWrapper.from_dicom(dcm, 
                                                     extractor, 
                                                     meta_filter)
            stack_key = key_format % wrp_dcm
            
            if not stack_key in results:
                results[stack_key] = DicomStack(time_order, 
                                                vector_order, 
                                                allow_dummies)
            results[stack_key].add_dcm(wrp_dcm)
        except Exception, e:
            if warn_on_except:
                warnings.warn('Error adding file %s to stack: %s' % 
                              (dcm_path, str(e)))
            else:
                raise
    
    return results
