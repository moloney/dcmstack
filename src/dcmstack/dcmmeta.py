"""
Nifti wrapper that includes addtional meta data. The meta data is embedded into
the Nifti as an extension.

@author: moloney
"""
import json
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import nibabel as nb
from nibabel.nifti1 import Nifti1Extension

dcm_meta_ecode = 0

def is_constant(sequence):
    '''Returns true if all elements in the sequence are equal.'''
    return all(val == sequence[0] for val in sequence)
    
def is_repeating(sequence, period):
    '''Returns true if the elements in the sequence repeat with the given 
    period.'''
    if len(sequence) % period != 0:
        raise ValueError('The sequence length is not evenly divisible by the '
                         'period length.')
    for period_idx in range(1, len(sequence) / period):
        start_idx = period_idx * period
        end_idx = start_idx * period
        if sequence[start_idx:end_idx] != sequence[:period]:
            return False
    
    return True

class DcmMetaExtension(Nifti1Extension):
    '''Subclass on Nifti1Extension. Handles conversion to and from json, checks
    the validity of the extension, and provides access to the "meta meta" data.
    '''
    
    _req_base_keys = set(('dcmmeta_affine', 
                          'dcmmeta_slice_dim',
                          'dcmmeta_shape',
                          'dcmmeta_version',
                          'global',
                         )
                        )
    
    def _unmangle(self, value):
        return json.loads(value)
    
    def _mangle(self, value):
        return json.dumps(value, indent=4)
    
    def is_valid(self):
        '''Return True if the extension is valid. Checks for the required 
        dictionaries and makes sure lists of values have the correct length.'''
        #Check for the required base keys in the json
        if not self._req_base_keys <= set(self._content):
            return False
            
        shape = self._content['dcmmeta_shape']
        
        #Check the 'global' dictionary
        if not set(('const' , 'slices')) == set(self._content['global']):
            return False
        
        total_slices = [self.get_n_slices()]
        for dim_size in shape[3:]:
            total_slices.append(total_slices[-1]*dim_size)
        for key, vals in self._content['global']['slices'].iteritems():
            if len(vals) != total_slices[-1]:
                return False
        
        #Check 'time' and 'vector' dictionaries if they exist
        if len(shape) > 3:
            if not 'time' in self._content:
                return False
            if not set(('samples', 'slices')) == set(self._content['time']):
                return False
            for key, vals in self._content['time']['samples'].iteritems():
                if len(vals) != shape[3]:
                    return False
            for key, vals in self._content['time']['slices'].iteritems():
                if len(vals) != total_slices[0]:
                    return False
        if len(shape) > 4:
            if not 'vector' in self._content:
                return False
            if not set(('samples', 'slices')) == set(self._content['vector']):
                return False
            for key, vals in self._content['time']['samples'].iteritems():
                if len(vals) != shape[4]:
                    return False
            for key, vals in self._content['vector']['slices'].iteritems():
                if len(vals) != total_slices[1]:
                    return False
        
        return True
    
    def get_affine(self):
        '''Return the affine associated with the meta data.'''
        return np.array(self._content['dcmmeta_affine'])
        
    def get_slice_dim(self):
        '''Get the index of the slice dimension.'''
        return self._content['dcmmeta_slice_dim']

    def get_shape(self):
        '''Returns the shape of the data associated with the meta data'''
        return tuple(self._content['dcmmeta_shape'])
        
    def get_n_slices(self):
        '''Returns the number of slices in each spatial volume.'''
        return self.get_shape()[self.get_slice_dim()]
        
    def get_version(self):
        '''Return the version of the meta data extension.'''
        return self._content['dcmmeta_version']
        
    def get_subset(self, dim, idx):
        '''Return a new DcmMetaExtension containing the subset of the meta data 
        corresponding to the index 'idx' along the dimension 'dim'.  The 
        dimension must be one of 'slice', 'time', or 'vector'.'''
        shape = self.get_shape()
        n_slices = self.get_n_slices()
            
        result = OrderedDict()
        result['global'] = OrderedDict()
        result['global']['const'] = deepcopy(self._content['global']['const'])
        result['global']['slices'] = OrderedDict()
        
        if dim == 'slice':
            #Per slice values become constant, everything else is the same
            for key, vals in self._content['global']['slices'].iteritems():
                result['global']['const'][key] = deepcopy(vals[idx])
            
            if 'time' in self._content:
                time_slice = idx % shape[3]
                for key, vals in self._content['time']['slices'].iteritems():
                    result['global']['const'][key] = deepcopy(vals[time_slice])
                result['time'] = OrderedDict()
                result['time']['samples'] = \
                    deepcopy(self._content['time']['samples'])
                result['time']['slices'] = OrderedDict()
            
            if 'vector' in self._content:
                vec_slice = idx % (shape[3] * shape[4])
                for key, vals in self._content['vector']['slices'].iteritems():
                    result['global']['const'][key] = deepcopy(vals[vec_slice])
                result['vector'] = OrderedDict()
                result['vector']['samples'] = \
                    deepcopy(self._content['vector']['samples'])
                result['vector']['slices'] = OrderedDict()                
                    
        elif dim == 'time':
            #Per time sample values become constant
            for key, vals in self._content['time']['samples'].iteritems():
                result['global']['const'][key] = deepcopy(vals[idx])
            
            if 'vector' in self._content:
                result['vector'] = OrderedDict()
                
                #Vector samples are left unchanged
                result['vector']['samples'] = \
                    deepcopy(self._content['vector']['samples'])
                
                result['vector']['slices'] = OrderedDict()
                
                #Need to take a subset of the global slices
                slices_per_vec_comp = n_slices * shape[3]
                for key, vals in self._content['global']['slices'].iteritems():
                    subset_vals = []
                    for vec_comp in shape[4]:
                        start_idx = ((vec_comp * slices_per_vec_comp) + 
                                     (idx * n_slices))
                        end_idx = start_idx + n_slices
                        subset_vals.append(deepcopy(vals[start_idx:end_idx]))
                    if is_constant(subset_vals):
                        result['global']['const'][key] = subset_vals[0]
                    elif is_repeating(subset_vals, n_slices):
                        result['vector']['slices'][key] = subset_vals[:n_slices]
                    else:
                        result['global']['slices'][key] = subset_vals
                    
                #Time point slices become vector slices
                time_slices = deepcopy(self._content['time']['slices'])
                result['vector']['slices'].update(time_slices)
                
                #Take a subset of vector slices, check if constant
                for key, vals in self._content['vector']['slices'].iteritems():
                    start_idx = idx * n_slices
                    end_idx = start_idx + n_slices
                    subset_vals = deepcopy(vals[start_idx:end_idx])
                    if is_constant(subset_vals):
                        result['global']['const'] = subset_vals[0]
                    else:
                        result['vector']['slices'][key] = subset_vals
                        
            else:
                #Take subset of global slices, check if constant
                for key, vals in self._content['global']['slices']:
                    start_idx = idx * n_slices
                    end_idx = start_idx + n_slices
                    subset_vals = deepcopy(vals[start_idx:end_idx])
                    if is_constant(subset_vals):
                        result['global']['const'] = subset_vals[0]
                    else:
                        result['global']['slices'][key] = subset_vals

                #Time point slices become global slices
                time_slices = deepcopy(self._content['time']['slices'])
                result['global']['slices'].update(time_slices)
        
        elif dim == 'vector':
            #Per vector sample values become constant
            for key, vals in self._content['vector']['samples'].iteritems():
                result['global']['const'][key] = deepcopy(vals[idx])
                
            if 'time' in self._content:
                result['time'] = OrderedDict()
                
                #Time samples and slices are left unchanged
                result['time']['samples'] = \
                    deepcopy(self._content['time']['samples'])
                result['time']['slices'] = \
                    deepcopy(self._content['time']['slices'])
                
                #Need to take a subset of the global slices
                slices_per_vec_comp = n_slices * shape[3]
                for key, vals in self._content['global']['slices'].iteritems():
                    start_idx = slices_per_vec_comp * idx
                    end_idx = start_idx + slices_per_vec_comp
                    subset_vals = deepcopy(vals[start_idx:end_idx])
                    if is_constant(subset_vals):
                        result['global']['const'][key] = subset_vals[0]
                    elif is_repeating(subset_vals, n_slices):
                        result['time']['slices'][key] = subset_vals[:n_slices]
                    else:
                        result['global']['slices'][key] = subset_vals
                        
                #Vector component slices become global slices
                vector_slices = deepcopy(self._content['vector']['slices'])
                result['global']['slices'].update(vector_slices)
                
            else:
                #Take subset of global slices, check if constant
                for key, vals in self._content['global']['slices']:
                    start_idx = idx * n_slices
                    end_idx = start_idx + n_slices
                    subset_vals = deepcopy(vals[start_idx:end_idx])
                    if is_constant(subset_vals):
                        result['global']['const'] = subset_vals[0]
                    else:
                        result['global']['slices'][key] = subset_vals
                        
                #Vector component slices become global slices
                vector_slices = deepcopy(self._content['vector']['slices'])
                result['global']['slices'].update(vector_slices)
        else:
            raise ValueError("The argument 'dim' must be one of 'slice', "
                             "'time', or 'vector'.")
        
        #Set the "meta meta" data
        result['dcmmeta_affine'] = deepcopy(self._content['dcmmeta_affine'])
        result['dcmmeta_slice_dim'] = deepcopy(self._content['dcmmeta_slice_dim'])
        result['dcmmeta_shape'] = deepcopy(self._content['dcmmeta_shape'])
        if dim == 'slice':
            result['dcmmeta_shape'][self.get_slice_dim] = 1
        elif dim == 'time':
            result['dcmmeta_shape'][3] = 1
        elif dim == 'vector':
            result['dcmmeta_shape'][4] = 1
        result['dcmmeta_version'] = self.get_meta_version()
        
        return self.from_runtime_repr(result)

    def to_json(self):
        '''Return the JSON string representation of the extension.'''
        if not self.is_valid():
            raise ValueError('The content dictionary is not valid.')
        return self._mangle(self._content)
        
    @classmethod
    def from_json(klass, json_str):
        '''Create an extension from the JSON string representation.'''
        result = klass(dcm_meta_ecode, json_str)
        if not result.is_valid():
            raise ValueError('The JSON is not valid.')
        return result
        
    @classmethod
    def from_runtime_repr(klass, runtime_repr):
        '''Create an extension from the Python runtime representation.'''
        result = klass(dcm_meta_ecode, '{}')
        result._content = runtime_repr
        if not result.is_valid():
            raise ValueError('The runtime representation is not valid.')
        return result
        
#Add our extension to nibabel
nb.nifti1.extension_codes.add_codes(((dcm_meta_ecode, 
                                      "dcmmeta", 
                                      DcmMetaExtension),)
                                   )

class NiftiWrapper(object):
    '''Wraps a nibabel.Nifti1Image object containing a DcmMetaExtension header 
    extension. Provides transparent access to the meta data through 'get_meta'.
    Allows the Nifti to be split into sub volumes or joined with others, while
    also updating the meta data appropriately.'''

    def __init__(self, nii_img):
        self.nii_img = nii_img
        self._meta_ext = None
        for extension in nii_img.get_header().extensions:
            if extension.get_code() == dcm_meta_ecode:
                if self._meta_ext:
                    raise ValueError("More than one DcmMetaExtension found")
                self._meta_ext = extension
        if not self._meta_ext:
            raise ValueError("No DcmMetaExtension found.")
        if not self._meta_ext.is_valid():
            raise ValueError("The meta extension is not valid")
    
    def samples_valid(self):
        '''Check if the meta data corresponding to individual time or vector 
        samples appears to be valid for the wrapped nifti image.'''
        #Check if the slice/time/vector dimensions match
        img_shape = self.nii_img.get_shape()
        meta_shape = self._meta_ext.get_shape()
        return meta_shape[3:] == img_shape[3:]
    
    def slices_valid(self):
        '''Check if the meta data corresponding to individual slices appears to 
        be valid for the wrapped nifti image.'''

        if (self._meta_ext.get_n_slices() != 
           self.nii_img.get_header().get_n_slices()):
            return False
        
        #Check that the affines match
        return np.allclose(self.nii_img.get_affine(), 
                           self._meta_ext.get_affine())
    
    def get_meta(self, key, index=None, default=None):
        '''Return the meta data value for the provided 'key', or 'default' if 
        there is no such (valid) key.
        
        If 'index' is not provided, only meta data values that are constant 
        across the entire data set will be considered. If 'index' is provided it 
        must be a valid index for the nifti voxel data, and all of the meta data 
        that is applicable to that index will be considered. The per-slice meta 
        data will only be considered if the object's 'is_aligned' method returns 
        True.'''
        
        #Pull out the meta dictionary
        meta_dict = self._meta_ext.get_content()
        
        #First check the constant values
        if key in meta_dict['global']['const']:
            return meta_dict['global']['const'][key]
        
        #If an index is provided check the varying values
        if not index is None:
            #Test if the index is valid
            shape = self.nii_img.get_shape()
            if len(index) != len(shape):
                raise IndexError('Incorrect number of indices.')
            for dim, ind_val in enumerate(index):
                if ind_val < 0 or ind_val >= shape[dim]:
                    raise IndexError('Index is out of bounds.')
            
            #First try per time/vector sample values
            if self.samples_valid():
                if (len(shape) > 3 and shape[3] > 1 and 
                   key in meta_dict['time']['samples']):
                    return meta_dict['time']['samples'][key][index[3]]
                if (len(shape) > 4 and shape[4] > 1 and 
                   key in meta_dict['vector']['samples']):
                    return meta_dict['vector']['samples'][key][index[4]]
                
            #Finally, if aligned, try per-slice values
            if self.slices_valid():
                slice_dim = self._meta_ext.get_slice_dim()
                if key in meta_dict['global']['slices']:
                    val_idx = index[slice_dim]
                    slices_per_sample = shape[slice_dim]
                    for count, idx_val in enumerate(index[3:]):
                        val_idx += idx_val * slices_per_sample
                        slices_per_sample *= shape[count+3]
                    return meta_dict['global']['slices'][key][val_idx]
                    
                if self.samples_valid():
                    if (len(shape) > 3 and shape[3] > 1 and 
                          key in meta_dict['time']['slices']):
                        val_idx = index[slice_dim]
                        return meta_dict['time']['slices'][key][val_idx]
                    elif (len(shape) > 4 and shape[4] > 1 and 
                          key in meta_dict['vector']['slices']):
                        val_idx = index[slice_dim]
                        val_idx += index[3]*shape[slice_dim]
                        return meta_dict['vector']['slices'][key][val_idx]
            
        return default
    
    def split(self, dim_idx=None):
        '''Split the array and meta data along the index 'dim_idx', returning a 
        list of NiftiWrapper objects. If 'dim_idx' is None it will prefer the 
        vector, then time, then slice dimensions.
        '''
        shape = self.nii_img.get_shape()
        slice_dim = self.nii_img.get_dim_info()[2]
        
        #If dim_idx is None, choose the vector/time/slice dim in that order
        if dim_idx is None:
            dim_idx = len(shape) - 1
        if dim_idx == 2:
            dim_idx = slice_dim
            
        #Make a string representation of the dim_idx
        dim_str = None
        if dim_idx < 3:
            dim_str = 'slice'
        elif dim_idx == 3:
            dim_str = 'time'
        elif dim_idx == 4:
            dim_str = 'vector'
        
        data = self.nii_img.get_data()
        header = self.nii_img.get_header()
        
        results = []
        slices = [slice(None)] * len(shape)
        for idx in shape[dim_idx]:
            #Create the initial Nifti1Image object
            slices[dim_idx] = idx
            split_data = data[slices].copy()
            split_nii = nb.Nifti1Image(split_data, None)
            split_hdr = split_nii.get_header()
            
            #Update the header
            split_hdr.set_qform(header.get_qform(), header['qform_code'])
            split_hdr.set_sform(header.get_sform(), header['sform_code'])
            split_hdr.set_slope_inter(*header.get_slope_inter())
            split_hdr.set_dim_info(*header.get_dim_info())
            split_hdr.set_intent(*header.get_intent())
            split_hdr.set_slice_duration(header.get_slice_duration())
            split_hdr.set_xyzt_units(*header.get_xyzt_units())
            split_hdr.set_xyzt_units(*header.get_xyzt_units())
            
            if dim_idx > 2:
                split_hdr.set_slice_times(header.get_slice_times())
                
            #Insert the subset of meta data
            split_meta = self._meta_ext.get_subset(dim_str, idx)
            split_hdr.extensions.append(split_meta)
            
            results.append(split_nii)
        
        return results
            
    def to_filename(self, out_path):
        if not self._meta_ext.is_valid:
            raise ValueError("Meta extension is not valid.")
        self.nii_img.to_filename(out_path)
    
    @classmethod
    def from_filename(klass, path):
        return klass(nb.load(path))
        
    @classmethod
    def from_sequence(klass, others, dim_idx=None):
        '''Create a NiftiWrapper from a sequence of other NiftiWrappers objects.
        The Nifti volumes are stacked along dim_idx in the given order.
        '''
            