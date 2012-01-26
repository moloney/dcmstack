"""
Nifti wrapper that includes addtional meta data. The meta data is embedded into
the Nifti as an extension.

@author: moloney
"""
import json
import numpy as np
import nibabel as nb
from nibabel.nifti1 import Nifti1Extension

dcm_meta_ecode = 19

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
        '''Check if the extension is valid.'''
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
        return np.array(self._content['dcmmeta_affine'])
        
    def get_slice_dim(self):
        return self._content['dcmmeta_slice_dim']

    def get_shape(self):
        return tuple(self._content['dcmmeta_shape'])
        
    def get_n_slices(self):
        return self.get_shape()[self.get_slice_dim()]
        
    def get_version(self):
        return self._content['dcmmeta_version']
    
    def to_json_file(self, path):
        '''Write out a JSON formatted text file with the extensions contents.'''
        if not self.is_valid():
            raise ValueError('The content dictionary is not valid.')
        out_file = open(path, 'w')
        out_file.write(self._mangle(self._content))
        out_file.close()
        
    @classmethod
    def from_json_file(klass, path):
        '''Read in a JSON formatted text file with the extensions contents.'''
        in_file = open(path)
        content = in_file.read()
        in_file.close()
        result = klass(dcm_meta_ecode, content)
        if not result.is_valid():
            raise ValueError('The JSON is not valid.')
        return result
        
    @classmethod
    def from_runtime_repr(klass, runtime_repr):
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

        if self._meta_ext.get_n_slices() != self.nii_img.get_n_slices():
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
                    return meta_dict['time']['samples'][key][index[2]]
                if (len(shape) > 4 and shape[4] > 1 and 
                   key in meta_dict['vector']['samples']):
                    return meta_dict['vector']['samples'][key][index[3]]
                
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
        '''Split the meta data along the index 'dim_idx', returning a list of
        NiftiWrapper objects. If 'dim_idx' is None it will prefer the vector, 
        then time, then slice dimensions.
        '''
#        shape = self.nii_img.get_shape()
#        slice_dim = self.nii_img.get_dim_info()[2]
#        
#        #If dim_idx is None, choose the vector/time/slice dim in that order
#        if dim_idx is None:
#            dim_idx = len(shape) - 1
#        if dim_idx == 2:
#            dim_idx = slice_dim
#            
#        data = self.nii_img.get_data()
#        affine = self.nii_img.get_affine()
#        header = self.nii_img.get_header()
#        if dim_idx == slice_dim:
#            header. #Need to unset slice specific bits of the header here.
#        results = []
#        slices = [slice(None)] * len(shape)
#        for idx in shape[dim_idx]:
#            slices[dim_idx] = idx
#            split_data = data[slices].copy()
#            results.append(nb.Nifti1Image(split_data, 
#                                          affine.copy(), 
#                                          header.copy()
#                                         )
#                          )
#        
#        return results
            
    
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
            