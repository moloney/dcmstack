"""
DcmMeta header extension and NiftiWrapper for working with extended Niftis.
"""
import sys, json, warnings, itertools, re
from copy import deepcopy
import numpy as np
import nibabel as nb
from nibabel.nifti1 import Nifti1Extension
from nibabel.spatialimages import HeaderDataError

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from nibabel.nicom.dicomwrappers import wrapper_from_data

dcm_meta_ecode = 0

_meta_version = 0.7

_req_base_keys_map= {0.5 : set(('dcmmeta_affine', 
                                'dcmmeta_slice_dim',
                                'dcmmeta_shape',
                                'dcmmeta_version',
                                'global',
                                )
                               ),
                     0.6 : set(('dcmmeta_affine', 
                                'dcmmeta_reorient_transform',
                                'dcmmeta_slice_dim',
                                'dcmmeta_shape',
                                'dcmmeta_version',
                                'global',
                                )
                               ),
                     0.7 : set(('meta',
                                'const',
                                'per_slice',
                               )
                              )
                    }
'''Minimum required keys in the base dictionary to be considered valid'''

def is_constant(sequence, period=None):
    '''Returns true if all elements in (each period of) the sequence are equal. 
    
    Parameters
    ----------
    sequence : sequence
        The sequence of elements to check.
    
    period : int
        If not None then each subsequence of that length is checked. 
    '''
    if period is None:
        return all(val == sequence[0] for val in sequence)
    else:
        if period <= 1:
            raise ValueError('The period must be greater than one')
        seq_len = len(sequence)
        if seq_len % period != 0:
            raise ValueError('The sequence length is not evenly divisible by '
                             'the period length.')
                             
        for period_idx in range(seq_len / period):
            start_idx = period_idx * period
            end_idx = start_idx + period
            if not all(val == sequence[start_idx] 
                       for val in sequence[start_idx:end_idx]):
                return False
    
    return True
    
def is_repeating(sequence, period):
    '''Returns true if the elements in the sequence repeat with the given 
    period.

    Parameters
    ----------
    sequence : sequence
        The sequence of elements to check.
        
    period : int
        The period over which the elements should repeat.    
    '''
    seq_len = len(sequence)
    if period <= 1 or period >= seq_len:
        raise ValueError('The period must be greater than one and less than '
                         'the length of the sequence')
    if seq_len % period != 0:
        raise ValueError('The sequence length is not evenly divisible by the '
                         'period length.')
                         
    for period_idx in range(1, seq_len / period):
        start_idx = period_idx * period
        end_idx = start_idx + period
        if sequence[start_idx:end_idx] != sequence[:period]:
            return False
    
    return True

class InvalidExtensionError(Exception):
    def __init__(self, msg):
        '''Exception denoting than a DcmMetaExtension is invalid.'''
        self.msg = msg
    
    def __str__(self):
        return 'The extension is not valid: %s' % self.msg
        

class DcmMeta(OrderedDict):
    '''Ordered mapping for storing a summary of the meta data from the source 
    DICOM files used to create a N-d volume. Each meta data element is 
    classified based on whether it is constant over some dimension.
    '''
    
    def __init__(self, shape, affine, reorient_transform=None, slice_dim=None):
        '''Make an empty DcmMeta object.
        
        Parameters
        ----------
        shape : tuple
            The shape of the data associated with this extension.
            
        affine : array
            The RAS affine for the data associated with this extension.
            
        reorient_transform : array
            The transformation matrix representing any reorientation of the 
            data array.
            
        slice_dim : int
            The index of the slice dimension for the data associated with this 
            extension
        '''
        super(DcmMeta, self).__init__()

        #Create nested dict for storing "meta meta" data
        self['meta'] = OrderedDict()
        
        #Set all of the "meta meta" data
        self.shape = shape
        self.affine = affine
        self.reorient_transform = reorient_transform
        self.slice_dim = slice_dim
        self.version = _meta_version
        
        #Create a nested dict for each classification of meta data
        for classification in self.get_valid_classes():
            self[classification] = OrderedDict()
    
    @property
    def affine(self):
        '''The affine associated with the meta data. If this differs from the 
        image affine, the per-slice meta data will not be used. '''
        return np.array(self['meta']['affine'])
        
    @affine.setter
    def affine(self, value):
        if value.shape != (4, 4):
            raise ValueError("Invalid shape for affine")
        self['meta']['affine'] = value.tolist()
        
    @property
    def slice_dim(self):
        '''The index of the slice dimension associated with the per-slice 
        meta data.'''
        return self['meta']['slice_dim']
    
    @slice_dim.setter
    def slice_dim(self, value):
        if not value is None and not (0 <= value < 3):
            raise ValueError("The slice dimension must be in the range [0,3)")
        self['meta']['slice_dim'] = value
    
    @property
    def shape(self):
        '''The shape of the data associated with the meta data. Defines the 
        number of values for the meta data classifications.'''
        return tuple(self['meta']['shape'])
    
    @shape.setter
    def shape(self, value):
        if len(value) < 2:
            raise ValueError("There must be at least two dimensions")
        if len(value) == 2:
            value = value + (1,)
        self['meta']['shape'] = value
    
    @property
    def reorient_transform(self):
        '''The transformation due to reorientation of the data array. Can be 
        used to update directional DICOM meta data (after converting to RAS if 
        needed) into the same space as the affine.'''
        if self.version < 0.6:
            return None
        if self['meta']['reorient_transform'] is None:
            return None
        return np.array(self['meta']['reorient_transform'])
        
    @reorient_transform.setter
    def reorient_transform(self, value):
        if not value is None and value.shape != (4, 4):
            raise ValueError("The reorient_transform must be none or (4,4) "
            "array")
        if value is None:
            self['meta']['reorient_transform'] = None
        else:
            self['meta']['reorient_transform'] = value.tolist()
            
    @property
    def version(self):
        '''The version of the meta data extension.'''
        return self['meta']['version']
        
    @version.setter
    def version(self, value):
        '''Set the version of the meta data extension.'''
        self['meta']['version'] = value
        
    @property
    def slice_normal(self):
        '''The slice normal associated with the per-slice meta data.'''
        slice_dim = self.slice_dim
        if slice_dim is None:
            return None
        return np.array(self.affine[slice_dim][:3])
    
    @property
    def n_slices(self):
        '''The number of slices associated with the per-slice meta data.'''
        slice_dim = self.slice_dim
        if slice_dim is None:
            return None
        return self.shape[slice_dim]
    
    def get_valid_classes(self):
        '''Return the meta data classifications that are valid for this 
        extension.
        
        Returns
        -------
        valid_classes : tuple
            The classifications that are valid for this extension (based on its 
            shape). The classifications go from most general (const) to most 
            specific (per_slice).
        
        '''
        shape = self.shape
        n_dims = len(shape)
        result = ['const']
        n_extra_spatial = 0
        for dim_idx in xrange(3,n_dims):
            if shape[dim_idx] != 1:
                n_extra_spatial += 1
        if n_extra_spatial > 1:
            for dim_idx in xrange(n_dims-1, 2, -1):
                if shape[dim_idx] != 1:
                    result.append('per_sample_%d' % dim_idx)
        if n_extra_spatial >= 1:
            result.append('per_volume')
        #TODO: This probably shouldn't be included if there is only one 
        #slice and no extra spatial dimensions. It wouldn't be valid to have 
        #any meta data with the 'per_slice' classification in this case
        result.append('per_slice')
        return tuple(result)

    per_sample_re = re.compile('per_sample_([0-9]+)')
    
    def get_per_sample_dim(self, classification):
        if not classification in self.get_valid_classes():
            raise ValueError("Invalid classification: %s" % classification)
        
        match = re.match(self.per_sample_re, classification)
        if not match:
            raise ValueError("Classification is not per_sample")
        return int(match.groups()[0])
            
    def get_n_vals(self, classification):
        '''Get the number of meta data values for all meta data of the provided 
        classification.
        
        Parameters
        ----------
        classification : tuple
            The meta data classification.
            
        Returns
        -------
        n_vals : int
            The number of values for any meta data of the provided 
            `classification`.
        '''
        if not classification in self.get_valid_classes():
            raise ValueError("Invalid classification: %s" % classification)
        
        if classification == 'const':
            n_vals = 1
        elif classification == 'per_slice':
            n_vals = self.n_slices
            if not n_vals is None:
                for dim_size in self.shape[3:]:
                    n_vals *= dim_size
        elif classification == 'per_volume':
            n_vals = self.shape[3]
            for dim_size in self.shape[4:]:
                n_vals *= dim_size
        else:
            dim = self.get_per_sample_dim(classification)
            n_vals = self.shape[dim]
        return n_vals
    
    def check_valid(self):
        '''Check if the extension is valid.
        
        Raises
        ------
        InvalidExtensionError 
            The extension is missing required meta data or classifications, or
            some element(s) have the wrong number of values for their 
            classification.
        '''
        #TODO: Update meta data from older versions
        #For now we just fail out with older versions of the meta data
        if self.version != _meta_version:
            raise InvalidExtensionError("Meta data version is out of date, "
                                        "you may be able to convert to the "
                                        "current version.")
                                        
        #Check for the required base keys in the json data
        if not _req_base_keys_map[self.version] <= set(self):
            raise InvalidExtensionError('Missing one or more required keys')
            
        #Check the orientation/shape/version
        if self.affine.shape != (4, 4):
            raise InvalidExtensionError('Affine has incorrect shape')
        slice_dim = self.slice_dim
        if slice_dim != None:
            if not (0 <= slice_dim < 3):
                raise InvalidExtensionError('Slice dimension is not valid')
        if not len(self.shape) >= 3:
            raise InvalidExtensionError('Shape is not valid')
            
        #Check all required meta dictionaries, make sure elements have correct
        #number of values
        valid_classes = self.get_valid_classes()
        for classification in valid_classes:
            if not classification in self:
                raise InvalidExtensionError('Missing required classification ' 
                                            '%s' % classification)
            cls_dict = self[classification]
            cls_n_vals = self.get_n_vals(classification)
            if cls_n_vals > 1:
                for key, vals in cls_dict.iteritems():
                    n_vals = len(vals)
                    if n_vals != cls_n_vals:
                        msg = (('Incorrect number of values for key %s with '
                                'classification %s, expected %d found %d') %
                               (key, classification, cls_n_vals, n_vals)
                              )
                        raise InvalidExtensionError(msg)
                        
        #Check that all keys are uniquely classified
        for classification in valid_classes:
            for other_class in valid_classes:
                if classification == other_class:
                    continue
                intersect = (set(self[classification]) & 
                             set(self[other_class])
                            )
                if len(intersect) != 0:
                    raise InvalidExtensionError("One or more keys have "
                                                "multiple classifications")
            
    def get_all_keys(self):
        '''Get a list of all the meta data keys that are available across 
        classifications.'''
        keys = []
        for classification in self.get_valid_classes():
            keys += self[classification].keys()
        return keys

    def get_classification(self, key):
        '''Get the classification for the given `key`.
        
        Parameters
        ----------
        key : str
            The meta data key.
        
        Returns
        -------
        classification : tuple or None
            The classification tuple for the provided key or None if the key is 
            not found.
            
        '''
        for classification in self.get_valid_classes():
            if key in self[classification]:
                return classification
        return None
        
    def get_values(self, key):
        '''Get all values for the provided key. 

        Parameters
        ----------
        key : str
            The meta data key.
            
        Returns
        -------
        values 
             The value or values for the given key. The number of values 
             returned depends on the classification (see 'get_multiplicity').
        '''
        classification = self.get_classification(key)
        if classification is None:
            return None
        return self[classification][key]
    
        
    def filter_meta(self, filter_func):
        '''Filter the meta data.
        
        Parameters
        ----------
        filter_func : callable
            Must take a key and values as parameters and return True if they 
            should be filtered out.
            
        '''
        for classification in self.get_valid_classes():
            filtered = []
            curr_dict = self[classification]
            for key, values in curr_dict.iteritems():
                if filter_func(key, values):
                    filtered.append(key)
            for key in filtered:
                del curr_dict[key]
    
    def simplify(self, key):
        '''Try to simplify (reduce the number of values) of a single meta data 
        element by changing its classification. Return True if the 
        classification is changed, otherwise False. Lookks for values that are 
        constant over some period. Constant elements with a value of None will 
        be deleted.
        
        You should only need to call this after adding some meta data yourself.
        '''
        curr_class = self.get_classification(key)
        values = self[curr_class][key]
        
        if curr_class == 'const':
            #If the class is const then just delete it if the value is None
            if not values is None:
                return False
        elif curr_class.startswith('per_sample'):
            #If the class is per sample, can only become const
            if is_constant(values):
                if values[0] != None:
                    self['const'][key] = values[0]
            else:
                return False
        elif curr_class == 'per_volume':
            #Can become const or per sample
            if is_constant(values):
                if values[0] != None:
                    self['const'][key] = values[0]
            else:
                shape = self.shape
                simp_vals = []
                for dim in xrange(len(shape) - 1, 2, -1):
                    dest_class = 'per_sample_%d' % dim
                    if not dest_class in self.get_valid_classes():
                        return False
                    for sample_idx in xrange(shape[dim]):
                        sub_vals = self._per_volume_subset(key, 
                                                           dim, 
                                                           sample_idx)
                        if is_constant(sub_vals):
                            simp_vals.append(sub_vals[0])
                        else:
                            break
                    else:
                        self[dest_class][key] = simp_vals
                        break
                else:
                    return False
        else:
            #Can become const, per sample, or per volume
            if is_constant(values):
                if values[0] != None:
                    self['const'][key] = values[0]
            else:
                shape = self.shape
                simp_vals = []
                simp_found = False
                #Try any per sample classifications
                for dim in xrange(len(shape) - 1, 2, -1):
                    dest_class = 'per_sample_%d' % dim
                    if not dest_class in self.get_valid_classes():
                        break
                    for sample_idx in xrange(shape[dim]):
                        sub_vals = self._per_slice_subset(key, 
                                                          dim, 
                                                          sample_idx)
                        if is_constant(sub_vals):
                            simp_vals.append(sub_vals[0])
                        else:
                            break
                    else:
                        self[dest_class][key] = simp_vals
                        simp_found = True
                        break
                if not simp_found:
                    n_slices = self.n_slices
                    if (n_slices == 1 or 
                        (not n_slices is None and 
                         is_constant(values, n_slices))):
                        self['per_volume'][key] = values[::n_slices]
                    else:
                        return False
                
        #The element was reclassified, delete original and return True
        del self[curr_class][key]
        return True
    
    def get_subset(self, dim, idx):
        '''Get a DcmMeta object containing a subset of the meta data 
        corresponding to a single index along a given dimension.
        
        Parameters
        ----------
        dim : int
            The dimension we are taking the subset along.
            
        idx : int
            The position on the dimension `dim` for the subset.
        
        Returns
        -------
        result : DcmMeta
            A new DcmMeta object corresponding to the subset.
            
        '''
        shape = self.shape
        n_dim = len(shape)
        if not 0 <= dim < shape:
            raise ValueError("The argument 'dim' is out of range")
        valid_classes = self.get_valid_classes()
        
        #Make an empty extension for the result
        result_shape = list(shape)
        result_shape[dim] = 1
        while result_shape[-1] == 1 and len(result_shape) > 3:
            result_shape = result_shape[:-1]
        result = DcmMeta(result_shape, 
                         self.affine,
                         self.reorient_transform,
                         self.slice_dim
                        )
        
        for src_class in valid_classes:
            #Constants always remain constant
            if src_class == 'const':
                for key, val in self['const'].iteritems():
                    result['const'][key] = deepcopy(val)
                continue
            
            if dim == self.slice_dim:
                if src_class != 'per_slice':
                    #Any per volume/sample meta data stays the same
                    for key, vals in self[src_class].iteritems():
                        result[src_class][key] = deepcopy(vals)
                else:
                    #Per slice meta data gets reclassified as per 
                    #volume/sample or const
                    result._copy_slice(self, idx)
                    
            elif dim < 3:
                #If splitting spatial, non-slice, dim keep everything
                for key, vals in self[src_class].iteritems():
                    result[src_class][key] = deepcopy(vals)
            elif dim >= 3:
                #Per slice/sample meta data may need to be reclassified
                result._copy_sample(self, src_class, dim, idx)
                
        return result
        
    @classmethod
    def from_sequence(klass, seq, dim, affine=None, slice_dim=None):
        '''Create an DcmMeta object from a sequence of DcmMeta objects by 
        joining them along the given dimension.
        
        Parameters
        ----------
        seq : sequence
            The sequence of DcmMeta objects.
        
        dim : int
            The dimension to merge the extensions along.
        
        affine : array
            The affine to use in the resulting extension. If None, the affine 
            from the first extension in `seq` will be used.
            
        slice_dim : int
            The slice dimension to use in the resulting extension. If None, the 
            slice dimension from the first extension in `seq` will be used.
        
        Returns
        -------
        result : DcmMeta
            The result of merging the extensions in `seq` along the dimension
            `dim`.
        '''
        n_inputs = len(seq)
        first_input = seq[0]
        input_shape = first_input.shape
        
        if len(input_shape) > dim and input_shape[dim] != 1:
            raise ValueError("The dim must be singular or not exist for the "
                             "inputs.")
        
        output_shape = list(input_shape)
        while len(output_shape) <= dim:
            output_shape.append(1)
        output_shape[dim] = n_inputs
        
        if affine is None:
            affine = first_input.affine
        if slice_dim is None:
            slice_dim = first_input.slice_dim
            
        result = klass(output_shape, 
                       affine, 
                       None, 
                       slice_dim)
        
        #Need to initialize the result with the first extension in 'seq'
        result_slc_norm = result.slice_normal
        first_slc_norm = first_input.slice_normal
        use_slices = (not result_slc_norm is None and
                      not first_slc_norm is None and 
                      np.allclose(result_slc_norm, first_slc_norm))
        for classes in first_input.get_valid_classes():
            if classes == 'per_slice' and not use_slices:
                continue
            result[classes] = deepcopy(first_input[classes])
        
        #Adjust the shape to what the extension actually contains
        shape = list(result.shape)
        shape[dim] = 1
        result.shape = shape

        #Initialize reorient transform
        reorient_transform = first_input.reorient_transform
        
        #Add the other extensions, updating the shape as we go
        for input_ext in seq[1:]:
            #If the affines or reorient_transforms don't match, we set the 
            #reorient_transform to None as we can not reliably use it to update 
            #all of the directional meta data
            if ((reorient_transform is None or 
                 input_ext.reorient_transform is None) or 
                not np.allclose(input_ext.affine, affine) or 
                not np.allclose(input_ext.reorient_transform, 
                                reorient_transform)
               ):
                reorient_transform = None
            result._insert(dim, input_ext)
            shape[dim] += 1
            result.shape = shape
            
        #Set the reorient transform 
        result.reorient_transform = reorient_transform
            
        #Try simplifying all of the keys classified as per slice/volume
        if 'per_volume' in result.get_valid_classes():
            for key in result['per_volume'].keys():
                result.simplify(key)
        for key in result['per_slice'].keys():
            result.simplify(key)
            
        return result
        
    @classmethod
    def from_mapping(cls, mapping):
        result = cls((0,0,0), np.eye(4))
        result.update(mapping)
        return result
        
    def __eq__(self, other):
        if not np.allclose(self.affine, other.affine):
            return False
        if self.shape != other.shape:
            return False
        if self.slice_dim != other.slice_dim:
            return False
        if not ((self.reorient_transform is None and 
                 other.reorient_transform is None) or
                not np.allclose(self.reorient_transform, 
                                other.reorient_transform)
               ):
            return False
        if self.version != other.version:
            return False
        for classes in self.get_valid_classes():
            if dict(self[classes]) != dict(other[classes]):
                return False
                
        return True
    
    def __deepcopy__(self, memo):
        result = DcmMeta(self.shape, 
                         self.affine, 
                         self.reorient_transform, 
                         self.slice_dim)
        for classification in self.get_valid_classes():
            result[classification] = deepcopy(self[classification])
        return result
    
    def _get_const_period(self, src_cls, dest_cls):
        '''Get the period over which we test for const-ness with for the 
        given classification change.'''
        if dest_cls == 'const':
            return None
        elif src_cls == 'per_slice':
            return self.get_n_vals(src_cls) / self.get_n_vals(dest_cls)
        else:
            src_dim = self.get_per_sample_dim(src_cls)
            dest_dim = self.get_per_sample_dim(dest_cls)
            result = self.shape[src_dim + 1]
            for dim_size in self.shape[src_dim + 2:dest_dim]:
                result *= dim_size
            return result
        assert False #Should take one of the above branches

    def _get_preserving_changes(self, classification):
        '''Get a list of classifications we can change to from 
        'classification' without losing any data (just increasing the number 
        of values). Results are ordered from smallest to largest increase in 
        number of values'''
        valid_classes = self.get_valid_classes()
        if classification == None:
            return valid_classes
        if classification == 'per_slice':
            return [None]
        if classification == 'per_volume':
            return ['per_slice']
        if classification == 'const':
            return [cls for cls in valid_classes if not cls == 'const']
            
        #Classification must be per_sample
        result = []
        curr_per_sample_dim = self.get_per_sample_dim(classification)
        for cls in valid_classes:
            if cls == 'const':
                continue
            if cls.startswith('per_sample_'):
                sample_dim = self.get_per_sample_dim(cls)
                if sample_dim > curr_per_sample_dim:
                    continue
            result.append(cls)
        return result
       
    def _get_changed_class(self, key, new_class, slice_dim=None):
        '''Get an array of values corresponding to a single meta data 
        element with its classification changed by duplicating values. This 
        will preserve all the meta data and allow easier merging of values 
        with different classifications.'''
        curr_class = self.get_classification(key)
        if curr_class is None:
            values = None
            curr_n_vals = 1
            if new_class == 'const':
                return values
        else:
            values = self[curr_class][key]
            curr_n_vals = self.get_n_vals(curr_class)
            
        if curr_class == new_class:
            return deepcopy(values)
        
        if new_class in self.get_valid_classes():
            new_n_vals = self.get_n_vals(new_class)
            #Only way we get None for n_vals is if slice dim is undefined, so 
            #we require the slice_dim argument
            if new_n_vals == None:
                new_n_vals = self.shape[slice_dim]
        else:
            new_n_vals = 1
        mult_fact = new_n_vals / curr_n_vals
        if curr_n_vals == 1:
            values = [values]
            
        result = []
        for value in values:
            result.extend([deepcopy(value) for idx in xrange(mult_fact)])
        
        return result
        
    def _change_class(self, key, new_class):
        '''Change the classification of the meta data element in place. See 
        _get_changed_class.'''
        curr_class = self.get_classification(key)
        self[new_class][key] = self._get_changed_class(key, new_class)
        if not curr_class is None:
            del self[curr_class][key]
    
    def _copy_slice(self, other, idx):
        '''Get a copy of the meta data from the 'other' instance with 
        classification 'per_slice', corresponding to the slice with index
        'idx'.'''
        n_dims = len(self.shape)
        valid_classes = self.get_valid_classes()
        if 'per_volume' in valid_classes:
            dest_class = 'per_volume'
        else:
            dest_class = 'const'
        
        dest_n_vals = self.get_n_vals(dest_class)
        stride = other.n_slices
        for key, vals in other['per_slice'].iteritems():
            subset_vals = vals[idx::stride]
            if len(subset_vals) == 1:
                subset_vals = subset_vals[0]
            self[dest_class][key] = deepcopy(subset_vals)
            self.simplify(key)

    def _per_slice_subset(self, key, dim, idx):
        '''Get a subset of the meta data values with the classificaion 
        'per_slice' corresponding to a single sample along dimnesion 'dim' at 
        index 'idx'.
        '''
        n_slices = self.n_slices
        shape = self.shape
        n_dims = len(shape)
        slices_per_sample = [self.n_slices]
        for dim_idx in xrange(3, n_dims):
            slices_per_sample.append(slices_per_sample[-1] * shape[dim_idx])
        
        if dim == n_dims - 1:
            #If dim is the last non-spatial dimension, just take contiguous set
            start_idx = idx * slices_per_sample[dim - 3]
            end_idx = start_idx + slices_per_sample[dim - 3]
            return self['per_slice'][key][start_idx:end_idx]
        else:
            #Otherwise we need to iterate over higher dimensions
            result = []
            slc_offset = idx * slices_per_sample[dim - 3]
            higher_dim_iters = [xrange(size) for size in shape[dim+1:]]
            for higher_indices in itertools.product(*higher_dim_iters):
                start_idx = slc_offset
                higher_dim = dim + 1
                for higher_idx in higher_indices:
                    start_idx += higher_idx * slices_per_sample[higher_dim - 3]
                    higher_dim += 1
                end_idx = start_idx + slices_per_sample[dim - 3]
                result.extend(self['per_slice'][key][start_idx:end_idx])
            return result
    
    def _per_volume_subset(self, key, dim, idx):
        '''Get a subset of the meta data values with the classificaion 
        'per_volume' corresponding to a single sample along dimnesion 'dim' at 
        index 'idx'.
        '''
        shape = self.shape
        n_dims = len(shape)
        vols_per_sample = [1]
        for dim_idx in xrange(3, n_dims-1):
            vols_per_sample.append(vols_per_sample[-1] * shape[dim_idx])
        if dim == n_dims - 1:
            #If dim is the last non-spatial dimension, just take contiguous set
            start_idx = idx * vols_per_sample[dim - 3]
            end_idx = start_idx + vols_per_sample[dim - 3]
            return self['per_volume'][key][start_idx:end_idx]
        else:
            #Otherwise we need to iterate over higher dimensions
            result = []
            vol_offset = idx * vols_per_sample[dim - 3]
            higher_dim_iters = [xrange(size) for size in shape[dim+1:]]
            for higher_indices in itertools.product(*higher_dim_iters):
                start_idx = vol_offset
                higher_dim = dim + 1
                for higher_idx in higher_indices:
                    start_idx += higher_idx * vols_per_sample[higher_dim - 3]
                    higher_dim += 1
                end_idx = start_idx + vols_per_sample[dim - 3]
                result.extend(self['per_volume'][key][start_idx:end_idx])
            return result
            
    
    def _copy_sample(self, other, src_class, dim, idx):
        '''Get a copy of meta data from 'other' instance with classification 
        'src_class', corresponding to one sample along the dimension 'dim' at 
        index 'idx'.'''
        if src_class == 'per_slice':
            #Take a subset of per_slice meta data
            for key, vals in other['per_slice'].iteritems():
                subset_vals = other._per_slice_subset(key, dim, idx)
                self['per_slice'][key] = deepcopy(subset_vals)
                self.simplify(key)
        elif src_class == 'per_volume':
            #Per volume meta data may become constant
            if not 'per_volume' in self.get_valid_classes():
                for key, vals in other['per_volume'].iteritems():
                    self['const'][key] = deepcopy(vals[idx])
            else:
                #Otherwise we take a subset of the per_volume meta
                for key, vals in other['per_volume'].iteritems():
                    subset_vals = other._per_volume_subset(key, dim, idx)
                    self['per_volume'][key] = deepcopy(subset_vals)
                    self.simplify(key)
        elif src_class == 'per_sample_%d' % dim:
            #Per sample meta on the split dim becomes constant
            for key, vals in other[src_class].iteritems():
                self['const'][key] = deepcopy(vals[idx])
        else:
            #Per sample meta that is not on the split dim stays the same or 
            #becomes per volume
            if src_class in self.get_valid_classes():
                dest_class = src_class
            else:
                dest_class = 'per_volume'
            for key, vals in other[src_class].iteritems():
                self[dest_class][key] = deepcopy(vals)
                self.simplify(key)
    
    def _insert(self, dim, other):
        '''Insert the meta data from 'other' along the given 'dim'.'''
        self_slc_norm = self.slice_normal
        other_slc_norm = other.slice_normal
        
        #If we are not using slice meta data, temporarily remove it from the 
        #other dcmmeta object
        use_slices = (not self_slc_norm is None and
                      not other_slc_norm is None and 
                      np.allclose(self_slc_norm, other_slc_norm))
        if not use_slices:
            other_slice_meta = other['per_slice']
            other['per_slice'] = {}
        
        missing_keys = list(set(self.get_all_keys()) - 
                            set(other.get_all_keys()))
        for other_class in other.get_valid_classes():
            other_keys = other[other_class].keys()
            
            #Treat missing keys as if they were in const and have a value
            #of None
            if other_class == 'const':
                other_keys += missing_keys
            
            #When possible, reclassify our meta data so it matches the other
            #classification
            for key in other_keys:
                local_class = self.get_classification(key)
                if local_class != other_class:
                    local_allow = self._get_preserving_changes(local_class)
                    other_allow = self._get_preserving_changes(other_class)
                   
                    if other_class in local_allow:
                        self._change_class(key, other_class)
                    elif not local_class in other_allow:
                        #If we can't directly reclassify the meta data from 
                        #self or other to match, reclassify self meta data to 
                        #something that the other meta data can also be 
                        #reclassified to
                        best_dest = None
                        for dest_class in local_allow:
                            if dest_class in other_allow:
                                best_dest = dest_class
                                break
                        self._change_class(key, best_dest)
                        
            #Insert new meta data and further reclassify as necessary
            for key in other_keys:
                if dim == self.slice_dim:
                    self._insert_slice(key, other)
                elif dim < 3:
                    self._insert_non_slice(key, other)
                else:
                    self._insert_sample(key, other, dim)
        
        #Restore per slice meta if needed
        if not use_slices:
            other['per_slice'] = other_slice_meta
               
    def _insert_slice(self, key, other):
        '''Insert the meta data from 'other', corresponding to a single index 
        along the slice dimension.'''
        curr_class = self.get_classification(key)
        if curr_class != 'per_slice':
            self._change_class(key, 'per_slice')
        
        local_vals = self['per_slice'][key]
        other_vals = other._get_changed_class(key, 'per_slice', self.slice_dim)
        
        #Need to interleave slices
        n_slices = self.n_slices
        shape = self.shape
        n_vols = 1
        for dim_size in shape[3:]:
            n_vols *= dim_size
        
        intlv = []
        loc_start = 0
        oth_start = 0
        for vol_idx in xrange(n_vols):
            intlv += local_vals[loc_start:loc_start + n_slices]
            intlv += other_vals[oth_start:oth_start + 1]
            loc_start += n_slices
            oth_start += 1
            
        self['per_slice'][key] = intlv
            
    def _insert_non_slice(self, key, other):
        '''Insert the meta data from 'other', corresponding to a single index 
        along a non-slice spatial dimension.'''
        curr_class = self.get_classification(key)
        local_vals = self[curr_class][key]
        other_vals = other._get_changed_class(key, curr_class, self.slice_dim)
        
        #We can't keep track of variations over non-slice spatial dimensions
        if local_vals != other_vals:
            del self[curr_class][key]
            
    def _insert_sample(self, key, other, dim):
        '''Insert the meta data from 'other', corresponding to a single index 
        along an extra-spatial dimension.'''
        curr_class = self.get_classification(key)
        local_vals = self[curr_class][key]
        other_vals = other._get_changed_class(key, curr_class, self.slice_dim)
        shape = self.shape
        n_dims = len(shape)
            
        if curr_class == 'const':
            #If the element was const for both, but values don't match, we 
            #change the classification to per volume/sample
            if local_vals != other_vals:
                #Need to calculate this here instead of using the 
                #get_valid_classes method as the dim we are joining on will be 
                #singular initially
                n_extra_spatial = 0
                for dim_idx in xrange(3, n_dims):
                    if shape[dim_idx] != 1 or dim_idx == dim:
                        n_extra_spatial += 1
                if n_extra_spatial > 1:
                    new_class = 'per_sample_%d' % dim
                else:
                    new_class = 'per_volume'
                self._change_class(key, new_class)
                local_vals = self[new_class][key]
                other_vals = other._get_changed_class(key, 
                                                      new_class,
                                                      self.slice_dim
                                                     )
                local_vals.extend(other_vals)
        elif curr_class == 'per_slice':
            if dim == n_dims - 1:
                #If we are inserting along the last dim, we can just extend 
                #the list of values
                local_vals.extend(other_vals)
            else:
                #Otherwise we need to interleave slices from higher dimensions
                slc_per_samp_self = [self.n_slices]
                for dim_idx in xrange(3, n_dims):
                    slc_per_samp_self.append(slc_per_samp_self[-1] * 
                                             shape[dim_idx])
                n_slice_self = shape[dim] * slc_per_samp_self[dim-3]
                slc_per_samp_other = [other.n_slices]
                for dim_idx in xrange(3, len(other.shape)):
                    slc_per_samp_other.append(slc_per_samp_other[-1] * 
                                              other.shape[dim_idx])
                n_slice_other = other.shape[dim] * slc_per_samp_other[dim-3]
                
                intlv = []
                loc_start = 0
                oth_start = 0
                higher_dim_iters = [xrange(size) for size in shape[dim:]]            
                for higher_indices in itertools.product(*higher_dim_iters):
                    intlv += local_vals[loc_start:loc_start + n_slice_self]
                    intlv += other_vals[oth_start:oth_start + n_slice_other]
                    loc_start += n_slice_self
                    oth_start += n_slice_other
                    
                self['per_slice'][key] = intlv
        elif curr_class == 'per_sample_%d' % dim:
            local_vals.extend(other_vals)
        else:
            #If the classification is not already per_volume change it
            if curr_class != 'per_volume':
                if local_vals == other_vals:
                    return
                self._change_class(key, 'per_volume')
                local_vals = self['per_volume'][key]
                other_vals = other._get_changed_class(key, 
                                                      'per_volume',
                                                      self.slice_dim
                                                     )
                                                     
            if dim == n_dims - 1:
                #If we are inserting along the last dim, we can just extend 
                #the list of values
                local_vals.extend(other_vals)
            else:
                #Otherwise need to interleave per volume meta
                vol_per_samp_self = [1]
                for dim_idx in xrange(3, n_dims):
                    vol_per_samp_self.append(vol_per_samp_self[-1] * 
                                             shape[dim_idx])
                n_vol_self = shape[dim] * vol_per_samp_self[dim-3]
                vol_per_samp_other = [1]
                for dim_idx in xrange(3, len(other.shape)):
                    vol_per_samp_other.append(vol_per_samp_other[-1] * 
                                              other.shape[dim_idx])
                n_vol_other = other.shape[dim] * vol_per_samp_other[dim-3]
                
                intlv = []
                loc_start = 0
                oth_start = 0
                higher_dim_iters = [xrange(size) for size in shape[dim:]]            
                for higher_indices in itertools.product(*higher_dim_iters):
                    intlv += local_vals[loc_start:loc_start + n_vol_self]
                    intlv += other_vals[oth_start:oth_start + n_vol_other]
                    loc_start += n_vol_self
                    oth_start += n_vol_other
                    
                self['per_volume'][key] = intlv
            
class DcmMetaExtension(Nifti1Extension):
    def _unmangle(self, value):
        '''Go from extension data to runtime representation.'''
        #Its not possible to preserve order while loading with python 2.6
        kwargs = {}
        if sys.version_info >= (2, 7):
            kwargs['object_pairs_hook'] = OrderedDict
        return DcmMeta.from_mapping(json.loads(value, **kwargs))
    
    def _mangle(self, value):
        '''Go from runtime representation to extension data.'''
        return json.dumps(value, indent=4)
        
    @classmethod
    def from_dcmmeta(cls, dcmmeta):
        #Call constructor with empty json, then set the runtime representation
        result = cls(dcm_meta_ecode, '{}')
        result._content = dcmmeta
        return result 
            
#Add our extension to nibabel
nb.nifti1.extension_codes.add_codes(((dcm_meta_ecode, 
                                      "dcmmeta", 
                                      DcmMetaExtension),)
                                   )

class MissingExtensionError(Exception):
    '''Exception denoting that there is no DcmMetaExtension in the Nifti header.
    '''
    def __str__(self):
        return 'No dcmmeta extension found.'
                                   
def patch_dcm_ds_is(dcm):
    '''Convert all elements in `dcm` with VR of 'DS' or 'IS' to floats and ints.
    This is a hackish work around for the backwards incompatability of pydicom 
    0.9.7 and should not be needed once nibabel is updated. 
    '''
    for elem in dcm:
        if elem.VM == 1:
            if elem.VR in ('DS', 'IS'):
                if elem.value == '':
                    continue
                if elem.VR == 'DS':
                    elem.VR = 'FD'
                    elem.value = float(elem.value)
                else:
                    elem.VR = 'SL'
                    elem.value = int(elem.value)
        else:
            if elem.VR in ('DS', 'IS'):
                if elem.value == '':
                    continue
                if elem.VR == 'DS':
                    elem.VR = 'FD'
                    elem.value = [float(val) for val in elem.value]
                else:
                    elem.VR = 'SL'
                    elem.value = [int(val) for val in elem.value]
        

class NiftiWrapper(object):
    '''Wraps a Nifti1Image object containing a DcmMeta header extension. 
    Provides access to the meta data and the ability to split or merge the 
    data array while updating the meta data.
    
    Parameters
    ----------
    nii_img : nibabel.nifti1.Nifti1Image
        The Nifti1Image to wrap.
        
    make_empty : bool
        If True an empty DcmMetaExtension will be created if none is found.
        
    Raises
    ------
    MissingExtensionError
        No valid DcmMetaExtension was found. 
    
    ValueError
        More than one valid DcmMetaExtension was found.
    '''

    def __init__(self, nii_img, make_empty=False):
        self.nii_img = nii_img
        hdr = nii_img.get_header()
        self.meta_ext = None
        for extension in hdr.extensions:
            if extension.get_code() == dcm_meta_ecode:
                try:
                    meta_ext = extension.get_content()
                    meta_ext.check_valid()
                except InvalidExtensionError, e:
                    warnings.warn("Found candidate extension, but invalid: %s" % e)
                else:
                    if not self.meta_ext is None:
                        raise ValueError('More than one valid DcmMeta '
                                         'extension found.')
                    self.meta_ext = meta_ext
        if not self.meta_ext:
            if make_empty:
                slice_dim = hdr.get_dim_info()[2]
                self.meta_ext = DcmMeta(self.nii_img.shape, 
                                        hdr.get_best_affine(),
                                        None,
                                        slice_dim)
                hdr.extensions.append(DcmMetaExtension.from_dcmmeta(self.meta_ext))
            else:
                raise MissingExtensionError
    
    def __getitem__(self, key):
        '''Get the value for the given meta data key. Only considers meta data 
        that is globally constant. To access varying meta data you must use the 
        method 'get_meta'.'''
        return self.meta_ext['const'][key]
    
    def meta_valid(self, classification):
        '''Return true if the meta data with the given classification appears 
        to be valid for the wrapped Nifti image. Considers the shape and 
        orientation of the image and the meta data extension.'''
        if classification == 'const':
            return True
            
        img_shape = self.nii_img.get_shape()
        meta_shape = self.meta_ext.shape
        if classification == 'per_volume':
            return img_shape[3:] == meta_shape[3:]
            
        if classification.startswith('per_sample'):
            sample_dim = self.meta_ext.get_per_sample_dim(classification)
            return img_shape[sample_dim] == meta_shape[sample_dim]
            
        hdr = self.nii_img.get_header()
        if self.meta_ext.n_slices != hdr.get_n_slices():
            return False
            
        slice_dim = hdr.get_dim_info()[2]
        slice_dir = self.nii_img.get_affine()[slice_dim, :3]
        slices_aligned = np.allclose(slice_dir, 
                                     self.meta_ext.slice_normal,
                                     atol=1e-6)
                                     
        return meta_shape[3:] == img_shape[3:] and slices_aligned
    
    def get_meta(self, key, index=None, default=None):
        '''Return the meta data value for the provided `key`.
        
        Parameters
        ----------
        key : str
            The meta data key.
            
        index : tuple
            The voxel index we are interested in.
            
        default
            This will be returned if the meta data for `key` is not found.
            
        Returns
        -------
        value
            The meta data value for the given `key` (and optionally `index`)
            
        Notes
        -----
        The meta data that varies will only be considered if the shape and/or 
        slice direction match for the image and DcmMeta object.
        '''
        #Get the value(s) and classification for the key
        classification = self.meta_ext.get_classification(key)
        if classification is None:
            return default
        values = self.meta_ext[classification][key]
        
        #Check if the value is constant
        if classification == 'const':
            return values
            
        #Check if the classification is valid
        if not self.meta_valid(classification):
            return default
        
        #If an index is provided check the varying values
        if not index is None:
            #Test if the index is valid
            shape = self.nii_img.get_shape()
            if len(index) != len(shape):
                raise IndexError('Incorrect number of indices.')
            for dim, ind_val in enumerate(index):
                if not 0 <= ind_val < shape[dim]:
                    raise IndexError('Index is out of bounds.')
            
            #Handle per sample or per volume meta
            if classification.startswith('per_sample'):
                per_sample_dim = self.meta_ext.get_per_sample_dim(classification)
                return values[index[per_sample_dim]]
            
            if classification == 'per_volume':
                val_idx = index[3]
                vols_per_sample = shape[3]
                for dim_idx in xrange(4, len(shape)):
                    val_idx += vols_per_sample * index[dim_idx]
                    vols_per_sample *= shape[dim_idx]
                return values[val_idx]
                
            #Finally try per-slice values
            if classification == 'per_slice':
                slice_dim = self.nii_img.get_header().get_dim_info()[2]
                n_slices = shape[slice_dim]
                val_idx = index[slice_dim]
                for count, idx_val in enumerate(index[3:]):
                    val_idx += idx_val * n_slices
                    n_slices *= shape[count+3]
                return values[val_idx]
            else:
                raise ValueError("Unknown meta data classification: %s" % 
                                 classification)
            
        return default
    
    def remove_extension(self):
        '''Remove the DcmMetaExtension from the header of nii_img. The 
        attribute `meta_ext` will still point to the DcmMeta object.'''
        hdr = self.nii_img.get_header()
        target_idx = None
        for idx, ext in enumerate(hdr.extensions):
            if id(ext.get_content()) == id(self.meta_ext):
                target_idx = idx
                break
        else:
            raise IndexError('Extension not found in header')
        del hdr.extensions[target_idx]
        #Nifti1Image.update_header will increase this if necessary
        hdr['vox_offset'] = 352

    def replace_extension(self, dcmmeta):
        '''Replace the DcmMeta embedded in the Nifti.
        
        Parameters
        ----------
        dcmmeta : DcmMeta
            The new DcmMeta object.
        
        '''
        self.remove_extension()
        self.nii_img.get_header().extensions.append(DcmMetaExtension.from_dcmmeta(dcmmeta))
        self.meta_ext = dcmmeta
    
    def split(self, dim=None):
        '''Generate splits of the array and meta data along the specified 
        dimension.
        
        Parameters
        ----------
        dim : int
            The dimension to split the voxel array along. If None it will 
            split along the last extra spatial dimension or, if there are no 
            extra spatial dimensions, the slice dimension.
            
        Returns
        -------
        result
            Generator which yields a NiftiWrapper result for each index 
            along `dim`.
            
        '''
        shape = self.nii_img.get_shape()
        data = self.nii_img.get_data()
        header = self.nii_img.get_header()
        slice_dim = header.get_dim_info()[2]
        
        #If dim is None, choose the vector/time/slice dim in that order
        if dim is None:
            dim = len(shape) - 1
            if dim == 2:
                if slice_dim is None:
                    raise ValueError("Slice dimension is not known")
                dim = slice_dim
        
        #If we are splitting on a spatial dimension, we need to update the 
        #translation
        trans_update = None
        if dim < 3:
            trans_update = header.get_best_affine()[:3, dim]
        
        split_hdr = header.copy()
        slices = [slice(None)] * len(shape)
        for idx in xrange(shape[dim]):
            #Grab the split data, get rid of trailing singular dimensions
            if dim >= 3 and dim == len(shape) - 1:
                slices[dim] = idx
            else:
                slices[dim] = slice(idx, idx+1)
            
            split_data = data[slices].copy()

            #Update the translation in any affines if needed
            if not trans_update is None and idx != 0:
                qform = split_hdr.get_qform()
                if not qform is None:
                    qform[:3, 3] += trans_update
                    split_hdr.set_qform(qform)
                sform = split_hdr.get_sform()
                if not sform is None:
                    sform[:3, 3] += trans_update
                    split_hdr.set_sform(sform)
            
            #Create the initial Nifti1Image object
            split_nii = nb.Nifti1Image(split_data, 
                                       split_hdr.get_best_affine(), 
                                       header=split_hdr)
            
            #Replace the meta data with the appropriate subset
            meta_dim = dim
            if dim == slice_dim:
                meta_dim = self.meta_ext.slice_dim
            split_meta = self.meta_ext.get_subset(meta_dim, idx)
            result = NiftiWrapper(split_nii)
            result.replace_extension(split_meta)
            
            yield result
            
    def to_filename(self, out_path):
        '''Write out the wrapped Nifti to a file
        
        Parameters
        ----------
        out_path : str
            The path to write out the file to
        
        Notes
        -----
        Will check that the DcmMetaExtension is valid before writing the file.
        '''
        self.meta_ext.check_valid()
        self.nii_img.to_filename(out_path)
    
    @classmethod
    def from_filename(klass, path):
        '''Create a NiftiWrapper from a file.

        Parameters
        ----------
        path : str
            The path to the Nifti file to load. 
        '''
        return klass(nb.load(path))
      
    @classmethod
    def from_dicom_wrapper(klass, dcm_wrp, meta_dict=None):
        '''Create a NiftiWrapper from a nibabel DicomWrapper.
        
        Parameters
        ----------
        dcm_wrap : nicom.dicomwrappers.DicomWrapper
            The dataset to convert into a NiftiWrapper.
            
        meta_dict : dict
            An optional dictionary of meta data extracted from `dcm_data`. See 
            the `extract` module for generating this dict.
           
        '''
        data = dcm_wrp.get_data()
        
        #The Nifti patient space flips the x and y directions
        affine = np.dot(np.diag([-1., -1., 1., 1.]), dcm_wrp.get_affine())

        #Make 2D data 3D
        if len(data.shape) == 2:
            data = data.reshape(data.shape + (1,))
        
        #Create the nifti image and set header data
        nii_img = nb.nifti1.Nifti1Image(data, affine)
        hdr = nii_img.get_header()
        hdr.set_xyzt_units('mm', 'sec')
        dim_info = {'freq' : None, 
                    'phase' : None, 
                    'slice' : 2
                   }
        if hasattr(dcm_wrp.dcm_data, 'InplanePhaseEncodingDirection'):
            if dcm_wrp['InplanePhaseEncodingDirection'] == 'ROW':
                dim_info['phase'] = 1
                dim_info['freq'] = 0
            else:
                dim_info['phase'] = 0
                dim_info['freq'] = 1
        hdr.set_dim_info(**dim_info)
        
        #Embed the meta data extension
        result = klass(nii_img, make_empty=True)
        
        result.meta_ext.reorient_transform = np.eye(4)
        if meta_dict:
            result.meta_ext['const'].update(meta_dict)
        
        return result
      
    @classmethod
    def from_dicom(klass, dcm_data, meta_dict=None):
        '''Create a NiftiWrapper from a single DICOM dataset.
        
        Parameters
        ----------
        dcm_data : dicom.dataset.Dataset
            The DICOM dataset to convert into a NiftiWrapper.
            
        meta_dict : dict
            An optional dictionary of meta data extracted from `dcm_data`. See 
            the `extract` module for generating this dict.
           
        '''
        dcm_wrp = wrapper_from_data(dcm_data)
        return klass.from_dicom_wrapper(dcm_wrp, meta_dict)
        
    @classmethod
    def from_sequence(klass, seq, dim=None):
        '''Create a NiftiWrapper by joining a sequence of NiftiWrapper objects
        along the given dimension. 
        
        Parameters
        ----------
        seq : sequence
            The sequence of NiftiWrapper objects.
            
        dim : int
            The dimension to join the NiftiWrapper objects along. If None, 
            2D inputs will become 3D, 3D inputs will become 4D, and 4D inputs 
            will become 5D.
            
        Returns
        -------
        result : NiftiWrapper
            The merged NiftiWrapper with updated meta data.
        '''
        n_inputs = len(seq)
        first_input = seq[0]
        first_nii = first_input.nii_img
        first_hdr = first_nii.get_header()
        shape = first_nii.shape
        affine = first_nii.get_affine().copy()
        
        #If dim is None, choose a sane default
        if dim is None:
            if len(shape) == 3: 
                singular_dim = None
                for dim_idx, dim_size in enumerate(shape):
                    if dim_size == 1:
                        singular_dim = dim_idx
                if singular_dim is None:
                    dim = 3
                else:
                    dim = singular_dim
            if len(shape) == 4:
                dim = 4
        else:
            if not 0 <= dim < 5:
                raise ValueError("The argument 'dim' must be in the range "
                                 "[0, 5).")
            if dim < len(shape) and shape[dim] != 1:
                raise ValueError('The dimension must be singular or not exist')
                
        #Pull out the three axes vectors for validation of other input affines
        axes = []
        for axis_idx in xrange(3):
            axis_vec = affine[:3, axis_idx]
            if axis_idx == dim:
                axis_vec = axis_vec.copy()
                axis_vec /= np.sqrt(np.dot(axis_vec, axis_vec))
            axes.append(axis_vec)
        #Pull out the translation
        trans = affine[:3, 3]            
        
        #Determine the shape of the result data array and create it
        result_shape = list(shape)
        while dim >= len(result_shape):
            result_shape.append(1)
        result_shape[dim] = n_inputs
        
        result_dtype = max(input_wrp.nii_img.get_data().dtype 
                           for input_wrp in seq)
        result_data = np.empty(result_shape, dtype=result_dtype)
        
        #Start with the header info from the first input
        hdr_info = {'qform' : first_hdr.get_qform(),
                    'qform_code' : first_hdr['qform_code'],
                    'sform' : first_hdr.get_sform(),
                    'sform_code' : first_hdr['sform_code'],
                    'dim_info' : list(first_hdr.get_dim_info()),
                    'xyzt_units' : list(first_hdr.get_xyzt_units()),
                   }
                   
        try:
            hdr_info['slice_duration'] = first_hdr.get_slice_duration()
        except HeaderDataError:
            hdr_info['slice_duration'] = None
        try:
            hdr_info['intent'] = first_hdr.get_intent()
        except HeaderDataError:
            hdr_info['intent'] = None
        try:
            hdr_info['slice_times'] = first_hdr.get_slice_times()
        except HeaderDataError:
            hdr_info['slice_times'] = None
               
        #Fill the data array, check header consistency
        data_slices = [slice(None)] * len(result_shape)
        for dim_idx, dim_size in enumerate(result_shape):
            if dim_size == 1:
                data_slices[dim_idx] = 0
        last_trans = None #Keep track of the translation from last input
        for input_idx in range(n_inputs):
            
            input_wrp = seq[input_idx]
            input_nii = input_wrp.nii_img
            input_aff = input_nii.get_affine()
            input_hdr = input_nii.get_header()
            
            #Check that the affines match appropriately
            for axis_idx, axis_vec in enumerate(axes):
                in_vec = input_aff[:3, axis_idx]
                
                #If we are joining on this dimension
                if axis_idx == dim:
                    #Allow scaling difference as it will be updated later
                    in_vec = in_vec.copy()
                    in_vec /= np.sqrt(np.dot(in_vec, in_vec))
                    
                    in_trans = input_aff[:3, 3]
                    if not last_trans is None:
                        #Must be translated along the axis
                        trans_diff = in_trans - last_trans
                        if not np.allclose(trans_diff, 0.0):
                            trans_diff /= np.sqrt(np.dot(trans_diff, trans_diff))
                        
                        if (np.allclose(trans_diff, 0.0) or 
                            not np.allclose(np.dot(trans_diff, in_vec), 
                                            1.0, 
                                            atol=1e-6)
                           ):
                            raise ValueError("Slices must be translated along the "
                                             "normal direction")
                        
                    #Update reference to last translation
                    last_trans = in_trans
            
                #Check that axis vectors match
                if not np.allclose(in_vec, axis_vec):
                    raise ValueError("Cannot join images with different "
                                     "orientations.")
                                     
            
            data_slices[dim] = input_idx
            result_data[data_slices] = input_nii.get_data().squeeze()
            
            if input_idx != 0:
                if (hdr_info['qform'] is None or 
                    input_hdr.get_qform() is None or
                    not np.allclose(input_hdr.get_qform(), hdr_info['qform'])
                   ):
                    hdr_info['qform'] = None
                if input_hdr['qform_code'] != hdr_info['qform_code']:
                    hdr_info['qform_code'] = None
                if (hdr_info['sform'] is None or 
                    input_hdr.get_sform() is None or
                    not np.allclose(input_hdr.get_sform(), hdr_info['sform'])
                   ):
                    hdr_info['sform'] = None
                if input_hdr['sform_code'] != hdr_info['sform_code']:
                    hdr_info['sform_code'] = None
                in_dim_info = list(input_hdr.get_dim_info())
                if in_dim_info != hdr_info['dim_info']:
                    for idx in xrange(3):
                        if in_dim_info[idx] != hdr_info['dim_info'][idx]:
                            hdr_info['dim_info'][idx] = None
                in_xyzt_units = list(input_hdr.get_xyzt_units())
                if in_xyzt_units != hdr_info['xyzt_units']:
                    for idx in xrange(2):
                        if in_xyzt_units[idx] != hdr_info['xyzt_units'][idx]:
                            hdr_info['xyzt_units'][idx] = None
                    
                try:
                    if input_hdr.get_slice_duration() != hdr_info['slice_duration']:
                        hdr_info['slice_duration'] = None
                except HeaderDataError:
                    hdr_info['slice_duration'] = None
                try:
                    if input_hdr.get_intent() != hdr_info['intent']:
                        hdr_info['intent'] = None
                except HeaderDataError:
                    hdr_info['intent'] = None
                try:
                    if input_hdr.get_slice_times() != hdr_info['slice_times']:
                        hdr_info['slice_times'] = None
                except HeaderDataError:
                    hdr_info['slice_times'] = None
            
        #If we joined along a spatial dim, rescale the appropriate axis
        scaled_dim_dir = None
        if dim < 3:
            scaled_dim_dir = seq[1].nii_img.get_affine()[:3, 3] - trans
            affine[:3, dim] = scaled_dim_dir
            
        #Create the resulting Nifti and wrapper
        result_nii = nb.Nifti1Image(result_data, affine)
        result_hdr = result_nii.get_header()
        
        #Update the header with any info that is consistent across inputs
        if hdr_info['qform'] != None and hdr_info['qform_code'] != None:
            if not scaled_dim_dir is None:
                hdr_info['qform'][:3, dim] = scaled_dim_dir
            result_nii.set_qform(hdr_info['qform'], 
                                 int(hdr_info['qform_code']), 
                                 update_affine=True)
        if hdr_info['sform'] != None and hdr_info['sform_code'] != None:
            if not scaled_dim_dir is None:
                hdr_info['sform'][:3, dim] = scaled_dim_dir
            result_nii.set_sform(hdr_info['sform'], 
                                 int(hdr_info['sform_code']),
                                 update_affine=True)
        if hdr_info['dim_info'] != None:
            result_hdr.set_dim_info(*hdr_info['dim_info'])
            slice_dim = hdr_info['dim_info'][2]
        else:
            slice_dim = None
        if hdr_info['intent'] != None:
            result_hdr.set_intent(*hdr_info['intent'])
        if hdr_info['xyzt_units'] != None:
            result_hdr.set_xyzt_units(*hdr_info['xyzt_units'])
        if hdr_info['slice_duration'] != None:
            result_hdr.set_slice_duration(hdr_info['slice_duration'])
        if hdr_info['slice_times'] != None:
            result_hdr.set_slice_times(hdr_info['slice_times'])
        
        #Create the meta data extension and insert it
        seq_exts = [elem.meta_ext for elem in seq]
        result_meta = DcmMeta.from_sequence(seq_exts, 
                                            dim, 
                                            affine,
                                            slice_dim)
        result_ext = DcmMetaExtension.from_dcmmeta(result_meta)
        result_hdr.extensions.append(result_ext)
        
        return NiftiWrapper(result_nii)
