"""
Nifti extension for embedding additional meta data and a NiftiWrapper class for
providing access to the meta data and related functionality.

@author: moloney
"""
import json, warnings
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import nibabel as nb
from nibabel.nifti1 import Nifti1Extension
from nibabel.spatialimages import HeaderDataError

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from nibabel.nicom.dicomwrappers import wrapper_from_data

dcm_meta_ecode = 0

_meta_version = 0.5

def is_constant(sequence, period=None):
    '''Returns true if all elements in the sequence are equal. If period is not
    None then each subsequence of that length is checked.'''
    if period is None:
        return all(val == sequence[0] for val in sequence)
    else:
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
    period.'''
    seq_len = len(sequence)
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
        self.msg = msg
    
    def __str__(self):
        return 'The extension is not valid: %s' % self.msg
        

class DcmMetaExtension(Nifti1Extension):
    '''Nifti extension for storing a complete but concise representation of the 
    meta data from the source DICOM files.
    '''
    
    classifications = (('global', 'const'),
                       ('global', 'slices'),
                       ('time', 'samples'),
                       ('time', 'slices'),
                       ('vector', 'samples'),
                       ('vector', 'slices'),
                      )
    '''The classifications used to seperate meta data based on if and how the
    values repeat. Each class is a tuple with a base class and a sub class.'''
    
    def get_valid_classes(self):
        '''Return the tuples of meta data classifications that are valid for 
        this extension's shape.'''
        shape = self.get_shape()
        n_dims = len(shape)
        if n_dims == 3:
            return self.classifications[:2]
        elif n_dims == 4:
            return self.classifications[:4]
        elif n_dims == 5:
            if shape[3] != 1:
                return self.classifications
            else:
                return self.classifications[:2] + self.classifications[-2:]
        else:
            raise ValueError("There must be 3 to 5 dimensions.")
    
    def check_valid(self):
        '''Raise an InvalidExtensionError if the extension is not valid. Checks 
        for the required elements and makes sure different classifications of 
        meta data have the correct number of values (multiplicity) for each key.
        '''
        #Check for the required base keys in the json data
        if not self._req_base_keys <= set(self._content):
            raise InvalidExtensionError('Missing one or more required keys')
            
        #Check the orientation/shape/version
        if self.get_affine().shape != (4, 4):
            raise InvalidExtensionError('Affine has incorrect shape')
        slice_dim = self.get_slice_dim()
        if slice_dim != None:
            if not (0 <= slice_dim < 3):
                raise InvalidExtensionError('Slice dimension is not valid')
        if not (3 <= len(self.get_shape()) < 6):
            raise InvalidExtensionError('Shape is not valid')
            
        #Check all required meta dictionaries, make sure values have correct
        #multiplicity
        for classes in self.get_valid_classes():
            if not classes[0] in self._content:
                raise InvalidExtensionError('Missing required base '
                                            'classification %s' % classes[0])
            if not classes[1] in self._content[classes[0]]:
                raise InvalidExtensionError(('Missing required sub '
                                             'classification %s in base '
                                             'classification %s') % classes)
            cls_meta = self.get_class_dict(classes)
            cls_mult = self.get_multiplicity(classes)
            if cls_mult == 0 and len(cls_meta) != 0:
                raise InvalidExtensionError('Slice dim is None but per-slice '
                                            'meta data is present')
            elif cls_mult > 1:
                for key, vals in cls_meta.iteritems():
                    n_vals = len(vals)
                    if n_vals != cls_mult:
                        msg = (('Incorrect number of values for key %s with '
                                'classification %s, expected %d found %d') %
                               (key, classes, cls_mult, n_vals)
                              )
                        raise InvalidExtensionError(msg)
    
    def get_affine(self):
        '''Return the affine associated with the per-slice meta data.'''
        return np.array(self._content['dcmmeta_affine'])
        
    def set_affine(self, affine):
        '''Set the affine associated with the per-slice meta data.'''
        self._content['dcmmeta_affine'] = affine.tolist()
        
    def get_slice_dim(self):
        '''Get the index of the slice dimension associated with the per-slice 
        meta data.'''
        return self._content['dcmmeta_slice_dim']
        
    def set_slice_dim(self, dim):
        '''Set the index of the slice dimension associated with the per-slice 
        meta data.'''
        self._content['dcmmeta_slice_dim'] = dim
    
    def get_shape(self):
        '''Return the shape of the data associated with the meta data'''
        return tuple(self._content['dcmmeta_shape'])
        
    def set_shape(self, shape):
        '''Set the shape of the data associated with the meta data'''
        self._content['dcmmeta_shape'][:] = shape
        
    def get_version(self):
        '''Return the version of the meta data extension.'''
        return self._content['dcmmeta_version']
        
    def set_version(self, version_number):
        '''Set the version of the meta data extension.'''
        self._content['dcmmeta_version'] = version_number
        
    def get_slice_dir(self):
        '''Return the slice direction vector.'''
        slice_dim = self.get_slice_dim()
        if slice_dim is None:
            return None
        return np.array(self._content['dcmmeta_affine'][slice_dim][:3])
        
    def get_n_slices(self):
        '''Returns the number of slices in each spatial volume.'''
        slice_dim = self.get_slice_dim()
        if slice_dim is None:
            return None
        return self.get_shape()[slice_dim]
        
    def get_keys(self):
        '''Get a list of all the meta data keys that are available.'''
        keys = []
        for base_class, sub_class in self.get_valid_classes():
            keys += self._content[base_class][sub_class].keys()
        return keys

    def get_classification(self, key):
        '''Return the classification tuple for the provided key or None if the
        key is not found.'''
        for base_class, sub_class in self.get_valid_classes():
            if key in self._content[base_class][sub_class]:
                    return (base_class, sub_class)
                    
        return None
    
    def get_class_dict(self, classification):
        '''Return the meta dictionary for the given classification'''
        base, sub = classification
        return self._content[base][sub]
        
    def get_values(self, key):
        '''Get all meta data values for the provided key. The number of values
        depends on the classification (see 'get_multiplicity').'''
        classification = self.get_classification(key)
        return self.get_class_dict(classification)[key]
    
    def get_values_and_class(self, key):
        '''Return a tuple containing the values and the classification for the
        provided key. Returns None for both the value and classification if the
        key is not found.'''
        classification = self.get_classification(key)
        if classification is None:
            return (None, None)
        return (self.get_class_dict(classification)[key], classification)
        
    def filter_meta(self, filter_func):
        for classes in self.get_valid_classes():
            filtered = []
            curr_dict = self.get_class_dict(classes)
            for key, values in curr_dict.iteritems():
                if filter_func(key, values):
                    filtered.append(key)
            for key in filtered:
                del curr_dict[key]
        
    def clear_slice_meta(self):
        '''Clear all meta data that is per slice.'''
        for base_class, sub_class in self.get_valid_classes():
            if sub_class == 'slices':
                self.get_class_dict((base_class, sub_class)).clear()
    
    def get_multiplicity(self, classification):
        '''Return the number of meta data values for each key of the given 
        classification.
        '''
        if not classification in self.get_valid_classes():
            raise ValueError("Invalid classification: %s" % classification)
        
        base, sub = classification
        shape = self.get_shape()
        n_vals = 1
        if sub == 'slices':
            n_vals = self.get_n_slices()
            if n_vals is None:
                return 0
            if base == 'vector':
                n_vals *= shape[3]
            elif base == 'global':
                for dim_size in shape[3:]:
                    n_vals *= dim_size
        elif sub == 'samples':
            if base == 'time':
                n_vals = shape[3]
            elif base == 'vector':
                n_vals = shape[4]
                
        return n_vals
    
    def get_subset(self, dim, idx):
        '''Return a new DcmMetaExtension containing the subset of the meta data 
        corresponding to the index 'idx' along the dimension 'dim'.
        '''
        if not 0 <= dim < 5:
            raise ValueError("The argument 'dim' must be in the range [0, 5).")
        
        shape = self.get_shape()
        valid_classes = self.get_valid_classes()
        
        #Make an empty extension for the result
        result_shape = list(shape)
        result_shape[dim] = 1
        while result_shape[-1] == 1 and len(result_shape) > 3:
            result_shape = result_shape[:-1]
        result = self.make_empty(result_shape, 
                                 self.get_affine(), 
                                 self.get_slice_dim()
                                )
        
        for src_class in valid_classes:
            #Constants remain constant
            if src_class == ('global', 'const'):
                for key, val in self.get_class_dict(src_class).iteritems():
                    result.get_class_dict(src_class)[key] = deepcopy(val)
                continue
            
            if dim == self.get_slice_dim():
                if src_class[1] != 'slices':
                    for key, vals in self.get_class_dict(src_class).iteritems():
                        result.get_class_dict(src_class)[key] = deepcopy(vals)
                else:
                    result._copy_slice(self, src_class, idx)
            elif dim < 3:
                for key, vals in self.get_class_dict(src_class).iteritems():
                    result.get_class_dict(src_class)[key] = deepcopy(vals)
            elif dim == 3:
                result._copy_sample(self, src_class, 'time', idx)
            else:
                result._copy_sample(self, src_class, 'vector', idx)
                
        return result
        
    @classmethod
    def make_empty(klass, shape, affine, slice_dim=None):
        result = klass(dcm_meta_ecode, '{}')
        result._content['global'] = OrderedDict()
        result._content['global']['const'] = OrderedDict()
        result._content['global']['slices'] = OrderedDict()
        
        if len(shape) > 3 and shape[3] != 1:
            result._content['time'] = OrderedDict()
            result._content['time']['samples'] = OrderedDict()
            result._content['time']['slices'] = OrderedDict()
            
        if len(shape) > 4:
            result._content['vector'] = OrderedDict()
            result._content['vector']['samples'] = OrderedDict()
            result._content['vector']['slices'] = OrderedDict()
        
        result._content['dcmmeta_shape'] = []
        result.set_shape(shape)
        result.set_affine(affine)
        result.set_slice_dim(slice_dim)
        result.set_version(_meta_version)
        
        return result

    def to_json(self):
        '''Return the JSON string representation of the extension.'''
        self.check_valid()
        return self._mangle(self._content)
        
    @classmethod
    def from_json(klass, json_str):
        '''Create an extension from the JSON string representation.'''
        result = klass(dcm_meta_ecode, json_str)
        result.check_valid()
        return result
        
    @classmethod
    def from_runtime_repr(klass, runtime_repr):
        '''Create an extension from the Python runtime representation.'''
        result = klass(dcm_meta_ecode, '{}')
        result._content = runtime_repr
        result.check_valid()
        return result
        
    @classmethod
    def from_sequence(klass, seq, dim, affine=None, slice_dim=None):
        '''Create an extension from the sequence of extensions 'seq' by joining
        them along the dimension with index 'dim'. The affine and slice 
        dimension to use in the result can optionally be provided, otherwise 
        they will be taken from the first extension in the sequence. '''
        if not 0 <= dim < 5:
            raise ValueError("The argument 'dim' must be in the range [0, 5).")
        
        n_inputs = len(seq)
        first_input = seq[0]
        input_shape = first_input.get_shape()
        
        if len(input_shape) > dim and input_shape[dim] != 1:
            raise ValueError("The dim must be singular or not exist for the"
                             "inputs.")
        
        output_shape = list(input_shape)
        while len(output_shape) <= dim:
            output_shape.append(1)
        output_shape[dim] = n_inputs
        
        if affine is None:
            affine = first_input.get_affine()
        if slice_dim is None:
            slice_dim = first_input.get_slice_dim()
            
        result = klass.make_empty(output_shape, affine, slice_dim)
        
        #Need to initialize the result with the first extension in 'seq'
        use_slices = np.allclose(result.get_slice_dir(), 
                                 first_input.get_slice_dir())
        for classes in first_input.get_valid_classes():
            if classes[1] == 'slices' and not use_slices:
                continue
            result._content[classes[0]][classes[1]] = \
                deepcopy(first_input.get_class_dict(classes))
        
        #Adjust the shape to what the extension actually contains
        shape = list(result.get_shape())
        shape[dim] = 1
        result.set_shape(shape)
        
        #Add the other extensions, updating the shape as we go
        for input_ext in seq[1:]:
            result._insert(dim, input_ext)
            shape[dim] += 1
            result.set_shape(shape)
            
        #Try simplifying any keys in global slices
        for key in result.get_class_dict(('global', 'slices')).keys():
            result._simplify(key)
            
        return result
        
    def __str__(self):
        return self._mangle(self._content)
        
    def __eq__(self, other):
        if not np.allclose(self.get_affine(), other.get_affine()):
            return False
        if self.get_shape() != other.get_shape():
            return False
        if self.get_slice_dim() != other.get_slice_dim():
            return False
        if self.get_version() != other.get_version():
            return False
        for classes in self.get_valid_classes():
            if (dict(self.get_class_dict(classes)) != 
               dict(other.get_class_dict(classes))):
                return False
                
        return True

    _req_base_keys = set(('dcmmeta_affine', 
                          'dcmmeta_slice_dim',
                          'dcmmeta_shape',
                          'dcmmeta_version',
                          'global',
                         )
                        )

    _preserving_changes = {None : (('vector', 'samples'),
                                   ('time', 'samples'),
                                   ('time', 'slices'),
                                   ('vector', 'slices'),
                                   ('global', 'slices'),
                                  ),
                           ('global', 'const') : (('vector', 'samples'),
                                                  ('time', 'samples'),
                                                  ('time', 'slices'),
                                                  ('vector', 'slices'),
                                                  ('global', 'slices'),
                                                 ),
                           ('vector', 'samples') : (('time', 'samples'),
                                                    ('global', 'slices'),
                                                   ),
                           ('time', 'samples') : (('global', 'slices'),
                                                 ),
                           ('time', 'slices') : (('vector', 'slices'),
                                                 ('global', 'slices'),
                                                ),
                           ('vector', 'slices') : (('global', 'slices'),
                                                  ),
                           ('global', 'slices') : tuple(),
                          }
                          
    _const_test_order = (('global', 'const'),
                         ('vector', 'samples'),
                         ('time', 'samples'),
                        )

    _repeat_test_order = (('time', 'slices'),
                          ('vector', 'slices'),
                         )
    
    def _unmangle(self, value):
        return json.loads(value, object_pairs_hook=OrderedDict)
    
    def _mangle(self, value):
        return json.dumps(value, indent=4)
        
    def _get_changed_class(self, key, new_class):
        values, curr_class = self.get_values_and_class(key)
        if curr_class == new_class:
            return values
            
        if not new_class in self._preserving_changes[curr_class]:
            raise ValueError("Classification change would lose data.")
        
        if curr_class is None:
            curr_mult = 1
        else:
            curr_mult = self.get_multiplicity(curr_class)
        if new_class in self.get_valid_classes():
            new_mult = self.get_multiplicity(new_class)
        else:
            new_mult = 1
        mult_fact = new_mult / curr_mult
        if curr_mult == 1:
            values = [values] 
            
        result = []
        for value in values:
            result.extend([deepcopy(value)] * mult_fact)
        return result
        
        
    def _change_class(self, key, new_class):
        values, curr_class = self.get_values_and_class(key)
        if curr_class == new_class:
            return
        
        self.get_class_dict(new_class)[key] = self._get_changed_class(key, 
                                                                      new_class)
        
        if not curr_class is None:
            del self.get_class_dict(curr_class)[key]
    
    def _simplify(self, key):
        values, curr_class = self.get_values_and_class(key)
        curr_mult = self.get_multiplicity(curr_class)
        
        #If the class is global const then just delete it if the value is None
        if curr_class == ('global', 'const'):
            if values is None:
                del self.get_class_dict(curr_class)[key]
                return True
            return False
        
        #Test if the values are constant with some period
        for classes in self._const_test_order:
            if classes != curr_class and classes[0] in self._content:
                mult = self.get_multiplicity(classes)
                reduce_factor = curr_mult / mult
                if mult == 1:
                    if is_constant(values):
                        self.get_class_dict(classes)[key] = values[0]
                        break
                elif is_constant(values, reduce_factor):
                    self.get_class_dict(classes)[key] = values[::reduce_factor]
                    break
        else: #Otherwise test if they are repeating with some period
            for classes in self._repeat_test_order:
                if classes[0] in self._content:
                    mult = self.get_multiplicity(classes)
                    if is_repeating(values, mult):
                        self.get_class_dict(classes)[key] = values[:mult]
                        break
            else: #Can't simplify
                return False
            
        del self.get_class_dict(curr_class)[key]
        return True
    
    def _copy_slice(self, other, src_class, idx):
        if src_class[0] == 'global':
            for classes in (('time', 'samples'),
                            ('vector', 'samples'),
                            ('global', 'const')):
                if classes in self.get_valid_classes():
                    dest_class = classes
                    break
        elif src_class[0] == 'vector':
            for classes in (('time', 'samples'),
                            ('global', 'const')):
                if classes in self.get_valid_classes():
                    dest_class = classes
                    break
        else:
            dest_class = ('global', 'const')
            
        stride = other.get_n_slices()
        for key, vals in other.get_class_dict(src_class).iteritems():
            subset_vals = vals[idx::stride]
            if len(subset_vals) == 1:
                subset_vals = subset_vals[0]
            self.get_class_dict(dest_class)[key] = deepcopy(subset_vals)
            self._simplify(key)

    def _global_slice_subset(self, key, sample_base, idx):
        n_slices = self.get_n_slices()
        shape = self.get_shape()
        src_dict = self.get_class_dict(('global', 'slices'))
        if sample_base == 'vector':
            slices_per_vec = n_slices * shape[3]
            start_idx = idx * slices_per_vec
            end_idx = start_idx + slices_per_vec
            return src_dict[key][start_idx:end_idx]
        else:
            if not ('vector', 'samples') in self.get_valid_classes():
                start_idx = idx * n_slices
                end_idx = start_idx + n_slices
                return src_dict[key][start_idx:end_idx]
            else:
                result = []
                slices_per_vec = n_slices * shape[3]
                for vec_idx in shape[4]:
                    start_idx = (vec_idx * slices_per_vec) + (idx * n_slices)
                    end_idx = start_idx + n_slices
                    result.extend(src_dict[key][start_idx:end_idx])
                return result
    
    def _copy_sample(self, other, src_class, sample_base, idx):
        src_dict = other.get_class_dict(src_class)
        if src_class[1] == 'samples':
            if src_class[0] == sample_base:
                for key, vals in src_dict.iteritems():
                    self.get_class_dict(('global', 'const'))[key] = \
                        deepcopy(vals[idx])
            else:
                for key, vals in src_dict.iteritems():
                    self.get_class_dict(src_class)[key] = deepcopy(vals)
        else:
            if src_class[0] == sample_base:
                best_dest = None
                for dest_class in self._preserving_changes[src_class]:
                    if dest_class in self.get_valid_classes():
                        best_dest = dest_class
                        break     
                for key, vals in src_dict.iteritems():
                    self.get_class_dict(best_dest)[key] = deepcopy(vals)
            elif src_class[0] != 'global':
                if sample_base == 'time':
                    #Take a subset of vector slices
                    n_slices = self.get_n_slices()
                    start_idx = idx * n_slices
                    end_idx = start_idx + n_slices
                    for key, vals in src_dict.iteritems():
                        self.get_class_dict(src_class)[key] = \
                            deepcopy(vals[start_idx:end_idx])
                        self._simplify(key)
                else:
                    #Time slices are unchanged
                    for key, vals in src_dict.iteritems():
                        self.get_class_dict(src_class)[key] = deepcopy(vals)
            else:
                #Take a subset of global slices
                for key, vals in src_dict.iteritems():
                    subset_vals = \
                        other._global_slice_subset(key, sample_base, idx)
                    self.get_class_dict(src_class)[key] = deepcopy(subset_vals)
                    self._simplify(key)
    
    def _insert(self, dim, other):
        use_slices = np.allclose(other.get_slice_dir(), self.get_slice_dir())
        missing_keys = list(set(self.get_keys()) - set(other.get_keys()))
        for other_classes in other.get_valid_classes():
            if other_classes[1] == 'slices' and not use_slices:
                continue
            
            other_keys = other.get_class_dict(other_classes).keys()
            
            #Treat missing keys as if they were in global const and have a value
            #of None
            if other_classes == ('global', 'const'):
                other_keys += missing_keys
            
            #When possible, reclassify our meta data so it matches the other
            #classificatoin
            for key in other_keys:
                local_classes = self.get_classification(key)
                if local_classes != other_classes:
                    local_allow = self._preserving_changes[local_classes]
                    other_allow = self._preserving_changes[other_classes]
                    if other_classes in local_allow:
                        self._change_class(key, other_classes)
                    elif not local_classes in other_allow:
                        best_dest = None
                        for dest_class in local_allow:
                            if (dest_class[0] in self._content and 
                               dest_class in other_allow):
                                best_dest = dest_class
                                break
                        self._change_class(key, best_dest)
                        
            #Insert new meta data and further reclassify as necessary
            for key in other_keys:
                if dim == self.get_slice_dim():
                    self._insert_slice(key, other)
                elif dim < 3:
                    self._insert_non_slice(key, other)
                elif dim == 3:
                    self._insert_sample(key, other, 'time')
                elif dim == 4:
                    self._insert_sample(key, other, 'vector')
               
    def _insert_slice(self, key, other):
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, classes)

        #Handle some common / simple insertions with special cases
        if classes == ('global', 'const'):
            if local_vals != other_vals:
                for dest_base in ('time', 'vector', 'global'):
                    if dest_base in self._content:
                        self._change_class(key, (dest_base, 'slices'))
                        other_vals = other._get_changed_class(key, 
                                                              (dest_base, 
                                                               'slices')
                                                             )
                        self.get_values(key).extend(other_vals)
                        break
        elif classes == ('time', 'slices'):
            local_vals.extend(other_vals)
        else:
            #Default to putting in global slices and simplifying later
            if classes != ('global', 'slices'):
                self._change_class(key, ('global', 'slices'))
                local_vals = self.get_class_dict(('global', 'slices'))[key]
                other_vals = other._get_changed_class(key, ('global', 'slices'))
            
            #Need to interleave slices from different volumes
            n_slices = self.get_n_slices()
            other_n_slices = other.get_n_slices()
            shape = self.get_shape()
            n_vols = 1
            for dim_size in shape[3:]:
                n_vols *= dim_size
            
            intlv = []
            loc_start = 0
            oth_start = 0
            for vol_idx in xrange(n_vols):
                intlv += local_vals[loc_start:loc_start + n_slices]
                intlv += other_vals[oth_start:oth_start + other_n_slices]
                loc_start += n_slices
                oth_start += other_n_slices
                
            self.get_class_dict(('global', 'slices'))[key] = intlv
            
    def _insert_non_slice(self, key, other):
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, classes)
        
        if local_vals != other_vals:
            del self.get_class_dict(classes)[key]
            
    def _insert_sample(self, key, other, sample_base):
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, classes)
        
        if classes == ('global', 'const'):
            if local_vals != other_vals:
                self._change_class(key, (sample_base, 'samples'))
                local_vals = self.get_values(key)
                other_vals = other._get_changed_class(key, (sample_base, 
                                                            'samples')
                                                     )
                local_vals.extend(other_vals)
        elif classes == (sample_base, 'samples'):
            local_vals.extend(other_vals)
        else:
            if classes != ('global', 'slices'):
                self._change_class(key, ('global', 'slices'))
                local_vals = self.get_values(key)
                other_vals = other._get_changed_class(key, ('global', 'slices'))
            
            shape = self.get_shape()
            n_dims = len(shape)
            if sample_base == 'time' and n_dims == 5:
                #Need to interleave values from the time points in each vector 
                #component
                n_slices = self.get_n_slices()
                slices_per_vec = n_slices * shape[3]
                oth_slc_per_vec = n_slices * other.get_shape()[3]
                
                intlv = []
                loc_start = 0
                oth_start = 0
                for vec_idx in xrange(shape[4]):
                    intlv += local_vals[loc_start:loc_start+slices_per_vec]
                    intlv += other_vals[oth_start:oth_start+oth_slc_per_vec]
                    loc_start += slices_per_vec
                    oth_start += oth_slc_per_vec
                    
                self.get_class_dict(('global', 'slices'))[key] = intlv
            else:
                local_vals.extend(other_vals)
            
#Add our extension to nibabel
nb.nifti1.extension_codes.add_codes(((dcm_meta_ecode, 
                                      "dcmmeta", 
                                      DcmMetaExtension),)
                                   )

class MissingExtensionError(Exception):
    def __str__(self):
        return 'No dcmmeta extension found.'
                                   
def patch_dcm_ds_is(dcm):
    '''Convert all elements with VR of 'DS' or 'IS' to floats and ints. This is 
    a hackish work around for the backwards incompatability of pydicom 0.9.7 
    and should not be needed once nibabel is updated. 
    '''
    for elem in dcm:
        if elem.VM == 1:
            if elem.VR == 'DS':
                elem.VR = 'FD'
                elem.value = float(elem.value)
            elif elem.VR == 'IS':
                elem.VR = 'SL'
                elem.value = int(elem.value)
        else:
            if elem.VR == 'DS':
                elem.VR = 'FD'
                elem.value = [float(val) for val in elem.value]
            elif elem.VR == 'IS':
                elem.VR = 'SL'
                elem.value = [int(val) for val in elem.value]
    

class NiftiWrapper(object):
    '''Wraps a nibabel.Nifti1Image object containing a DcmMetaExtension header 
    extension. Provides access to the meta data through the method 'get_meta'. 
    Allows the Nifti to be split into sub volumes or merged with others, while 
    also updating the meta data appropriately.'''

    def __init__(self, nii_img, make_empty=False):
        '''Initialize wrapper from Nifti1Image object. Looks for a valid dcmmeta
        extension. If no extension is found a MissingExtensionError will be 
        raised unless 'make_empty' is True, in which case an empty extension 
        will be created. If more than one valid extension is found a ValueError
        will be raised.
        '''
        self.nii_img = nii_img
        hdr = nii_img.get_header()
        self.meta_ext = None
        for extension in hdr.extensions:
            if extension.get_code() == dcm_meta_ecode:
                try:
                    extension.check_valid()
                except InvalidExtensionError, e:
                    print "Found candidate extension, but invalid: %s" % e
                else:
                    if not self.meta_ext is None:
                        raise ValueError('More than one valid DcmMeta '
                                         'extension found.')
                    self.meta_ext = extension
        if not self.meta_ext:
            if make_empty:
                slice_dim = hdr.get_dim_info()[2]
                self.meta_ext = \
                    DcmMetaExtension.make_empty(self.nii_img.shape, 
                                                self.nii_img.get_affine(),
                                                slice_dim)
                hdr.extensions.append(self.meta_ext)
            else:
                raise MissingExtensionError
        self.meta_ext.check_valid()
    
    def __getitem__(self, key):
        '''Get the value for the given meta data key. Only considers meta data 
        that is globally constant. To access varying meta data you must use the 
        method 'get_meta'.'''
        return self.meta_ext.get_class_dict(('global', 'const'))[key]
    
    def samples_valid(self):
        '''Check if the meta data corresponding to individual time or vector 
        samples appears to be valid for the wrapped nifti image.'''
        img_shape = self.nii_img.get_shape()
        meta_shape = self.meta_ext.get_shape()
        return meta_shape[3:] == img_shape[3:]
    
    def slices_valid(self):
        '''Check if the meta data corresponding to individual slices appears to 
        be valid for the wrapped nifti image.'''
        hdr = self.nii_img.get_header()
        if self.meta_ext.get_n_slices() != hdr.get_n_slices():
            return False
        
        slice_dim = hdr.get_dim_info()[2]
        if slice_dim is None:
            return False
        slice_dir = hdr.get_best_affine()[slice_dim, :3]
        return np.allclose(slice_dir, 
                           self.meta_ext.get_slice_dir())
    
    def get_meta(self, key, index=None, default=None):
        '''Return the meta data value for the provided 'key', or 'default' if 
        there is no such (valid) key.
        
        If 'index' is not provided, only meta data values that are constant 
        across the entire data set will be considered. If 'index' is provided it 
        must be a valid index for the nifti voxel data. All of the meta data 
        that is applicable to that index will be considered. The per-slice and 
        per-sample meta data will only be considered if the object's 
        'slices_valid' and 'samples_valid' methods (respectively) return True.
        '''
        #Get the value(s) and classification for the key
        values, classes = self.meta_ext.get_values_and_class(key)
        if classes is None:
            return default
        
        #Check if the value is constant
        if classes == ('global', 'const'):
            return values
        
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
                if classes == ('time', 'samples'):
                    return values[index[3]]
                if classes == ('vector', 'samples'):
                    return values[index[4]]
                
            #Finally, if aligned, try per-slice values
            if self.slices_valid():
                slice_dim = self.nii_img.get_header().get_dim_info()[2]
                n_slices = shape[slice_dim]
                if classes == ('global', 'slices'):
                    val_idx = index[slice_dim]
                    for count, idx_val in enumerate(index[3:]):
                        val_idx += idx_val * n_slices
                        n_slices *= shape[count+3]
                    return values[val_idx]    
                
                if self.samples_valid():
                    if classes == ('time', 'slices'):
                        val_idx = index[slice_dim]
                        return values[val_idx]
                    if classes == ('vector', 'slices'):
                        val_idx = index[slice_dim]
                        val_idx += index[3]*n_slices
                        return values[val_idx]
            
        return default
    
    def remove_extension(self):
        '''Remove the extension from the header of nii_img. The attribute 
        meta_ext will still point to the extension.'''
        hdr = self.nii_img.get_header()
        target_idx = None
        for idx, ext in enumerate(hdr.extensions):
            if id(ext) == id(self.meta_ext):
                target_idx = idx
                break
        else:
            raise IndexError('Extension not found in header')
        del hdr.extensions[target_idx]
        #Nifti1Image.update_header will increase this if necessary
        hdr['vox_offset'] = 352

    def replace_extension(self, dcmmeta_ext):
        '''Remove the existing extension and replace it with the provided one. 
        The attribute meta_ext will be update to the new extension.
        '''
        self.remove_extension()
        self.nii_img.get_header().extensions.append(dcmmeta_ext)
        self.meta_ext = dcmmeta_ext
    
    def generate_splits(self, dim=None):
        '''Generator method that splits the array and meta data along the 
        dimension 'dim', yielding a NiftiWrapper object for each subset of the 
        data. If 'dim' is None it will prefer the vector, then time, then slice 
        dimensions.
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
        
        split_hdr = header.copy()
        slices = [slice(None)] * len(shape)
        for idx in xrange(shape[dim]):
            #Grab the split data, keeping singular spatial dimensions
            if dim < 3:
                slices[dim] = slice(idx, idx+1)
            else:
                slices[dim] = idx
            split_data = data[slices].copy()
            
            #Create the initial Nifti1Image object
            #TODO: Nibabel 1.2 will allow us to set the affine here without 
            #wiping out qform/sform and associated codes
            split_nii = nb.Nifti1Image(split_data, None, header=split_hdr)
            
            #Replace the meta data with the appropriate subset
            meta_dim = dim
            if dim == slice_dim:
                meta_dim = self.meta_ext.get_slice_dim()
            split_meta = self.meta_ext.get_subset(meta_dim, idx)
            result = NiftiWrapper(split_nii)
            result.replace_extension(split_meta)
            
            yield result
    
    def split(self, dim=None):
        '''Convienance method, returns a list containing the results from 
        'generate_splits'.'''
        return list(self.generate_splits(dim))
            
    def to_filename(self, out_path):
        self.meta_ext.check_valid()
        self.nii_img.to_filename(out_path)
    
    @classmethod
    def from_filename(klass, path):
        return klass(nb.load(path))
        
    @classmethod
    def from_dicom(klass, dcm_data, meta_dict=None):
        '''Create a NiftiWrapper from the DICOM data set 'dcm_data'. 
        
        A meta data dict 'meta_dict' can be provided to embed into the dcmmeta 
        extension. To generate such a dict refer to the module dcmstack.extract.
        
        Use dcmstack.parse_and_stack to convert a collection of DICOM images.'''
        #Work around until nibabel supports pydicom >= 0.9.7
        patch_dcm_ds_is(dcm_data)
        
        dcm_wrp = wrapper_from_data(dcm_data)
        data = dcm_wrp.get_data()
        
        #Make 2D data 3D
        if len(data.shape) == 2:
            data = data.reshape(data.shape + (1,))
        
        #Create the nifti image and set header data
        nii_img = nb.nifti1.Nifti1Image(data, None)
        hdr = nii_img.get_header()
        hdr.set_qform(dcm_wrp.get_affine(), 'scanner')
        nii_img._affine = hdr.get_best_affine()
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
        if meta_dict:
            result.meta_ext.get_class_dict(('global', 'const')).update(meta_dict)
        
        return result
        
    @classmethod
    def from_sequence(klass, seq, dim=None):
        '''Create a NiftiWrapper from a sequence of other NiftiWrapper objects.
        The Nifti volumes are stacked along the dimension 'dim' in the given 
        order. If 'dim' is None then 2D inputs will become 3D, 3D inputs will
        be stacked along the fourth (time) dimension, and 4D inputs will be 
        stacked along the fifth (vector) dimension.
        '''
        n_inputs = len(seq)
        first_input = seq[0]
        first_nii = first_input.nii_img
        first_hdr = first_nii.get_header()
        shape = first_nii.shape
        affine = first_hdr.get_best_affine()
        
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
                    'slope_intercept' : first_hdr.get_slope_inter(),
                    'dim_info' : first_hdr.get_dim_info(),
                    'xyzt_units' : first_hdr.get_xyzt_units(),
                    'slice_duration' : first_hdr.get_slice_duration(),
                   }
                   
        try:
            hdr_info['intent'] = first_hdr.get_intent()
        except Exception:
            hdr_info['intent'] = None
        try:
            hdr_info['slice_times'] = first_hdr.get_slice_times()
        except Exception:
            hdr_info['slice_times'] = None
               
        #Fill the data array, check header consistency
        data_slices = [slice(None)] * len(result_shape)
        for dim_idx, dim_size in enumerate(result_shape):
            if dim_size == 1:
                data_slices[dim_idx] = 0
        for input_idx in range(n_inputs):
            
            input_wrp = seq[input_idx]
            input_nii = input_wrp.nii_img
            input_hdr = input_nii.get_header()
            
            if not np.allclose(affine, input_hdr.get_best_affine()):
                raise ValueError('Cannot join images with different affines.')
            
            data_slices[dim] = input_idx
            result_data[data_slices] = input_nii.get_data().squeeze()
            
            if input_idx != 0:
                if not np.allclose(input_hdr.get_qform(), hdr_info['qform']):
                    hdr_info['qform'] = None
                if input_hdr['qform_code'] != hdr_info['qform_code']:
                    hdr_info['qform_code'] = None
                if not np.allclose(input_hdr.get_sform(), hdr_info['sform']):
                    hdr_info['sform'] = None
                if input_hdr['sform_code'] != hdr_info['sform_code']:
                    hdr_info['sform_code'] = None
                if input_hdr.get_slope_inter() != hdr_info['slope_intercept']:
                    hdr_info['slope_intercept'] = None
                if input_hdr.get_dim_info() != hdr_info['dim_info']:
                    hdr_info['dim_info'] = None
                if input_hdr.get_xyzt_units() != hdr_info['xyzt_units']:
                    hdr_info['xyzt_units'] = None
                if input_hdr.get_slice_duration() != hdr_info['slice_duration']:
                    hdr_info['slice_duration'] = None
                if input_hdr.get_intent() != hdr_info['intent']:
                    hdr_info['intent'] = None
                try:
                    if input_hdr.get_slice_times() != hdr_info['slice_times']:
                        hdr_info['slice_times'] = None
                except Exception:
                    hdr_info['slice_times'] = None
            
        #Create the resulting Nifti and wrapper
        result_nii = nb.Nifti1Image(result_data, None)
        result_hdr = result_nii.get_header()
        
        #Update the header with any info that is consistent across inputs
        if hdr_info['qform'] != None and hdr_info['qform_code'] != None:
            result_hdr.set_qform(hdr_info['qform'], int(hdr_info['qform_code']))
        if hdr_info['sform'] != None and hdr_info['sform_code'] != None:
            result_hdr.set_sform(hdr_info['sform'], int(hdr_info['sform_code']))
        if hdr_info['slope_intercept'] != None:
            result_hdr.set_slope_inter(*hdr_info['slope_intercept'])
        if hdr_info['dim_info'] != None:
            result_hdr.set_dim_info(*hdr_info['dim_info'])
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
        result_ext = DcmMetaExtension.from_sequence(seq_exts, dim, affine,
                                                    hdr_info['dim_info'][2])
        result_hdr.extensions.append(result_ext)
        
        return NiftiWrapper(result_nii)
        