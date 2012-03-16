"""
Nifti extension for embedding additional meta data and a NiftiWrapper class for
providing access to the meta data and related functionality.

@author: moloney
"""
import json
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import nibabel as nb
from nibabel.nifti1 import Nifti1Extension
from nibabel.spatialimages import HeaderDataError

dcm_meta_ecode = 0

_meta_version = 0.5

def is_constant(sequence, period=None):
    '''Returns true if all elements in the sequence are equal. If period is not
    None then each non-overlapping subsequence of that length is checked.'''
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
        n_dims = len(self.get_shape())
        if n_dims == 3:
            return self.classifications[:2]
        elif n_dims == 4:
            return self.classifications[:4]
        elif n_dims == 5:
            return self.classifications
        else:
            raise ValueError("There must be 3 to 5 dimensions.")
    
    def check_valid(self):
        '''Raise and exception if the extension is not valid. Checks for the 
        required dictionaries and makes sure different classifications of meta 
        data have the correct number of values (multiplicity) for each key.'''
        #Check for the required base keys in the json data
        if not self._req_base_keys <= set(self._content):
            raise InvalidExtensionError('Missing one or more required keys')
            
        #Check the orientation/shape/version
        if self.get_affine().shape != (4, 4):
            raise InvalidExtensionError('Affine has incorrect shape')
        if not (0 <= self.get_slice_dim() < 3):
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
            if cls_mult > 1:
                for key, vals in cls_meta.iteritems():
                    n_vals = len(vals)
                    if n_vals != cls_mult:
                        msg = (('Incorrect number of values for key %s with '
                                'classification %s, expected %d found %d') %
                               (key, classes, cls_mult, n_vals)
                              )
                        raise InvalidExtensionError(msg)
    
    def get_affine(self):
        '''Return the affine associated with the meta data.'''
        return np.array(self._content['dcmmeta_affine'])
        
    def set_affine(self, affine):
        '''Set the affine associated with the meta data.'''
        self._content['dcmmeta_affine'] = affine.tolist()
        
    def get_slice_dim(self):
        '''Get the index of the slice dimension.'''
        return self._content['dcmmeta_slice_dim']
        
    def set_slice_dim(self, dim):
        '''Set the index of the slice dimension.'''
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
        '''Return the version of the meta data extension.'''
        self._content['dcmmeta_version'] = version_number
        
    def get_slice_dir(self):
        '''Return the slice direction vector.'''
        slice_dim = self.get_slice_dim()
        return np.array(self._content['dcmmeta_affine'][slice_dim][:3])
        
    def get_n_slices(self):
        '''Returns the number of slices in each spatial volume.'''
        return self.get_shape()[self.get_slice_dim()]
        
    def get_keys(self):
        '''Get a list of all the meta data keys that are available.'''
        keys = []
        for base_class, sub_class in self.classifications:
            if base_class in self._content:
                keys += self._content[base_class][sub_class].keys()
        return keys

    def get_classification(self, key):
        '''Return the classification tuple for the provided key or None if the
        key is not found.'''
        for base_class, sub_class in self.classifications:
            if base_class in self._content:
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
        
    def clear_slice_meta(self):
        '''Clear all meta data that is per slice.'''
        for base_class, sub_class in self.classifications:
            if base_class in self._content and sub_class == 'slices':
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
        n_slices = self.get_n_slices()
            
        result = OrderedDict()
        result['global'] = OrderedDict()
        result['global']['const'] = deepcopy(self._content['global']['const'])
        result['global']['slices'] = OrderedDict()
        
        if dim < 3 and dim != self.get_slice_dim():
            #Preserve everything
            result['global']['slices'] = \
                deepcopy(self._content['global']['slices'])
            
            if 'time' in self._content:
                result['time'] = deepcopy(self._content['time'])
            
            if 'vector' in self._content:
                result['vector'] = deepcopy(self._content['vector'])
        elif dim < 3:
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
        elif dim == 3:
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
                for key, vals in self._content['global']['slices'].iteritems():
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
        
        elif dim == 4:
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
        
        #Set the "meta meta" data
        result['dcmmeta_affine'] = deepcopy(self._content['dcmmeta_affine'])
        result['dcmmeta_slice_dim'] = deepcopy(self._content['dcmmeta_slice_dim'])
        split_shape = deepcopy(self._content['dcmmeta_shape'])
        split_shape[dim] = 1
        while split_shape[-1] == 1 and len(split_shape) > 3:
            split_shape = split_shape[:-1]
        result['dcmmeta_shape'] = split_shape
        result['dcmmeta_version'] = self.get_version()
        
        return self.from_runtime_repr(result)
        
    @classmethod
    def make_empty(klass, shape, affine, slice_dim):
        result = klass(dcm_meta_ecode, '{}')
        result._content['global'] = OrderedDict()
        result._content['global']['const'] = OrderedDict()
        result._content['global']['slices'] = OrderedDict()
        
        if len(shape) > 3:
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
        them along the dimension index 'dim'. The affine and slice dimension to
        use in the result can optionally be provided, otherwise they will be 
        taken from the first extension in the sequence. '''
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
        for key in result.get_class_dict(('global', 'slices')):
            result._simplify(key)
            
        return result
        
    def __str__(self):
        return self._mangle(self._content)
        
    def __eq__(self, other):
        return self._content == other._content

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
        return json.loads(value)
    
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
        
        #If the class is global const then just delete it if the value is None
        if curr_class == ('global', 'const'):
            if values is None:
                del self.get_class_dict(curr_class)[key]
                return True
            return False
        
        #Test of the values are constant with some period
        for classes in self._const_test_order:
            if classes[0] in self._content:
                mult = self.get_multiplicity(classes)
                if mult == 1:
                    if is_constant(values):
                        self.get_class_dict(classes)[key] = values[0]
                        break
                elif is_constant(values, mult):
                    self.get_class_dict(classes)[key] = values[::mult]
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
        elif classes[1] == 'slices':
            self.get_values(key).extend(other_vals)
        else:
            self._change_class(key, ('global', 'slices'))
            other_vals = other._get_changed_class(key, ('global', 'slices'))
            self.get_values(key).extend(other_vals)
            
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
                other_vals = other._get_changed_class(key, (sample_base, 
                                                            'samples')
                                                     )
                self.get_values(key).extend(other_vals)
        elif classes == (sample_base, 'samples'):
            self.get_values(key).extend(other_vals)
        else:
            self._change_class(key, ('global', 'slices'))
            other_vals = other._get_changed_class(key, ('global', 'slices'))
            self.get_values(key).extend(other_vals)
        
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
            #TODO: update this since we are now using the generic ecode of 0
            if extension.get_code() == dcm_meta_ecode:
                if self._meta_ext:
                    raise ValueError("More than one DcmMetaExtension found")
                self._meta_ext = extension
        if not self._meta_ext:
            raise ValueError("No DcmMetaExtension found.")
        self._meta_ext.check_valid()
    
    def samples_valid(self):
        '''Check if the meta data corresponding to individual time or vector 
        samples appears to be valid for the wrapped nifti image.'''
        #Check if the time/vector dimensions match
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
        return np.allclose(self.nii_img.get_header().get_best_affine(), 
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
        meta_dict = self._meta_ext._content
        
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
                dim = slice_dim
        
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
            split_nii = nb.Nifti1Image(split_data, None)
            split_hdr = split_nii.get_header()
            
            #Update the header
            split_hdr.set_qform(header.get_qform(), int(header['qform_code']))
            split_hdr.set_sform(header.get_sform(), int(header['sform_code']))
            split_hdr.set_slope_inter(*header.get_slope_inter())
            split_hdr.set_dim_info(*header.get_dim_info())
            split_hdr.set_intent(*header.get_intent())
            split_hdr.set_slice_duration(header.get_slice_duration())
            split_hdr.set_xyzt_units(*header.get_xyzt_units())
            
            if dim > 2:
                try:
                    split_hdr.set_slice_times(header.get_slice_times())
                except HeaderDataError:
                    pass
                
            #Insert the subset of meta data
            split_meta = self._meta_ext.get_subset(dim, idx)
            split_hdr.extensions.append(split_meta)
            
            yield NiftiWrapper(split_nii)
    
    def split(self, dim=None):
        '''Convienance method, returns a list containing the results from 
        'generate_splits'.'''
        return list(self.generate_splits(dim))
            
    def to_filename(self, out_path):
        if not self._meta_ext.is_valid:
            raise ValueError("Meta extension is not valid.")
        self.nii_img.to_filename(out_path)
    
    @classmethod
    def from_filename(klass, path):
        return klass(nb.load(path))
        
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
        
        result_data = np.empty(result_shape)
        
        #Start with the header info from the first input
        hdr_info = {'qform' : first_hdr.get_qform(),
                    'qform_code' : first_hdr['qform_code'],
                    'sform' : first_hdr.get_sform(),
                    'sform_code' : first_hdr['sform_code'],
                    'slope_intercept' : first_hdr.get_slope_inter(),
                    'dim_info' : first_hdr.get_dim_info(),
                    #'intent' : first_hdr.get_intent(),
                    'xyzt_units' : first_hdr.get_xyzt_units(),
                    'slice_duration' : first_hdr.get_slice_duration(),
                    #'slice_times' : first_hdr.get_slice_times(),
                   }
        
        #Fill the data array, check header consistency
        slice_dim = None
        data_slices = [slice(None)] * len(result_shape)
        for input_idx in range(n_inputs):
            
            input_wrp = seq[input_idx]
            input_nii = input_wrp.nii_img
            input_hdr = input_nii.get_header()
            
            if not np.allclose(affine, input_hdr.get_best_affine()):
                raise ValueError('Cannot join images with different affines.')
            
            data_slices[dim] = input_idx
            result_data[data_slices] = input_nii.get_data().squeeze()
            
            if slice_dim is None:
                if input_wrp.slices_valid():
                    _, _, slice_dim = input_hdr.get_dim_info()
            
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
                #if input_hdr.get_intent() != hdr_info['intent']:
                #    hdr_info['intent'] = None
                if input_hdr.get_xyzt_units() != hdr_info['xyzt_units']:
                    hdr_info['xyzt_units'] = None
                if input_hdr.get_slice_duration() != hdr_info['slice_duration']:
                    hdr_info['slice_duration'] = None
                #if input_hdr.get_slice_times() != hdr_info['slice_times']:
                #    hdr_info['slice_times'] = None
                
            
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
        #if hdr_info['intent'] != None:
        #    result_hdr.set_intent(hdr_info['intent'])
        if hdr_info['xyzt_units'] != None:
            result_hdr.set_xyzt_units(*hdr_info['xyzt_units'])
        if hdr_info['slice_duration'] != None:
            result_hdr.set_slice_duration(hdr_info['slice_duration'])
        #if hdr_info['slice_times'] != None:
        #    result_hdr.set_slice_times(hdr_info['slice_times'])
        
        #Create the meta data extension and insert it
        seq_exts = [elem._meta_ext for elem in seq]
        result_ext = DcmMetaExtension.from_sequence(seq_exts, dim, affine,
                                                    slice_dim)
        result_hdr.extensions.append(result_ext)
        
        return NiftiWrapper(result_nii)
        