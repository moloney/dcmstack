"""
DcmMeta header extension and NiftiWrapper for working with extended Niftis.
"""
from __future__ import print_function

import sys
import json, warnings
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

_meta_version = 0.6

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
                    }
'''Minimum required keys in the base dictionaty to be considered valid'''

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


class DcmMetaExtension(Nifti1Extension):
    '''Nifti extension for storing a summary of the meta data from the source
    DICOM files.
    '''

    @property
    def reorient_transform(self):
        '''The transformation due to reorientation of the data array. Can be
        used to update directional DICOM meta data (after converting to RAS if
        needed) into the same space as the affine.'''
        if self.version < 0.6:
            return None
        if self._content['dcmmeta_reorient_transform'] is None:
            return None
        return np.array(self._content['dcmmeta_reorient_transform'])

    @reorient_transform.setter
    def reorient_transform(self, value):
        if not value is None and value.shape != (4, 4):
            raise ValueError("The reorient_transform must be none or (4,4) "
            "array")
        if value is None:
            self._content['dcmmeta_reorient_transform'] = None
        else:
            self._content['dcmmeta_reorient_transform'] = value.tolist()

    @property
    def affine(self):
        '''The affine associated with the meta data. If this differs from the
        image affine, the per-slice meta data will not be used. '''
        return np.array(self._content['dcmmeta_affine'])

    @affine.setter
    def affine(self, value):
        if value.shape != (4, 4):
            raise ValueError("Invalid shape for affine")
        self._content['dcmmeta_affine'] = value.tolist()

    @property
    def slice_dim(self):
        '''The index of the slice dimension associated with the per-slice
        meta data.'''
        return self._content['dcmmeta_slice_dim']

    @slice_dim.setter
    def slice_dim(self, value):
        if not value is None and not (0 <= value < 3):
            raise ValueError("The slice dimension must be between zero and "
                             "two")
        self._content['dcmmeta_slice_dim'] = value

    @property
    def shape(self):
        '''The shape of the data associated with the meta data. Defines the
        number of values for the meta data classifications.'''
        return tuple(self._content['dcmmeta_shape'])

    @shape.setter
    def shape(self, value):
        if not (3 <= len(value) < 6):
            raise ValueError("The shape must have a length between three and "
                             "six")
        self._content['dcmmeta_shape'][:] = value

    @property
    def version(self):
        '''The version of the meta data extension.'''
        return self._content['dcmmeta_version']

    @version.setter
    def version(self, value):
        '''Set the version of the meta data extension.'''
        self._content['dcmmeta_version'] = value

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

    classifications = (('global', 'const'),
                       ('global', 'slices'),
                       ('time', 'samples'),
                       ('time', 'slices'),
                       ('vector', 'samples'),
                       ('vector', 'slices'),
                      )
    '''The classifications used to separate meta data based on if and how the
    values repeat. Each class is a tuple with a base class and a sub class.'''

    def get_valid_classes(self):
        '''Return the meta data classifications that are valid for this
        extension.

        Returns
        -------
        valid_classes : tuple
            The classifications that are valid for this extension (based on its
            shape).

        '''
        shape = self.shape
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

    def get_multiplicity(self, classification):
        '''Get the number of meta data values for all meta data of the provided
        classification.

        Parameters
        ----------
        classification : tuple
            The meta data classification.

        Returns
        -------
        multiplicity : int
            The number of values for any meta data of the provided
            `classification`.
        '''
        if not classification in self.get_valid_classes():
            raise ValueError("Invalid classification: %s" % classification)

        base, sub = classification
        shape = self.shape
        n_vals = 1
        if sub == 'slices':
            n_vals = self.n_slices
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
                if len(shape) == 5:
                    n_vals *= shape[4]
            elif base == 'vector':
                n_vals = shape[4]

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
        #Check for the required base keys in the json data
        if not _req_base_keys_map[self.version] <= set(self._content):
            raise InvalidExtensionError('Missing one or more required keys')

        #Check the orientation/shape/version
        if self.affine.shape != (4, 4):
            raise InvalidExtensionError('Affine has incorrect shape')
        slice_dim = self.slice_dim
        if slice_dim is not None:
            if not (0 <= slice_dim < 3):
                raise InvalidExtensionError('Slice dimension is not valid')
        if not (3 <= len(self.shape) < 6):
            raise InvalidExtensionError('Shape is not valid')

        #Check all required meta dictionaries, make sure values have correct
        #multiplicity
        valid_classes = self.get_valid_classes()
        for classes in valid_classes:
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

        #Check that all keys are uniquely classified
        for classes in valid_classes:
            for other_classes in valid_classes:
                if classes == other_classes:
                    continue
                intersect = (set(self.get_class_dict(classes)) &
                             set(self.get_class_dict(other_classes))
                            )
                if len(intersect) != 0:
                    raise InvalidExtensionError("One or more keys have "
                                                "multiple classifications")

    def get_keys(self):
        '''Get a list of all the meta data keys that are available.'''
        keys = []
        for base_class, sub_class in self.get_valid_classes():
            keys += self._content[base_class][sub_class].keys()
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
        for base_class, sub_class in self.get_valid_classes():
            if key in self._content[base_class][sub_class]:
                    return (base_class, sub_class)

        return None

    def get_class_dict(self, classification):
        '''Get the dictionary for the given classification.

        Parameters
        ----------
        classification : tuple
            The meta data classification.

        Returns
        -------
        meta_dict : dict
            The dictionary for the provided classification.
        '''
        base, sub = classification
        return self._content[base][sub]

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
        return self.get_class_dict(classification)[key]

    def get_values_and_class(self, key):
        '''Get the values and the classification for the provided key.

        Parameters
        ----------
        key : str
            The meta data key.

        Returns
        -------
        vals_and_class : tuple
            None for both the value and classification if the key is not found.

        '''
        classification = self.get_classification(key)
        if classification is None:
            return (None, None)
        return (self.get_class_dict(classification)[key], classification)

    def filter_meta(self, filter_func):
        '''Filter the meta data.

        Parameters
        ----------
        filter_func : callable
            Must take a key and values as parameters and return True if they
            should be filtered out.

        '''
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

    def get_subset(self, dim, idx):
        '''Get a DcmMetaExtension containing a subset of the meta data.

        Parameters
        ----------
        dim : int
            The dimension we are taking the subset along.

        idx : int
            The position on the dimension `dim` for the subset.

        Returns
        -------
        result : DcmMetaExtension
            A new DcmMetaExtension corresponding to the subset.

        '''
        if not 0 <= dim < 5:
            raise ValueError("The argument 'dim' must be in the range [0, 5).")

        shape = self.shape
        valid_classes = self.get_valid_classes()

        #Make an empty extension for the result
        result_shape = list(shape)
        result_shape[dim] = 1
        while result_shape[-1] == 1 and len(result_shape) > 3:
            result_shape = result_shape[:-1]
        result = self.make_empty(result_shape,
                                 self.affine,
                                 self.reorient_transform,
                                 self.slice_dim
                                )

        for src_class in valid_classes:
            #Constants remain constant
            if src_class == ('global', 'const'):
                for key, val in self.get_class_dict(src_class).iteritems():
                    result.get_class_dict(src_class)[key] = deepcopy(val)
                continue

            if dim == self.slice_dim:
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

    def to_json(self):
        '''Return the extension encoded as a JSON string.'''
        self.check_valid()
        return self._mangle(self._content)

    @classmethod
    def from_json(klass, json_str):
        '''Create an extension from the JSON string representation.'''
        result = klass(dcm_meta_ecode, json_str)
        result.check_valid()
        return result

    @classmethod
    def make_empty(klass, shape, affine, reorient_transform=None,
                   slice_dim=None):
        '''Make an empty DcmMetaExtension.

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

        Returns
        -------
        result : DcmMetaExtension
            An empty DcmMetaExtension with the required values set to the
            given arguments.

        '''
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
        result.shape = shape
        result.affine = affine
        result.reorient_transform = reorient_transform
        result.slice_dim = slice_dim
        result.version = _meta_version

        return result

    @classmethod
    def from_runtime_repr(klass, runtime_repr):
        '''Create an extension from the Python runtime representation (nested
        dictionaries).
        '''
        result = klass(dcm_meta_ecode, '{}')
        result._content = runtime_repr
        result.check_valid()
        return result

    @classmethod
    def from_sequence(klass, seq, dim, affine=None, slice_dim=None):
        '''Create an extension from a sequence of extensions.

        Parameters
        ----------
        seq : sequence
            The sequence of DcmMetaExtension objects.

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
        result : DcmMetaExtension
            The result of merging the extensions in `seq` along the dimension
            `dim`.
        '''
        if not 0 <= dim < 5:
            raise ValueError("The argument 'dim' must be in the range [0, 5).")

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

        result = klass.make_empty(output_shape,
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
            if classes[1] == 'slices' and not use_slices:
                continue
            result._content[classes[0]][classes[1]] = \
                deepcopy(first_input.get_class_dict(classes))

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
            #directional meta data
            if ((reorient_transform is None or
                 input_ext.reorient_transform is None) or
                not (np.allclose(input_ext.affine, affine) or
                     np.allclose(input_ext.reorient_transform,
                                 reorient_transform)
                    )
               ):
                reorient_transform = None
            result._insert(dim, input_ext)
            shape[dim] += 1
            result.shape = shape

        #Set the reorient transform
        result.reorient_transform = reorient_transform

        #Try simplifying any keys in global slices
        for key in result.get_class_dict(('global', 'slices')).keys():
            result._simplify(key)

        return result

    def __str__(self):
        return self._mangle(self._content)

    def __eq__(self, other):
        if not np.allclose(self.affine, other.affine):
            return False
        if self.shape != other.shape:
            return False
        if self.slice_dim != other.slice_dim:
            return False
        if self.version != other.version:
            return False
        for classes in self.get_valid_classes():
            if (dict(self.get_class_dict(classes)) !=
               dict(other.get_class_dict(classes))):
                return False

        return True

    def _unmangle(self, value):
        '''Go from extension data to runtime representation.'''
        #Its not possible to preserve order while loading with python 2.6
        kwargs = {}
        if sys.version_info >= (2, 7):
            kwargs['object_pairs_hook'] = OrderedDict
        return json.loads(value, **kwargs)

    def _mangle(self, value):
        '''Go from runtime representation to extension data.'''
        return json.dumps(value, indent=4)

    _const_tests = {('global', 'slices') : (('global', 'const'),
                                            ('vector', 'samples'),
                                            ('time', 'samples')
                                           ),
                    ('vector', 'slices') : (('global', 'const'),
                                            ('time', 'samples')
                                           ),
                    ('time', 'slices') : (('global', 'const'),
                                         ),
                    ('time', 'samples') : (('global', 'const'),
                                           ('vector', 'samples'),
                                          ),
                    ('vector', 'samples') : (('global', 'const'),)
                   }
    '''Classification mapping showing possible reductions in multiplicity for
    values that are constant with some period.'''

    def _get_const_period(self, src_cls, dest_cls):
        '''Get the period over which we test for const-ness with for the
        given classification change.'''
        if dest_cls == ('global', 'const'):
            return None
        elif src_cls == ('global', 'slices'):
            return self.get_multiplicity(src_cls) / self.get_multiplicity(dest_cls)
        elif src_cls == ('vector', 'slices'): #implies dest_cls == ('time', 'samples'):
            return  self.n_slices
        elif src_cls == ('time', 'samples'): #implies dest_cls == ('vector', 'samples')
            return self.shape[3]
        assert False #Should take one of the above branches

    _repeat_tests = {('global', 'slices') : (('time', 'slices'),
                                             ('vector', 'slices')
                                            ),
                     ('vector', 'slices') : (('time', 'slices'),),
                    }
    '''Classification mapping showing possible reductions in multiplicity for
    values that are repeating with some period.'''

    def _simplify(self, key):
        '''Try to simplify (reduce the multiplicity) of a single meta data
        element by changing its classification. Return True if the
        classification is changed, otherwise False.

        Looks for values that are constant or repeating with some pattern.
        Constant elements with a value of None will be deleted.
        '''
        values, curr_class = self.get_values_and_class(key)

        #If the class is global const then just delete it if the value is None
        if curr_class == ('global', 'const'):
            if values is None:
                del self.get_class_dict(curr_class)[key]
                return True
            return False

        #Test if the values are constant with some period
        dests = self._const_tests[curr_class]
        for dest_cls in dests:
            if dest_cls[0] in self._content:
                period = self._get_const_period(curr_class, dest_cls)
                #If the period is one, the two classifications have the
                #same multiplicity so we are dealing with a degenerate
                #case (i.e. single slice data). Just change the
                #classification to the "simpler" one in this case
                if period == 1 or is_constant(values, period):
                    if period is None:
                        self.get_class_dict(dest_cls)[key] = \
                            values[0]
                    else:
                        self.get_class_dict(dest_cls)[key] = \
                            values[::period]
                    break
        else: #Otherwise test if values are repeating with some period
            if curr_class in self._repeat_tests:
                for dest_cls in self._repeat_tests[curr_class]:
                    if dest_cls[0] in self._content:
                        dest_mult = self.get_multiplicity(dest_cls)
                        if is_repeating(values, dest_mult):
                            self.get_class_dict(dest_cls)[key] = \
                                values[:dest_mult]
                            break
                else: #Can't simplify
                    return False
            else:
                return False

        del self.get_class_dict(curr_class)[key]
        return True

    _preserving_changes = {None : (('global', 'const'),
                                   ('vector', 'samples'),
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
    '''Classification mapping showing allowed changes when increasing the
    multiplicity.'''

    def _get_changed_class(self, key, new_class, slice_dim=None):
        '''Get an array of values corresponding to a single meta data
        element with its classification changed by increasing its
        multiplicity. This will preserve all the meta data and allow easier
        merging of values with different classifications.'''
        values, curr_class = self.get_values_and_class(key)
        if curr_class == new_class:
            return values

        if not new_class in self._preserving_changes[curr_class]:
            raise ValueError("Classification change would lose data.")

        if curr_class is None:
            curr_mult = 1
            per_slice = False
        else:
            curr_mult = self.get_multiplicity(curr_class)
            per_slice = curr_class[1] == 'slices'
        if new_class in self.get_valid_classes():
            new_mult = self.get_multiplicity(new_class)
            #Only way we get 0 for mult is if slice dim is undefined
            if new_mult == 0:
                new_mult = self.shape[slice_dim]
        else:
            new_mult = 1
        mult_fact = new_mult / curr_mult
        if curr_mult == 1:
            values = [values]


        if per_slice:
            result = values * mult_fact
        else:
            result = []
            for value in values:
                result.extend([deepcopy(value)] * mult_fact)

        if new_class == ('global', 'const'):
            result = result[0]

        return result


    def _change_class(self, key, new_class):
        '''Change the classification of the meta data element in place. See
        _get_changed_class.'''
        values, curr_class = self.get_values_and_class(key)
        if curr_class == new_class:
            return

        self.get_class_dict(new_class)[key] = self._get_changed_class(key,
                                                                      new_class)

        if not curr_class is None:
            del self.get_class_dict(curr_class)[key]



    def _copy_slice(self, other, src_class, idx):
        '''Get a copy of the meta data from the 'other' instance with
        classification 'src_class', corresponding to the slice with index
        'idx'.'''
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

        src_dict = other.get_class_dict(src_class)
        dest_dict = self.get_class_dict(dest_class)
        dest_mult = self.get_multiplicity(dest_class)
        stride = other.n_slices
        for key, vals in src_dict.iteritems():
            subset_vals = vals[idx::stride]

            if len(subset_vals) < dest_mult:
                full_vals = []
                for val_idx in xrange(dest_mult / len(subset_vals)):
                    full_vals += deepcopy(subset_vals)
                subset_vals = full_vals
            if len(subset_vals) == 1:
                subset_vals = subset_vals[0]
            dest_dict[key] = deepcopy(subset_vals)
            self._simplify(key)

    def _global_slice_subset(self, key, sample_base, idx):
        '''Get a subset of the meta data values with the classificaion
        ('global', 'slices') corresponding to a single sample along the
        time or vector dimension (as specified by 'sample_base' and 'idx').
        '''
        n_slices = self.n_slices
        shape = self.shape
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
                for vec_idx in xrange(shape[4]):
                    start_idx = (vec_idx * slices_per_vec) + (idx * n_slices)
                    end_idx = start_idx + n_slices
                    result.extend(src_dict[key][start_idx:end_idx])
                return result

    def _copy_sample(self, other, src_class, sample_base, idx):
        '''Get a copy of meta data from 'other' instance with classification
        'src_class', corresponding to one sample along the time or vector
        dimension.'''
        assert src_class != ('global', 'const')
        src_dict = other.get_class_dict(src_class)
        if src_class[1] == 'samples':
            #If we are indexing on the same dim as the src_class we need to
            #change the classification
            if src_class[0] == sample_base:
                #Time samples may become vector samples, otherwise const
                best_dest = None
                for dest_cls in (('vector', 'samples'),
                                 ('global', 'const')):
                    if (dest_cls != src_class and
                        dest_cls in self.get_valid_classes()
                       ):
                        best_dest = dest_cls
                        break

                dest_mult = self.get_multiplicity(dest_cls)
                if dest_mult == 1:
                    for key, vals in src_dict.iteritems():
                        self.get_class_dict(dest_cls)[key] = \
                            deepcopy(vals[idx])
                else: #We must be doing time samples -> vector samples
                    stride = other.shape[3]
                    for key, vals in src_dict.iteritems():
                        self.get_class_dict(dest_cls)[key] = \
                            deepcopy(vals[idx::stride])
                    for key in src_dict.keys():
                        self._simplify(key)

            else: #Otherwise classification does not change
                #The multiplicity will change for time samples if splitting
                #vector dimension
                if src_class == ('time', 'samples'):
                    dest_mult = self.get_multiplicity(src_class)
                    start_idx = idx * dest_mult
                    end_idx = start_idx + dest_mult
                    for key, vals in src_dict.iteritems():
                        self.get_class_dict(src_class)[key] = \
                            deepcopy(vals[start_idx:end_idx])
                        self._simplify(key)
                else: #Otherwise multiplicity is unchanged
                    for key, vals in src_dict.iteritems():
                        self.get_class_dict(src_class)[key] = deepcopy(vals)
        else: #The src_class is per slice
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
                    n_slices = self.n_slices
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
        self_slc_norm = self.slice_normal
        other_slc_norm = other.slice_normal

        #If we are not using slice meta data, temporarily remove it from the
        #other dcmmeta object
        use_slices = (not self_slc_norm is None and
                      not other_slc_norm is None and
                      np.allclose(self_slc_norm, other_slc_norm))
        other_slc_meta = {}
        if not use_slices:
            for classes in other.get_valid_classes():
                if classes[1] == 'slices':
                    other_slc_meta[classes] = other.get_class_dict(classes)
                    other._content[classes[0]][classes[1]] = {}
        missing_keys = list(set(self.get_keys()) - set(other.get_keys()))
        for other_classes in other.get_valid_classes():
            other_keys = other.get_class_dict(other_classes).keys()

            #Treat missing keys as if they were in global const and have a value
            #of None
            if other_classes == ('global', 'const'):
                other_keys += missing_keys

            #When possible, reclassify our meta data so it matches the other
            #classification
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
                if dim == self.slice_dim:
                    self._insert_slice(key, other)
                elif dim < 3:
                    self._insert_non_slice(key, other)
                elif dim == 3:
                    self._insert_sample(key, other, 'time')
                elif dim == 4:
                    self._insert_sample(key, other, 'vector')

        #Restore per slice meta if needed
        if not use_slices:
            for classes in other.get_valid_classes():
                if classes[1] == 'slices':
                    other._content[classes[0]][classes[1]] = \
                        other_slc_meta[classes]

    def _insert_slice(self, key, other):
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, classes, self.slice_dim)


        #Handle some common / simple insertions with special cases
        if classes == ('global', 'const'):
            if local_vals != other_vals:
                for dest_base in ('time', 'vector', 'global'):
                    if dest_base in self._content:
                        self._change_class(key, (dest_base, 'slices'))
                        other_vals = other._get_changed_class(key,
                                                              (dest_base,
                                                               'slices'),
                                                               self.slice_dim
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
                other_vals = other._get_changed_class(key,
                                                      ('global', 'slices'),
                                                      self.slice_dim)

            #Need to interleave slices from different volumes
            n_slices = self.n_slices
            other_n_slices = other.n_slices
            shape = self.shape
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
        other_vals = other._get_changed_class(key, classes, self.slice_dim)

        if local_vals != other_vals:
            del self.get_class_dict(classes)[key]

    def _insert_sample(self, key, other, sample_base):
        local_vals, classes = self.get_values_and_class(key)
        other_vals = other._get_changed_class(key, classes, self.slice_dim)

        if classes == ('global', 'const'):
            if local_vals != other_vals:
                self._change_class(key, (sample_base, 'samples'))
                local_vals = self.get_values(key)
                other_vals = other._get_changed_class(key,
                                                      (sample_base, 'samples'),
                                                      self.slice_dim
                                                     )
                local_vals.extend(other_vals)
        elif classes == (sample_base, 'samples'):
            local_vals.extend(other_vals)
        else:
            if classes != ('global', 'slices'):
                self._change_class(key, ('global', 'slices'))
                local_vals = self.get_values(key)
                other_vals = other._get_changed_class(key,
                                                      ('global', 'slices'),
                                                      self.slice_dim)

            shape = self.shape
            n_dims = len(shape)
            if sample_base == 'time' and n_dims == 5:
                #Need to interleave values from the time points in each vector
                #component
                n_slices = self.n_slices
                slices_per_vec = n_slices * shape[3]
                oth_slc_per_vec = n_slices * other.shape[3]

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
    '''Exception denoting that there is no DcmMetaExtension in the Nifti header.
    '''
    def __str__(self):
        return 'No dcmmeta extension found.'

def patch_dcm_ds_is(dcm):
    '''Convert all elements in `dcm` with VR of 'DS' or 'IS' to floats and ints.
    This is a hackish work around for the backwards incompatibility of pydicom
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
                    extension.check_valid()
                except InvalidExtensionError as e:
                    print("Found candidate extension, but invalid: %s" % e)
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
                                                hdr.get_best_affine(),
                                                None,
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

    def meta_valid(self, classification):
        '''Return true if the meta data with the given classification appears
        to be valid for the wrapped Nifti image. Considers the shape and
        orientation of the image and the meta data extension.'''
        if classification == ('global', 'const'):
            return True

        img_shape = self.nii_img.get_shape()
        meta_shape = self.meta_ext.shape
        if classification == ('vector', 'samples'):
            return meta_shape[4:] == img_shape[4:]
        if classification == ('time', 'samples'):
            return meta_shape[3:] == img_shape[3:]

        hdr = self.nii_img.get_header()
        if self.meta_ext.n_slices != hdr.get_n_slices():
            return False

        slice_dim = hdr.get_dim_info()[2]
        slice_dir = self.nii_img.get_affine()[slice_dim, :3]
        slices_aligned = np.allclose(slice_dir,
                                     self.meta_ext.slice_normal,
                                     atol=1e-6)

        if classification == ('time', 'slices'):
            return slices_aligned
        if classification == ('vector', 'slices'):
            return meta_shape[3] == img_shape[3] and slices_aligned
        if classification == ('global', 'slices'):
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
        The per-sample and per-slice meta data will only be considered if the
        `samples_valid` and `slices_valid` methods return True (respectively),
        and an `index` is specified.
        '''
        #Get the value(s) and classification for the key
        values, classes = self.meta_ext.get_values_and_class(key)
        if classes is None:
            return default

        #Check if the value is constant
        if classes == ('global', 'const'):
            return values

        #Check if the classification is valid
        if not self.meta_valid(classes):
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

            #First try per time/vector sample values
            if classes == ('time', 'samples'):
                return values[index[3]]
            if classes == ('vector', 'samples'):
                return values[index[4]]

            #Finally, if aligned, try per-slice values
            slice_dim = self.nii_img.get_header().get_dim_info()[2]
            n_slices = shape[slice_dim]
            if classes == ('global', 'slices'):
                val_idx = index[slice_dim]
                for count, idx_val in enumerate(index[3:]):
                    val_idx += idx_val * n_slices
                    n_slices *= shape[count+3]
                return values[val_idx]
            elif classes == ('time', 'slices'):
                val_idx = index[slice_dim]
                return values[val_idx]
            elif classes == ('vector', 'slices'):
                val_idx = index[slice_dim]
                val_idx += index[3]*n_slices
                return values[val_idx]

        return default

    def remove_extension(self):
        '''Remove the DcmMetaExtension from the header of nii_img. The
        attribute `meta_ext` will still point to the extension.'''
        hdr = self.nii_img.get_header()
        target_idx = None
        for idx, ext in enumerate(hdr.extensions):
            if id(ext) == id(self.meta_ext):
                target_idx = idx
                break
        else:
            raise IndexError('Extension not found in header')
        del hdr.extensions[target_idx]
        # Nifti1Image.update_header will increase this if necessary
        hdr['vox_offset'] = 0

    def replace_extension(self, dcmmeta_ext):
        '''Replace the DcmMetaExtension.

        Parameters
        ----------
        dcmmeta_ext : DcmMetaExtension
            The new DcmMetaExtension.

        '''
        self.remove_extension()
        self.nii_img.get_header().extensions.append(dcmmeta_ext)
        self.meta_ext = dcmmeta_ext

    def split(self, dim=None):
        '''Generate splits of the array and meta data along the specified
        dimension.

        Parameters
        ----------
        dim : int
            The dimension to split the voxel array along. If None it will
            prefer the vector, then time, then slice dimensions.

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
            result.meta_ext.get_class_dict(('global', 'const')).update(meta_dict)

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
                if not np.allclose(in_vec, axis_vec, atol=5e-4):
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
        if hdr_info['qform'] is not None and hdr_info['qform_code'] is not None:
            if not scaled_dim_dir is None:
                hdr_info['qform'][:3, dim] = scaled_dim_dir
            result_nii.set_qform(hdr_info['qform'],
                                 int(hdr_info['qform_code']),
                                 update_affine=True)
        if hdr_info['sform'] is not None and hdr_info['sform_code'] is not None:
            if not scaled_dim_dir is None:
                hdr_info['sform'][:3, dim] = scaled_dim_dir
            result_nii.set_sform(hdr_info['sform'],
                                 int(hdr_info['sform_code']),
                                 update_affine=True)
        if hdr_info['dim_info'] is not None:
            result_hdr.set_dim_info(*hdr_info['dim_info'])
            slice_dim = hdr_info['dim_info'][2]
        else:
            slice_dim = None
        if hdr_info['intent'] is not None:
            result_hdr.set_intent(*hdr_info['intent'])
        if hdr_info['xyzt_units'] is not None:
            result_hdr.set_xyzt_units(*hdr_info['xyzt_units'])
        if hdr_info['slice_duration'] is not None:
            result_hdr.set_slice_duration(hdr_info['slice_duration'])
        if hdr_info['slice_times'] is not None:
            result_hdr.set_slice_times(hdr_info['slice_times'])

        #Create the meta data extension and insert it
        seq_exts = [elem.meta_ext for elem in seq]
        result_ext = DcmMetaExtension.from_sequence(seq_exts,
                                                    dim,
                                                    affine,
                                                    slice_dim)
        result_hdr.extensions.append(result_ext)

        return NiftiWrapper(result_nii)
