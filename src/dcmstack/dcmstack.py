"""
Stack DICOM datasets into volumes. The contents of this module are imported
into the package namespace.
"""
import warnings, re, dicom
from copy import deepcopy
import nibabel as nb
from nibabel.nifti1 import Nifti1Extensions
from nibabel.spatialimages import HeaderDataError
from nibabel.orientations import (io_orientation,
                                  apply_orientation,
                                  inv_ornt_aff)
import numpy as np
from .dcmmeta import DcmMetaExtension, NiftiWrapper

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from nibabel.nicom.dicomwrappers import wrapper_from_data

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
                        'Station',
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
                        'UID',
                        'StudyDescription',
                        'DeviceSerialNumber',
                        'ReferencedImageSequence',
                        'RequestedProcedureDescription',
                        'PerformedProcedureStepDescription',
                        'PerformedProcedureStepID',
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

def ornt_transform(start_ornt, end_ornt):
    '''Return the orientation that transforms from `start_ornt` to `end_ornt`.

    Parameters
    ----------
    start_ornt : (n,2) orientation array
        Initial orientation.

    end_ornt : (n,2) orientation array
        Final orientation.

    Returns
    -------
    orientations : (p, 2) ndarray
       The orientation that will transform the `start_ornt` to the `end_ornt`.
    '''
    start_ornt = np.asarray(start_ornt)
    end_ornt = np.asarray(end_ornt)
    if start_ornt.shape != end_ornt.shape:
        raise ValueError("The orientations must have the same shape")
    if start_ornt.shape[1] != 2:
        raise ValueError("Invalid shape for an orientation: %s" %
                         start_ornt.shape)
    result = np.empty_like(start_ornt)
    for end_in_idx, (end_out_idx, end_flip) in enumerate(end_ornt):
        for start_in_idx, (start_out_idx, start_flip) in enumerate(start_ornt):
            if end_out_idx == start_out_idx:
                if start_flip == end_flip:
                    flip = 1
                else:
                    flip = -1
                result[start_in_idx, :] = [end_in_idx, flip]
                break
        else:
            raise ValueError("Unable to find out axis %d in start_ornt" %
                             end_out_idx)
    return result

def axcodes2ornt(axcodes, labels=None):
    """ Convert axis codes `axcodes` to an orientation

    Parameters
    ----------
    axcodes : (N,) tuple
        axis codes - see ornt2axcodes docstring
    labels : optional, None or sequence of (2,) sequences
        (2,) sequences are labels for (beginning, end) of output axis.  That
        is, if the first element in `axcodes` is ``front``, and the second
        (2,) sequence in `labels` is ('back', 'front') then the first
        row of `ornt` will be ``[1, 1]``. If None, equivalent to
        ``(('L','R'),('P','A'),('I','S'))`` - that is - RAS axes.

    Returns
    -------
    ornt : (N,2) array-like
        oritation array - see io_orientation docstring

    Examples
    --------
    >>> axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
    [[1, 1],[0,-1],[2,1]]
    """

    if labels is None:
        labels = zip('LPI', 'RAS')

    n_axes = len(axcodes)
    ornt = np.ones((n_axes, 2), dtype=np.int8) * np.nan
    for code_idx, code in enumerate(axcodes):
        for label_idx, codes in enumerate(labels):
            if code is None:
                continue
            if code in codes:
                if code == codes[0]:
                    ornt[code_idx, :] = [label_idx, -1]
                else:
                    ornt[code_idx, :] = [label_idx, 1]
                break
    return ornt

def reorder_voxels(vox_array, affine, voxel_order):
    '''Reorder the given voxel array and corresponding affine.

    Parameters
    ----------
    vox_array : array
        The array of voxel data

    affine : array
        The affine for mapping voxel indices to Nifti patient space

    voxel_order : str
        A three character code specifing the desired ending point for rows,
        columns, and slices in terms of the orthogonal axes of patient space:
        (l)eft, (r)ight, (a)nterior, (p)osterior, (s)uperior, and (i)nferior.

    Returns
    -------
    out_vox : array
        An updated view of vox_array.

    out_aff : array
        A new array with the updated affine

    reorient_transform : array
        The transform used to update the affine.

    ornt_trans : tuple
        The orientation transform used to update the orientation.

    '''
    #Check if voxel_order is valid
    voxel_order = voxel_order.upper()
    if len(voxel_order) != 3:
        raise ValueError('The voxel_order must contain three characters')
    dcm_axes = ['LR', 'AP', 'SI']
    for char in voxel_order:
        if not char in 'LRAPSI':
            raise ValueError('The characters in voxel_order must be one '
                             'of: L,R,A,P,I,S')
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
    orig_ornt = io_orientation(affine)
    new_ornt = axcodes2ornt(voxel_order)
    ornt_trans = ornt_transform(orig_ornt, new_ornt)
    orig_shape = vox_array.shape
    vox_array = apply_orientation(vox_array, ornt_trans)
    aff_trans = inv_ornt_aff(ornt_trans, orig_shape)
    affine = np.dot(affine, aff_trans)

    return (vox_array, affine, aff_trans, ornt_trans)

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
    #Allow ACR/NEMA style format by removing any colon chars
    time_str = time_str.replace(':', '')

    #Only the hours portion is required
    result = int(time_str[:2]) * 3600

    str_len = len(time_str)
    if str_len > 2:
        result += int(time_str[2:4]) * 60
    if str_len > 4:
        result += float(time_str[4:])

    return float(result)

class IncongruentImageError(Exception):
    def __init__(self, msg):
        '''An exception denoting that a DICOM with incorrect size or orientation
        was passed to `DicomStack.add_dcm`.'''
        self.msg = msg

    def __str__(self):
        return 'The image is not congruent to the existing stack: %s' % self.msg

class ImageCollisionError(Exception):
    '''An exception denoting that a DICOM which collides with one already in
    the stack was passed to a `DicomStack.add_dcm`.'''
    def __str__(self):
        return 'The image collides with one already in the stack'

class InvalidStackError(Exception):
    def __init__(self, msg):
        '''An exception denoting that a `DicomStack` is not currently valid'''
        self.msg = msg

    def __str__(self):
        return 'The DICOM stack is not valid: %s' % self.msg

class DicomOrdering(object):
    '''Object defining an ordering for a set of dicom datasets. Create a
    DicomOrdering with the given DICOM element keyword.

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

    def __init__(self, key, abs_ordering=None, abs_as_str=False):
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

def _make_dummy(reference, meta, iop):
    '''Make a "dummy" NiftiWrapper (no valid pixel data).'''
    #Create the dummy data array filled with largest representable value
    data = np.empty_like(reference.nii_img.get_data())
    data[...] = np.iinfo(np.int16).max

    #Create the nifti image and set header data
    aff = reference.nii_img.get_affine().copy()
    aff[:3, 3] = [iop[1], iop[0], iop[2]]
    nii_img = nb.nifti1.Nifti1Image(data, aff)
    hdr = nii_img.get_header()
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
    result.meta_ext.reorient_transform = np.diag([-1., -1., 1., 1.])
    result.meta_ext.get_class_dict(('global', 'const')).update(meta)

    return result

default_group_keys =  ('SeriesInstanceUID',
                       'SeriesNumber',
                       'ProtocolName',
                       'ImageOrientationPatient')
'''Default keys for grouping DICOM files that belong in the same
multi-dimensional array together.'''

class DicomStack(object):
    '''Defines a method for stacking together DICOM data sets into a multi
    dimensional volume.

    Tailored towards creating NiftiImage output, but can also just create numpy
    arrays. Can summarize all of the meta data from the input DICOM data sets
    into a Nifti header extension (see `dcmmeta.DcmMetaExtension`).

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

    Notes
    -----
    If both time_order and vector_order are None, the time_order will be
    guessed based off the data sets.
    '''

    sort_guesses = ['EchoTime',
                    'InversionTime',
                    'RepetitionTime',
                    'FlipAngle',
                    'TriggerTime',
                    'AcquisitionTime',
                    'ContentTime',
                    'AcquisitionNumber',
                    'InstanceNumber',
                   ]
    '''The meta data keywords used when trying to guess the sorting order.
    Keys that come earlier in the list are given higher priority.'''

    minimal_keys = set(sort_guesses +
                       ['Rows',
                        'Columns',
                        'PixelSpacing',
                        'ImageOrientationPatient',
                        'InPlanePhaseEncodingDirection',
                        'RepetitionTime',
                        'AcquisitionTime'
                       ] +
                       list(default_group_keys)
                      )
    '''Set of minimal meta data keys that should be provided if they exist in
    the source DICOM files.'''

    def __init__(self, time_order=None, vector_order=None,
                 allow_dummies=False, meta_filter=None):
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

    def _chk_close(self, keys, meta1, meta2):
        for key in keys:
            if not np.allclose(meta1[key], meta2[key], atol=5e-5):
                raise IncongruentImageError("%s is not close to matching" %
                                            key)

    def _chk_congruent(self, meta):
        is_dummy = not 'Rows' in meta or not 'Columns' in meta
        if is_dummy and not self._allow_dummies:
            raise IncongruentImageError('Missing Rows/Columns')

        if not self._ref_input is None:
            self._chk_close(('PixelSpacing',
                             'ImageOrientationPatient'),
                             meta,
                             self._ref_input
                            )
            if not is_dummy:
                self._chk_equal(('Rows', 'Columns'), meta, self._ref_input)
        elif len(self._dummies) != 0:
            self._chk_close(('PixelSpacing',
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
            from .extract import default_extractor
            meta = default_extractor(dcm)

        dw = wrapper_from_data(dcm)

        is_dummy = self._chk_congruent(meta)

        self._phase_enc_dirs.add(meta.get('InPlanePhaseEncodingDirection'))
        self._repetition_times.add(meta.get('RepetitionTime'))

        #Pull the info used for sorting
        slice_pos = dw.slice_indicator
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

        #If a explicit order was specified, raise an exception if image
        #collides with another already in the stack
        if ((not self._time_order is None or
             not self._vector_order is None) and
            sorting_tuple in self._sorting_tuples
           ):
            raise ImageCollisionError()
        self._sorting_tuples.add(sorting_tuple)

        #Create a NiftiWrapper for this input if possible
        nii_wrp = None
        if not is_dummy:
            nii_wrp = NiftiWrapper.from_dicom_wrapper(dw, meta)
            if self._ref_input is None:
                #We don't have a reference input yet, use this one
                self._ref_input = nii_wrp
                #Convert any dummies that we have stashed previously
                for dummy_meta, dummy_tuple, iop in self._dummies:
                    dummy_wrp = _make_dummy(self._ref_input, dummy_meta, iop)
                    self._files_info.append((dummy_wrp, dummy_tuple))
        else:
            if self._ref_input is None:
                #We don't have a reference input, so stash the dummy for now
                self._dummies.append((meta, sorting_tuple, dcm.ImagePositionPatient))
            else:
                #Convert dummy using the reference input
                nii_wrp = _make_dummy(self._ref_input, meta, dcm.ImagePositionPatient)

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

    def _chk_order(self, slice_positions, files_per_vol, num_volumes,
                   num_time_points, num_vec_comps):
        #Sort the files
        self._files_info.sort(key=lambda x: x[1])
        if files_per_vol > 1:
            for vol_idx in range(num_volumes):
                start_slice = vol_idx * files_per_vol
                end_slice = start_slice + files_per_vol
                self._files_info[start_slice:end_slice] = \
                    sorted(self._files_info[start_slice:end_slice],
                           key=lambda x: x[1][-1])

        #Do a thorough check for correctness
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
        slice_positions = sorted(list(self._slice_pos_vals))

        #If more than one file per volume, check that slice spacing is equal
        if files_per_vol > 1:
            spacings = []
            for idx in xrange(files_per_vol - 1):
                spacings.append(slice_positions[idx+1] - slice_positions[idx])
            spacings = np.array(spacings)
            avg_spacing = np.mean(spacings)
            if not np.allclose(avg_spacing, spacings, rtol=4e-2):
                raise InvalidStackError("Slice spacings are not consistent")

        #Simple check for an incomplete stack
        if len(self._files_info) % files_per_vol != 0:
            raise InvalidStackError("Number of files is not an even multiple "
                                    "of the number of unique slice positions.")
        num_volumes = len(self._files_info) / files_per_vol

        #Figure out the number of vector components and time points
        num_vec_comps = len(self._vector_vals)
        if num_vec_comps > num_volumes:
            raise InvalidStackError("Vector variable varies within volumes")
        if num_volumes % num_vec_comps != 0:
            raise InvalidStackError("Number of volumes not an even multiple "
                                    "of the number of vector components.")
        num_time_points = num_volumes / num_vec_comps

        #If both sort keys are None try to guess
        if (num_volumes > 1 and self._time_order is None and
                self._vector_order is None):
            #Get a list of possible sort orders
            possible_orders = []
            for key in self.sort_guesses:
                vals = set([file_info[0].get_meta(key)
                            for file_info in self._files_info]
                          )
                if len(vals) == num_volumes or len(vals) == len(self._files_info):
                    possible_orders.append(key)
            if len(possible_orders) == 0:
                raise InvalidStackError("Unable to guess key for sorting the "
                                        "fourth dimension")

            #Try out each possible sort order
            for time_order in possible_orders:
                #Update sorting tuples
                for idx in xrange(len(self._files_info)):
                    nii_wrp, curr_tuple = self._files_info[idx]
                    self._files_info[idx] = (nii_wrp,
                                             (curr_tuple[0],
                                              nii_wrp[time_order],
                                              curr_tuple[2]
                                             )
                                            )

                #Check the order
                try:
                    self._chk_order(slice_positions,
                                    files_per_vol,
                                    num_volumes,
                                    num_time_points,
                                    num_vec_comps)
                except InvalidStackError:
                    pass
                else:
                    break
            else:
                raise InvalidStackError("Unable to guess key for sorting the "
                                        "fourth dimension")
        else: #If at least on sort key was specified, just check the order
            self._chk_order(slice_positions,
                            files_per_vol,
                            num_volumes,
                            num_time_points,
                            num_vec_comps)

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
        stack_dtype = self._files_info[0][0].nii_img.get_data_dtype()
        #This is a hack to keep fslview happy, Shouldn't cause issues as the
        #original data should be 12-bit and any scaling will result in float
        #data
        if stack_dtype == np.uint16:
            stack_dtype = np.int16
        vox_array = np.empty(stack_shape, dtype=stack_dtype)

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
        aff = self._files_info[0][0].nii_img.get_affine()

        #If there is more than one file per volume, we need to fix slice scaling
        if files_per_vol > 1:
            first_offset = aff[:3, 3]
            second_offset = self._files_info[1][0].nii_img.get_affine()[:3, 3]
            scaled_slc_dir = second_offset - first_offset
            aff[:3, 2] = scaled_slc_dir

        return aff

    def to_nifti(self, voxel_order='LAS', embed_meta=False):
        '''Returns a NiftiImage with the data and affine from the stack.

        Parameters
        ----------
        voxel_order : str
            A three character string repsenting the voxel order in patient
            space (see the function `reorder_voxels`). Can be None or an empty
            string to disable reorientation.

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
        permutation = [0, 1, 2]
        slice_dim = 2
        reorient_transform = np.eye(4)
        if voxel_order:
            (data,
             affine,
             reorient_transform,
             ornt_trans) = reorder_voxels(data, affine, voxel_order)
            permutation, flips = zip(*ornt_trans)
            permutation = [int(val) for val in permutation]

            #Reverse file order in each volume's files if we flipped slice order
            #This will keep the slice times and meta data order correct
            if files_per_vol > 1 and flips[slice_dim] == -1:
                self._shape_dirty = True
                for vol_idx in xrange(n_vols):
                    start = vol_idx * files_per_vol
                    stop = start + files_per_vol
                    self._files_info[start:stop] = [self._files_info[idx]
                                                    for idx in xrange(stop - 1,
                                                                      start - 1,
                                                                      -1)
                                                   ]

            #Update the slice dim
            slice_dim = permutation[2]

        #Create the nifti image using the data array
        nifti_image = nb.Nifti1Image(data, affine)
        nifti_header = nifti_image.get_header()

        #Set the units and dimension info
        nifti_header.set_xyzt_units('mm', 'msec')
        if len(self._repetition_times) == 1 and not None in self._repetition_times:
            nifti_header['pixdim'][4] = list(self._repetition_times)[0]
        dim_info = {'freq' : None, 'phase' : None, 'slice' : slice_dim}
        if len(self._phase_enc_dirs) == 1 and not None in self._phase_enc_dirs:
            phase_dir = list(self._phase_enc_dirs)[0]
            if phase_dir == 'ROW':
                dim_info['phase'] = permutation[1]
                dim_info['freq'] = permutation[0]
            else:
                dim_info['phase'] = permutation[0]
                dim_info['freq'] = permutation[1]
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

            meta_ext.shape = data.shape
            meta_ext.slice_dim = slice_dim
            meta_ext.affine = nifti_header.get_best_affine()
            meta_ext.reorient_transform = reorient_transform

            #Filter and embed the meta data
            meta_ext.filter_meta(self._meta_filter)
            nifti_header.extensions = Nifti1Extensions([meta_ext])

        nifti_image.update_header()
        return nifti_image

    def to_nifti_wrapper(self, voxel_order=''):
        '''Convienance method. Calls `to_nifti` and returns a `NiftiWrapper`
        generated from the result.
        '''
        return NiftiWrapper(self.to_nifti(voxel_order, True))

def parse_and_group(src_paths, group_by=default_group_keys, extractor=None,
                    force=False, warn_on_except=False,
                    close_tests=('ImageOrientationPatient',)):
    '''Parse the given dicom files and group them together. Each group is
    stored as a (list) value in a dict where the key is a tuple of values
    corresponding to the keys in 'group_by'

    Parameters
    ----------
    src_paths : sequence
        A list of paths to the source DICOM files.

    group_by : tuple
        Meta data keys to group data sets with. Any data set with the same
        values for these keys will be grouped together. This tuple of values
        will also be the key in the result dictionary.

    extractor : callable
        Should take a dicom.dataset.Dataset and return a dictionary of the
        extracted meta data.

    force : bool
        Force reading source files even if they do not appear to be DICOM.

    warn_on_except : bool
        Convert exceptions into warnings, possibly allowing some results to be
        returned.

    close_tests : sequence
        Any `group_by` key listed here is tested with `numpy.allclose` instead
        of straight equality when determining group membership.

    Returns
    -------
    groups : dict
        A dict mapping tuples of values (corresponding to 'group_by') to groups
        of data sets. Each element in the list is a tuple containing the dicom
        object, the parsed meta data, and the filename.
    '''
    if extractor is None:
        from .extract import default_extractor
        extractor = default_extractor

    results = {}
    close_elems = {}
    for dcm_path in src_paths:
        #Read the DICOM file
        try:
            dcm = dicom.read_file(dcm_path, force=force)
        except Exception as e:
            if warn_on_except:
                warnings.warn('Error reading file %s: %s' % (dcm_path, str(e)))
                continue
            else:
                raise

        #Extract the meta data and group
        meta = extractor(dcm)
        key_list = [] # Values from group_by elems with equality testing
        close_list = [] # Values from group_by elems with np.allclose testing
        for grp_key in group_by:
            key_elem = meta.get(grp_key)
            if isinstance(key_elem, list):
                key_elem = tuple(key_elem)
            if grp_key in close_tests:
                close_list.append(key_elem)
            else:
                key_list.append(key_elem)

        # Initially each key has multiple sub_results (corresponding to
        # different values of the "close" keys)
        key = tuple(key_list)
        if not key in results:
            results[key] = [(close_list, [(dcm, meta, dcm_path)])]
        else:
            # Look for a matching sub_result
            for c_list, sub_res in results[key]:
                for c_idx, c_val in enumerate(c_list):
                    if not np.allclose(c_val, close_list[c_idx], atol=5e-5):
                        break
                else:
                    sub_res.append((dcm, meta, dcm_path))
                    break
            else:
                # No match found, append another sub result
                results[key].append((close_list, [(dcm, meta, dcm_path)]))

    # Unpack sub results, using the canonical value for the close keys
    full_results = {}
    for eq_key, sub_res_list in results.iteritems():
        for close_key, sub_res in sub_res_list:
            full_key = []
            eq_idx = 0
            close_idx = 0
            for grp_key in group_by:
                if grp_key in close_tests:
                    full_key.append(close_key[close_idx])
                    close_idx += 1
                else:
                    full_key.append(eq_key[eq_idx])
                    eq_idx += 1
            full_key = tuple(full_key)
            full_results[full_key] = sub_res

    return full_results

def stack_group(group, warn_on_except=False, **stack_args):
    result = DicomStack(**stack_args)
    for dcm, meta, fn in group:
        try:
            result.add_dcm(dcm, meta)
        except Exception as e:
            if warn_on_except:
                warnings.warn('Error adding file %s to stack: %s' %
                              (fn, str(e)))
            else:
                raise
    return result

def parse_and_stack(src_paths, group_by=default_group_keys, extractor=None,
                    force=False, warn_on_except=False, **stack_args):
    '''Parse the given dicom files into a dictionary containing one or more
    DicomStack objects.

    Parameters
    ----------
    src_paths : sequence
        A list of paths to the source DICOM files.

    group_by : tuple
        Meta data keys to group data sets with. Any data set with the same
        values for these keys will be grouped together. This tuple of values
        will also be the key in the result dictionary.

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
    results = parse_and_group(src_paths,
                              group_by,
                              extractor,
                              force,
                              warn_on_except)

    for key, group in results.iteritems():
        results[key] = stack_group(group, warn_on_except, **stack_args)

    return results
