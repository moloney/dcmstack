Python Tutorial
===============

This is a brief overview of how to use the dcmstack Python package. For 
details refer to :doc:`modules`.

Creating DicomStack Objects
---------------------------

If you have an aquisition that you would like to turn into a single 
*DicomStack* object then you may want to do this directly.

.. code-block:: python
    
    >>> import dcmstack, dicom
    >>> from glob import glob
    >>> src_paths = glob('032-MPRAGEAXTI900Pre/*.dcm')
    >>> my_stack = dcmstack.DicomStack()
    >>> for src_path in src_paths:
    ...     src_dcm = dicom.read_file(src_path)
    ...     my_stack.add_dcm(src_dcm)

If you are unsure how many stacks you want from a collection of DICOM data 
sets then you should use the *parse_and_stack* function. This will group 
together source DICOM data sets based on the *key_format* and *opt_key_suffix* 
parameters.

.. code-block:: python
    
    >>> import dcmstack
    >>> from glob import glob
    >>> src_paths = glob('dicom_data/*.dcm')
    >>> stacks = dcmstack.parse_and_stack(src_paths)

Specifying Time and Vector Order
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, if there is more than one 3D volume in the stack they will be 
ordered along the fourth (time) dimension using 'AcquisitionTime'. To order 
them differently or to order them along the fifth (vector) dimension, specify 
the *time_order* and/or *vector_order* to the *DicomStack* constructor. Any 
keyword arguments for the *DicomStack* constructor can also be passed to 
*parse_and_stack*.

Grouping Datasets
^^^^^^^^^^^^^^^^^

The *parse_and_stack* function takes two format strings as arguments: 
*key_format* and *opt_key_suffix*. These format strings determine the keys 
in the returned dictionary. The meta data from each source DICOM is used to 
format these strings and the result determines which stack it will be added 
to. The result of formatting *opt_key_suffix* is appended to the result of 
formatting *key_format* if (and only if) it varies across inputs.

Using DicomStack Objects
------------------------

Once you have created your *DicomStack* objects you will typically want to get 
the array of voxel data, get the affine transform, or create a Nifti1Image.

.. code-block:: python
    
    >>> stack_data = my_stack.get_data()
    >>> stack_affine = my_stack.get_affine()
    >>> nii = my_stack.to_nifti()
    
Embedding Meta Data
^^^^^^^^^^^^^^^^^^^

The meta data from the source DICOM data sets can be summarized into a 
*DcmMetaExtension* which is embeded into the Nifti header. To do this you can 
either pass True for the *embed_meta* parameter to *DicomStack.to_nifti* or 
you can immediately get a *NiftiWrapper* with *DicomStack.to_nifti_wrapper*.

By default the meta data is filtered to minimize the chance of including any 
PHI.  This filtering can be controlled with the *meta_filter* parameter to 
the *DicomStack* constructor.

Creating NiftiWrapper Objects
-----------------------------

The *NiftiWrapper* class can be used to work with extended Nifti files. As 
mentioned above, these can be created directly from a *DicomStack*.

.. code-block:: python
    
    >>> import dcmstack, dicom
    >>> from glob import glob
    >>> src_paths = glob('032-MPRAGEAXTI900Pre/*.dcm')
    >>> my_stack = dcmstack.DicomStack()
    >>> for src_path in src_paths:
    ...     src_dcm = dicom.read_file(src_path)
    ...     my_stack.add_dcm(src_dcm)
    ...
    >>> nii_wrp = my_stack.to_nifti_wrapper()
    >>> nii_wrp.get_meta('InversionTime')
    900.0

They can also be created by passing a *Nifti1Image* to the *NiftiWrapper* 
constructor or by passing the path to a Nifti file to 
*NiftiWrapper.from_filename*. 

Using NiftiWrapper Objects
--------------------------

The *NiftiWrapper* objects have two attributes: *nii_img* (the *Nifti1Image* 
being wrapped) and *meta_ext* (the *DcmMetaExtension*).

Meta data that is constant can be accessed with dict-style lookups. The more 
general access method is *get_meta* which can optionally take an index into 
the voxel array in order to provide access to varying meta data.

.. code-block:: python
    
    >>> import dcmstack, dicom
    >>> from glob import glob
    >>> src_paths = glob('032-MPRAGEAXTI900Pre/*.dcm')
    >>> my_stack = dcmstack.DicomStack()
    >>> for src_path in src_paths:
    ...     src_dcm = dicom.read_file(src_path)
    ...     my_stack.add_dcm(src_dcm)
    ...
    >>> nii_wrp = my_stack.to_nifti_wrapper()
    >>> nii_wrp['InversionTime']
    900.0
    >>> nii_wrp.get_meta('InversionTime')
    900.0
    >>> nii_wrp['InstanceNumber']
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "build/bdist.linux-x86_64/egg/dcmstack/dcmmeta.py", line 1026, in __getitem__
    KeyError: 'InstanceNumber'
    >>> nii_wrp.get_meta('InstanceNumber')
    >>> nii_wrp.get_meta('InstanceNumber', index=(0,0,0))
    1
    >>> nii_wrp.get_meta('InstanceNumber', index=(0,0,1))
    2

    
