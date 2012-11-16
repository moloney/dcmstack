Introduction
============

The *dcmstack* software allows series of DICOM images to be stacked into 
multi-dimensional arrays. These arrays can be written out as Nifti files with 
an optional header extension (the *DcmMeta* extension) containing a summary 
of all the meta data from the source DICOM files.

Dependencies
------------

Either Python 2.6 or 2.7 is required.  With Python 2.6 it is not possible 
to maintain the order of meta data keys when reading back the JSON.

DcmStack requires the packages pydicom_ (>=0.9.7) and NiBabel_.

.. _pydicom: http://code.google.com/p/pydicom/
.. _nibabel: http://nipy.sourceforge.net/nibabel/

Installation
------------

Download the latest release from github_, and run easy_install on the 
downloaded .zip file.

.. _github: https://github.com/moloney/dcmstack/tags


Basic Conversion 
----------------

The software consists of the python package (*dcmstack*) with two command 
line interfaces (*dcmstack* and *nitool*).

It is recommended that you sort your DICOM data into directories (at least 
per study, but perferably by series) before conversion.

To convert directories of DICOM data from the command line you generally 
just need to pass the directories to *dcmstack*:

.. code-block:: console
    
    $ dcmstack -v 032-MPRAGEAXTI900Pre/
    Processing source directory 032-MPRAGEAXTI900Pre/
    Found 64 source files in the directory
    Created 1 stacks of DICOM images
    Writing out stack to path 032-MPRAGEAXTI900Pre/032-MPRAGE_AX_TI900_Pre.nii.gz
    
Here we use the verbose flab (*-v*) to show what is going on behind the 
scenes. To embed the DcmMeta header extension we need to use the *--embed* 
option. For more information see :doc:`CLI_Tutorial`.

Performing the conversion from Python code requires a few extra steps
but is also much more flexible:

.. code-block:: python
    
    >>> import dcmstack
    >>> from glob import glob
    >>> src_dcms = glob('032-MPRAGEAXTI900Pre/*.dcm')
    >>> stacks = dcmstack.parse_and_stack(src_dcms)
    >>> stack = stacks.values[0]
    >>> nii = stack.to_nifti()
    >>> nii.to_filename('output.nii.gz')

The *parse_and_stack* function has many optional arguments that closely 
match the command line options for *dcmstack*. To embed the DcmMeta 
extension pass *embed_meta=True* to the *to_nifti* method. For more 
information see :doc:`Python_Tutorial`.

Basic Meta Data Usage
---------------------

To work with Nifti files containing the embedded DcmMeta extension on the 
command line, use the *nitool* command. The *nitool* command has several sub 
commands.

.. code-block:: console

    $ nitool lookup InversionTime 032-MPRAGE_AX_TI900_Pre.nii.gz 
    900.0

Here we use the *lookup* sub command to lookup up the value for 
'InversionTime'. For more information about using *nitool* see 
:doc:`CLI_Tutorial`.

To work with the extended Nifti files from Python, use the *NiftiWrapper* class.

.. code-block:: python

    >>> from dcmstack import dcmmeta
    >>> nii_wrp = dcmmeta.NiftiWrapper.from_filename('032-MPRAGE_AX_TI900_Pre.nii.gz')
    >>> nii_wrp.get_meta('InversionTime')
    900.0
    
For more information on using the *NiftiWrapper* class see 
:doc:`Python_Tutorial`.

For information on the DcmMeta extension see :doc:`DcmMeta_Extension`.


