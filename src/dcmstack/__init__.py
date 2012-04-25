"""
Package for stacking DICOM images into multi dimensional volumes, extracting 
the DICOM meta data, converting the result to Nifti files with the meta 
data stored in a header extension, and work with these extended Nifti files.
"""

from .dcmstack import *
