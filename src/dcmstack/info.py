""" Information for setup.py that we may also want to access in dcmstack. Can
not import dcmstack.
"""
import sys

_version_major = 0
_version_minor = 9
_version_micro = 0
_version_extra = 'dev'
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = 'Stack DICOM images into volumes and convert to Nifti'

# Hard dependencies
install_requires = ['pydicom >= 0.9.7',
                    'nibabel >= 2.5.1',
                    'requires-python = ">=2.7"'
                   ]

# Extra requirements for building documentation and testing
extras_requires = {'doc':  ["sphinx", "numpydoc"],
                   'test': [
                        'pytest <= 4.6 ; python_version == "2.7"',
                        'pytest ; python_version > "2.7"'
                    ],
                  }


NAME                = 'dcmstack'
AUTHOR              = "Brendan Moloney"
AUTHOR_EMAIL        = "moloney@ohsu.edu"
MAINTAINER          = "Brendan Moloney"
MAINTAINER_EMAIL    = "moloney@ohsu.edu"
URL                 = "https://github.com/moloney/dcmstack"
DESCRIPTION         = description
LICENSE             = "MIT license"
CLASSIFIERS         = CLASSIFIERS
PLATFORMS           = "OS Independent"
ISRELEASE           = _version_extra == ''
VERSION             = __version__
INSTALL_REQUIRES    = install_requires
EXTRAS_REQUIRES      = extras_requires
PROVIDES            = ["dcmstack"]
