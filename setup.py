from setuptools import setup, find_packages
import sys

#Hard dependencies
install_requires = ['pydicom >= 0.9.7', 
                    'nibabel',
                   ]

#Add version specific dependencies
if sys.version_info < (2, 6):
    raise Exception("must use python 2.6 or greater")
elif sys.version_info < (2, 7):
    install_requires.append('ordereddict')


setup(name='dcmstack',
      description='Stack DICOM images into volumes',
      version='0.7.dev',
      author='Brendan Moloney',
      author_email='moloney@ohsu.edu',
      packages=find_packages('src'),
      package_dir = {'':'src'},
      install_requires=install_requires,
      extras_require = {
                        'doc':  ["sphinx", "numpydoc"],
                        'test': ["nose"],
                       },
      entry_points = {'console_scripts' : \
                          ['dcmstack = dcmstack.dcmstack_cli:main',
                           'nitool = dcmstack.nitool_cli:main',
                          ],
                     },
      test_suite = 'nose.collector'
     )
