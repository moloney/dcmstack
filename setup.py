from setuptools import setup, find_packages

setup(name='dcmstack',
      description='Stack DICOM images into volumes',
      version='0.5',
      author='Brendan Moloney',
      author_email='moloney@ohsu.edu',
      packages=find_packages('src'),
      package_dir = {'':'src'},
      install_requires=['pydicom', 'nibabel'],
      entry_points = {'console_scripts' : \
                          ['dcmstack = dcmstack.dcmstack_cli:main',
                           'nitool = dcmstack.nitool_cli:main',
                          ],
                     }
     )
