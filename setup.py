from setuptools import setup, find_packages
import sys, os

# Most of the relevant info is stored in this file
info_file = os.path.join('src', 'dcmstack', 'info.py')
exec(open(info_file).read())


setup(name=NAME,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      classifiers=CLASSIFIERS,
      platforms=PLATFORMS,
      version=VERSION,
      provides=PROVIDES,
      packages=find_packages('src'),
      package_dir = {'':'src'},
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRES,
      entry_points = {'console_scripts' : \
                          ['dcmstack = dcmstack.dcmstack_cli:main',
                           'nitool = dcmstack.nitool_cli:main',
                          ],
                     },
      test_suite = 'nose.collector'
     )
