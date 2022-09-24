from setuptools import setup, find_packages
import sys, os

# Most of the relevant info is stored in this file
info_file = os.path.join('src', 'dcmstack', 'info.py')
exec(open(info_file).read())


setup(name=NAME,
      python_requires=">=2.7",
      description=DESCRIPTION,
      long_description=open("README.rst").read(),
      long_description_content_type="text/x-rst",
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
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
      test_suite = 'pytest'
     )
