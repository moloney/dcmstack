.. -*- rest -*-
.. vim:syntax=rest

========
dcmstack
========

This package provides DICOM to Nifti conversion with the added ability 
to extract and summarize meta data from the source DICOMs. The meta data
can be injected it into a Nifti header extension or written out as a JSON 
formatted text file.

Documentation
-------------

Documentation can be read online: https://dcmstack.readthedocs.org/

You can build the HTML documentation under build/sphinx/html with:

$ python setup.py build_sphinx

If you have the *sphinx* and *numpydoc* packages and a *make* command you 
can build the documentation by running the *make* command in the *doc/* 
directory. For example, to create the HTML documentation you would do:

$ make html

And then view doc/_build/html/index.html with a web browser.

Running Tests
-------------

You can run the tests with:

$ python setup.py test

Or if you already have the *nose* package installed you can use the 
*nosetests* command in the top level directory:

$ nosetests

Installing
----------

You can install the *.zip* or *.tar.gz* package with the *easy_install* 
command.

$ easy_install dcmstack-0.6.zip

Or you can uncompress the package and in the top level directory run:

$ python setup.py install

