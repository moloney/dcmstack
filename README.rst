.. -*- rest -*-
.. vim:syntax=rest

========
dcmstack
========

This package provides DICOM to Nifti conversion with the added ability 
to extract and summarize meta data from the source DICOMs. The meta data
can be injected it into a Nifti header extension or written out as a JSON 
formatted text file.


Installing
----------

You can the latest release from PyPI by doing

.. code-block:: console

  $ pip install dcmstack


Documentation
-------------

Documentation can be read online: https://dcmstack.readthedocs.org/

You can build the HTML documentation under build/sphinx/html with:

If you have the *sphinx* and *numpydoc* packages and a *make* command you 
can build the documentation by running the *make* command in the *doc/* 
directory. For example, to create the HTML documentation you would do:

.. code-block:: console
  
  $ python setup.py build_sphinx
  $ make html

And then view doc/_build/html/index.html with a web browser.


Running Tests
-------------

You can install dcmstack along with any test dependencies by installing the `test` 
extra. Then you can run the `pytest` command to run the test-suite
