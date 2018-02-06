Contributing
------------

If you find errors, omissions, inconsistencies or other things that need
improvement, please create an issue or a pull request at
https://github.com/sfstoolbox/sfs-python/.
Contributions are always welcome!

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

Instead of pip-installing the latest release from PyPI, you should get the
newest development version from Github_::

   git clone https://github.com/sfstoolbox/sfs-python.git
   cd sfs-python
   python3 setup.py develop --user

.. _Github: https://github.com/sfstoolbox/sfs-python/

This way, your installation always stays up-to-date, even if you pull new
changes from the Github repository.

If you prefer, you can also replace the last command with::

   python3 -m pip install --user -e .

... where ``-e`` stands for ``--editable``.

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you make changes to the documentation, you can re-create the HTML pages
using Sphinx_.
You can install it and a few other necessary packages with::

   python3 -m pip install -r doc/requirements.txt --user

To create the HTML pages, use::

   python3 setup.py build_sphinx

The generated files will be available in the directory ``build/sphinx/html/``.

It is also possible to automatically check if all links are still valid::

   python3 setup.py build_sphinx -b linkcheck

.. _Sphinx: http://sphinx-doc.org/

Running the Tests
^^^^^^^^^^^^^^^^^

You'll need pytest_ for that.
It can be installed with::

   python3 -m pip install -r tests/requirements.txt --user

To execute the tests, simply run::

   python3 -m pytest

.. _pytest: https://pytest.org/

Creating a New Release
^^^^^^^^^^^^^^^^^^^^^^

New releases are made using the following steps:

#. Bump version number in ``sfs/__init__.py``
#. Update ``NEWS.rst``
#. Commit those changes as "Release x.y.z"
#. Create an (annotated) tag with ``git tag -a x.y.z``
#. Clear the ``dist/`` directory
#. Create a source distribution with ``python3 setup.py sdist``
#. Create a wheel distribution with ``python3 setup.py bdist_wheel``
#. Check that both files have the correct content
#. Upload them to PyPI with twine_: ``twine upload dist/*``
#. Push the commit and the tag to Github and `add release notes`_ containing a
   link to PyPI and the bullet points from ``NEWS.rst``
#. Check that the new release was built correctly on RTD_, delete the "stable"
   version and select the new release as default version

.. _twine: https://pypi.python.org/pypi/twine
.. _add release notes: https://github.com/sfstoolbox/sfs-python/tags
.. _RTD: http://readthedocs.org/projects/sfs/builds/
