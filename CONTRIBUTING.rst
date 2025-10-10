Contributing
------------

If you find errors, omissions, inconsistencies or other things that need
improvement, please create an issue or a pull request at
https://github.com/sfstoolbox/sfs-python/.
Contributions are always welcome!

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

Instead of installing the latest release from PyPI_, you should get the
newest development version from Github_::

   git clone https://github.com/sfstoolbox/sfs-python.git
   cd sfs-python
   uv sync

.. _PyPI: https://pypi.org/project/sfs/
.. _Github: https://github.com/sfstoolbox/sfs-python/


Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you make changes to the documentation, you can re-create the HTML pages
using Sphinx_.  From the main ``sfs-python`` directory, run::

   uv run sphinx-build doc _build

The generated files will be available in the directory ``_build/``.

.. _Sphinx: http://sphinx-doc.org/


Running the Tests
^^^^^^^^^^^^^^^^^

You'll need pytest_, which will be installed automatically.
To execute the tests, simply run::

   uv run pytest

.. _pytest: https://pytest.org/


Editable Installation
^^^^^^^^^^^^^^^^^^^^^

If you want to work in a different directory on your own files,
but using the latest development version (or a custom branch) of
the ``sfs`` module, you can switch to a directory of your choice
and enter this::

   uv init --bare
   uv add --editable --no-workspace path/to/your/sfs/repo

You can install further packages with ``uv add`` and then run
whatever you need with ``uv run``.


Creating a New Release
^^^^^^^^^^^^^^^^^^^^^^

It is proposed to perform these steps for creating a new release:

#. Bump version number in ``sfs/__init__.py``
#. Update ``NEWS.rst``
#. Commit those changes as "Release x.y.z"
#. Create an (annotated) tag with ``git tag -a x.y.z``
#. Push the commit and the tag to Github
#. Github workflow publish.yml will handle the build and upload to PyPi
#. `Add release notes`_ containing a link to PyPI and the bullet points
   from ``NEWS.rst``
#. Check that the new release was built correctly on RTD_
   and select the new release as the default version

Alternatively, an extra branch and PR can be utilised as follows:

#. Create a branch from the current master branch state. In this new branch:
#. Bump version number in ``sfs/__init__.py``
#. Update ``NEWS.rst``
#. Commit those changes as "Release x.y.z", push this commit and make a PR.
   This will trigger workflows to check that everything is fine, the PR is a
   nice additional documentation of this release process.
#. Create an (annotated) tag with ``git tag -a x.y.z`` and push it
#. Due to this tag push Github workflow publish.yml will handle the build
   and upload to PyPi
#. Merge the PR by ``git merge branch_name --ff-only``. This is needed for
   preserving the release and tag commits as they need to appear exactly this
   way in the master branch history.
#. Delete the branch
#. `Add release notes`_ containing a link to PyPI and the bullet points
   from ``NEWS.rst``
#. Check that the new release was built correctly on RTD_
   and select the new release as the default version


.. _add release notes: https://github.com/sfstoolbox/sfs-python/tags
.. _RTD: https://readthedocs.org/projects/sfs-python/builds/
