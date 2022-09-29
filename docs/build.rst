.. _install:

Installation
============

We develop the code using `Python <https://www.python.org/>`_ 3.10 using the
latest stable release of the `HDF5 <https://hdfgroup.org/solutions/hdf5>`_
library and Python packages:
`numpy <https://numpy.org>`_, `h5py <https://www.h5py.org>`_,
`xarray <https://xarray.dev/>`_, `matplotlib <https://matplotlib.org/>`_
and `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_.

.. important::
   The installation of Cartopy is needed by MONplot::draw_tracks. However,
   we could not get our documentation installed on `Read the Docs`, because
   the build of `moniplot` failed during the installation of Cartopy.
   Therefore, we made the availability of Cartopy optional.

Wheels
------

It is highly recommended that you use a pre-built wheel of `moniplot` from PyPI.

If you have an existing Python (3.8+) installation (e.g. a python.org download,
or one that comes with your OS), then on Windows, MacOS/OSX, and Linux on
Intel computers, pre-built `moniplot` wheels can be installed via pip
from PyPI::

  pip install moniplot

OS-Specific remarks
-------------------

On a Debian Bullseye or Ubuntu 22.04 installation,
we have successfully installed `moniplot` as follows::

  sudo apt install python3-numpy python3-scipy python3-h5py
  sudo apt install python3-matplotlib
  sudo apt install python3-cartopy # optional
  pip install --user moniplot

This will also install a working version of the package xarray.

.. important::
   The version of xarray which comes with the Debian package
   `python3-xarray` is too old, and will not work with `moniplot`.

Building from source
--------------------

The latest release of `moniplot` is available from
`gitHub <https://github.com/rmvanhees/moniplot>`_.
You can obtain the source code using::

  git clone https://github.com/rmvanhees/moniplot.git

To compile the code you need the Python packages: setuptools, setuptools-scm
and wheels. Then you can install `moniplot` as follows::

  python3 -m build
  pip3 install dist/moniplot-<version>.whl [--user]

