Installing moniplot
===================

Building from source
--------------------

The latest release of moniplot is available from:
[gitHub](https://github.com/rmvanhees/moniplot).

Where you can download the source as a tar-file or zipped archive.
Or you can use git do download the repository:

    git clone https://gitlab.sron.nl/Richardh/moniplot.git
	
Before you can install moniplot, you need:

 * Python v3.8+ with development headers
 * HDF5 v1.10+, installed with development headers

And the following Python modules have to be available:
 * setuptools v60+
 * setuptools-scm v6+
 * numpy v1.22+
 * h5py v3.5+
 * xarray v0.20+
 * matplotlib v3.5+
	
You can install moniplot once you have satisfied the requirements listed
above. Run at the top of the source tree:

    python3 -m build
    pip3 install dist/moniplot-<version>.whl [--user]
