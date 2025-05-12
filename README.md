# moniplot
[![Package Release](https://img.shields.io/pypi/v/moniplot.svg?label=version)](https://pypi.org/project/moniplot/)
[![Package Status](https://img.shields.io/pypi/status/moniplot.svg?label=status)](https://pypi.org/project/moniplot/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/moniplot.svg?label=PyPI%20downloads)](https://github.com/rmvanhees/moniplot/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7024743.svg)](https://doi.org/10.5281/zenodo.7024743)

Moniplot is a Python data visualization library based on
[matplotlib](https://matplotlib.org) and [xarray](https://xarray.pydata.org).
It provides a high-level interface for drawing attractive and informative
graphics for onground calibration data analysis or inflight instrument
monitoring.

## Documentation
Online documentation is available from [Read the Docs](https://moniplot.readthedocs.io).

## Installation
The module moniplot requires Python3.8+ and Python modules: cartopy, h5py, matplotlib, numpy and xarray.

Installation instructions are provided on [Read the Docs](https://moniplot.readthedocs.io/en/latest/build.html) or in the INSTALL file.

## Origin
Several modules from the package [pys5p](https://pypi.org/project/pys5p) have been moved to moniplot, because they are not specific for the data of Sentinel 5 precursor.
* module biweight.py - contains a Python implementation of the Tukey's biweight algorithm.
* module tol_colors.py - definition of colour schemes for lines and maps that also work for colour-blind
people by [Paul Tol](https://personal.sron.nl/~pault/).
* module s5p\_plot.py - the class S5Pplot is rewritten and now available as MONplot in the module mon_plot.py.


