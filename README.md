# moniplot
[![PyPI Latest Release](https://img.shields.io/pypi/v/moniplot.svg)](https://pypi.org/project/moniplot/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7024743.svg)](https://doi.org/10.5281/zenodo.7024743)
[![Package Status](https://img.shields.io/pypi/status/moniplot.svg)](https://pypi.org/project/moniplot/)
[![License](https://img.shields.io/pypi/l/moniplot.svg)](https://github.com/rmvanhees/moniplot/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/moniplot?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/moniplot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Moniplot is a Python data visualization library based on
[matplotlib](https://matplotlib.org) and [xarray](https://xarray.pydata.org).
It provides a high-level interface for drawing attractive and informative
graphics for onground calibration data analysis or inflight instrument
monitoring. 
 

Several modules from the package [pys5p](https://pypi.org/project/pys5p) have been moved to moniplot, because they are not specific for the data of Sentinel 5 precursor.
* module biweight.py - contains a Python implementation of the Tukey's biweight algorithm.
* module tol_colors.py - definition of colour schemes for lines and maps that also work for colour-blind
people by [Paul Tol](https://personal.sron.nl/~pault/).
* module s5p_plot.py - the class S5Pplot is rewritten and now available as MONplot in the module mon_plot.py.

The module pys5p requires Python3.8+ and Python modules: cartopy, h5py, matplotlib, numpy and xarray.
