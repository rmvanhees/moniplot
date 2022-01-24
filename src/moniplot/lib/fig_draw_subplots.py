"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Functions
---------

Notes
-----
The goal of the main function ... is to draw a number of subplots which all
have a common X-coordinate. The function is designed to draw in one or more
subplots line-plots with one or lines (all with the units). The units of the
data shown in each subplot may have different units.

The input data should be provided as a xarray Dataset with one or more
DataArrays with one common X-coordinate, which a DataArray has two dimensions
then each Y-element (limited to TBD) will be shown as an line in the plot.
The attribute "long_name" will be shown as the label of each subplot. The name
of the data coordinates and data units will be used for resp. xlabel and
ylabel.

Copyright (c) 2022 SRON - Netherlands Institute for Space Research

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

License:  GPLv3
"""
import numpy as np
import xarray as xr
