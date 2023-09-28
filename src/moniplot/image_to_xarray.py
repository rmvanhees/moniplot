#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2023 SRON - Netherlands Institute for Space Research
#
# License:  GPLv3
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This module contains the routines `h5_to_xr` and `data_to_xr`.

These functions store a HDF5 dataset or numpy array in a labeled array
(class `xarray.DataArray`).
"""
from __future__ import annotations

__all__ = ['h5_to_xr', 'data_to_xr']

from pathlib import PurePath
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import h5py


# - local functions --------------------------------
def __get_attrs(dset: h5py.Dataset, field: str) -> dict:
    """Return attributes of the HDF5 dataset.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the attributes are read
    field : str
       Name of field in compound dataset

    Returns
    -------
    dict with numpy arrays
    """
    _field = None
    if field is not None:
        try:
            _field = {'name': field,
                      'oneof': len(dset.dtype.names),
                      'index': dset.dtype.names.index(field)}
        except Exception as exc:
            raise RuntimeError(
                f'field {field} not found in dataset {dset.name}') from exc
        # print('_field ', _field)

    attrs = {}
    for key in dset.attrs:
        if key in ('CLASS', 'DIMENSION_LIST', 'NAME', 'REFERENCE_LIST',
                   '_Netcdf4Dimid', '_Netcdf4Coordinates'):
            continue

        attr_value = dset.attrs[key]
        # print('# ----- ', key, type(attr_value), attr_value)
        if isinstance(attr_value, np.ndarray):
            if len(attr_value) == 1:
                attr_value = attr_value[0]
                # print('# ----- ', key, type(attr_value), attr_value)
            elif _field is not None and len(attr_value) == _field['oneof']:
                attr_value = attr_value[_field['index']]
                # elif isinstance(attr_value, np.void):
                #    attr_value = attr_value[0]

        attrs[key] = (attr_value.decode('ascii')
                      if isinstance(attr_value, bytes) else attr_value)

    return attrs


def __get_coords(dset: h5py.Dataset, data_sel: tuple[slice | int]) -> list:
    r"""Return coordinates of the HDF5 dataset with dimension scales.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the data is read
    data_sel :  tuple of slice or int
       A numpy slice generated for example `numpy.s\_`

    Returns
    -------
    A sequence of tuples [(dims, data), ...]
    """
    coords = []
    if len(dset.dims) == dset.ndim:
        try:
            for ii, dim in enumerate(dset.dims):
                # get name of dimension
                name = PurePath(dim[0].name).name
                if name.startswith('row') or name.startswith('column'):
                    name = name.split(' ')[0]

                # determine coordinate
                buff = None
                if dim[0].size > 0 and not np.all(dim[0][()] == 0):
                    buff = dim[0][()]
                elif name in ('row', 'column'):
                    d_type = 'u2' if ((dset.shape[ii]-1) >> 16) == 0 else 'u4'
                    buff = np.arange(dset.shape[ii], dtype=d_type)

                if not (buff is None or data_sel is None):
                    buff = buff[data_sel[ii]]

                coords.append((name, buff))
        except RuntimeError:
            coords = []

    return coords


def __set_coords(dset: h5py.Dataset | np.ndarray,
                 data_sel: tuple[slice | int] | None,
                 dims: list | None) -> list:
    r"""Set coordinates of the HDF5 dataset.

    Parameters
    ----------
    dset :  h5py.Dataset or np.ndarray
       HDF5 dataset from which the data is read, or numpy array
    data_sel :  tuple of slice or int
       A numpy slice generated for example `numpy.s\_`
    dims : list of strings
       Alternative names for the dataset dimensions if not attached to dataset
       Default coordinate names are ['time', ['row', ['column']]]

    Returns
    -------
    A sequence of tuples [(dims, data), ...]
    """
    if dims is None:
        if dset.ndim > 3:
            raise ValueError('not implemented for ndim > 3')

        dims = ['time', 'row', 'column'][-dset.ndim:]

    coords = []
    for ii in range(dset.ndim):
        co_dtype = 'u2' if ((dset.shape[ii]-1) >> 16) == 0 else 'u4'
        buff = np.arange(dset.shape[ii], dtype=co_dtype)
        if data_sel is not None:
            buff = buff[data_sel[ii]]
        coords.append((dims[ii], buff))

    return coords


def __get_data(dset: h5py.Dataset, data_sel: tuple[slice | int] | None,
               field: str) -> np.ndarray:
    r"""Return data of the HDF5 dataset.

    Parameters
    ----------
    dset :  h5py.Dataset
       HDF5 dataset from which the data is read
    data_sel :  tuple of slice or int
       A numpy slice generated for example `numpy.s\_`
    field : str
       Name of field in compound dataset or None

    Returns
    -------
    Numpy array

    Notes
    -----
    Read floats always as doubles
    """
    if data_sel is None:
        data_sel = ()

    if np.issubdtype(dset.dtype, np.floating):
        data = dset.astype(float)[data_sel]
        data[data == float.fromhex('0x1.ep+122')] = np.nan
        return data

    if field is None:
        return dset[data_sel]

    data = dset.fields(field)[data_sel]
    if np.issubdtype(data.dtype, np.floating):
        data = data.astype(float)
        data[data == float.fromhex('0x1.ep+122')] = np.nan
    return data


def __check_selection(data_sel: slice | tuple | int,
                      ndim: int) -> slice | tuple | None:
    r"""Check and correct user provided data selection.

    Notes
    -----
    If data_sel is used to select data from a dataset then the number of
    dimensions of data_sel should agree with the HDF5 dataset or one and
    only one Ellipsis has to be used.
    Thus allowed values for data_sel are:
    * [always]: (), np.s\_[:], np.s\_[...]
    * [1-D dataset]: np.s\_[:-1], np.s\_[0]
    * [2-D dataset]: np.s\_[:-1, :], np.s\_[0, :], np.s\_[:-1, 0]
    * [3-D dataset]: np.s\_[:-1, :, 2:4], np.s\_[0, :, :], np.s\_[:-1, 0, 2:4]
    * [Ellipsis] np.s\_[0, ...], np.s\_[..., 4], np.s\_[0, ..., 4]
    """
    if data_sel in (np.s_[:], np.s_[...], np.s_[()]):
        return None

    if np.isscalar(data_sel):
        return np.s_[data_sel:data_sel+1]

    buff = ()
    for val in data_sel:
        if val == Ellipsis:
            for _ in range(ndim - len(data_sel) + 1):
                buff += np.index_exp[:]
        elif np.isscalar(val):
            buff += (np.s_[val:val+1],)
        else:
            buff += (val,)

    return buff


# - main function ----------------------------------
def h5_to_xr(h5_dset: h5py.Dataset, data_sel: tuple[slice | int] | None = None,
             *, dims: list[str] | None = None,
             field: str | None = None) -> xr.DataArray:
    r"""Create xarray.DataArray from a HDF5 dataset (with dimension scales).

    Implements a lite interface with the xarray.DataArray, should work for all
    2-D detector images, sequences of detector measurements and trend data.

    Parameters
    ----------
    h5_dset :  h5py.Dataset
       Data, dimensions, coordinates and attributes are read for this dataset
    data_sel :  tuple of slice or int, optional
       A numpy slice generated for example `numpy.s\_`
    dims :  list of strings, optional
       Alternative names for the dataset dimensions if not attached to dataset
    field : str, optional
       Name of field in compound dataset or None

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    All floating datasets are converted to Python type 'float'

    Dimensions and Coordinates:

    * The functions in this module should work with netCDF4 and HDF5 files.
    * In a HDF5 file the 'coordinates' of a dataset can be defined using \
      dimension scales.
    * In a netCDF4 file this is required: all variables have dimensions, \
      which can have coordinates. But under the hood also netCDF4 uses \
      dimension scales.
    * The xarray DataArray structure will have as dimensions, the names of \
      the dimension scales and as coordinates the names and data of the \
      dimensions scales, except when the data only contains zero's.
    * The default dimensions of an image are 'row' and 'column' with evenly \
      spaced values created with np.arange(len(dim), dtype=uint).

    Examples
    --------
    Read HDF5 dataset 'signal' from file::

    > fid = h5py.File(flname, 'r')        # flname is a HDF5/netCDF4 file
    > xdata = h5_to_xr(fid['signal'])
    > fid.close()

    Combine Tropomi SWIR data of band7 and band8::

    > fid = h5py.File(s5p_b7_prod, 'r')   # Tropomi band7 product
    > xdata7 = h5_to_xr(fid['signal'])
    > fid.close()
    > fid = h5py.File(s5p_b8_prod, 'r')   # Tropomi band8 product
    > xdata8 = h5_to_xr(fid['signal'])
    > fid.close()
    > xdata = xr.concat((xdata7, xdata8), dim='spectral_channel')

    Optionally, fix the 'column' dimension::

    > xdata = xdata.assign_coords(
    > ... column = np.arange(xdata.column.size, dtype='u4'))
    """
    # Check data selection
    if data_sel is not None:
        data_sel = __check_selection(data_sel, h5_dset.ndim)

    # Name of this array
    name = PurePath(h5_dset.name).name if field is None else field

    # Values for this array
    data = __get_data(h5_dset, data_sel, field)

    # Coordinates (tick labels) to use for indexing along each dimension
    coords = []
    if dims is None:
        coords = __get_coords(h5_dset, data_sel)
    if not coords:
        coords = __set_coords(h5_dset, data_sel, dims)

    # - check if dimension of dataset and coordinates agree
    if data.ndim < len(coords):
        for ii in reversed(range(len(coords))):
            if np.isscalar(coords[ii][1]):
                del coords[ii]

    # - remove empty coordinates
    dims = []
    co_dict = {}
    for key, val in coords:
        dims.append(key)
        if val is not None:
            co_dict[key] = val

    # Attributes to assign to the array
    attrs = __get_attrs(h5_dset, field)

    return xr.DataArray(data,
                        coords=co_dict, dims=dims, name=name, attrs=attrs)


def data_to_xr(data: np.ndarray, *, dims: list[str] | None = None,
               name: str | None = None, long_name: str | None = None,
               units: str | None = None) -> xr.DataArray:
    """Create xarray.DataArray from a dataset.

    Implements a lite interface with the xarray.DataArray, should work for all
    2-D detector images, sequences of detector measurements and trend data.

    Parameters
    ----------
    data :  np.ndarray
       Data to be stored in the xarray
    dims :  list of strings, optional
       Names for the dataset dimensions
    name : str, optional
       A string that names the instance
    units :  str, optional
       Units of the data, default: '1'
    long_name : str, optional
       Long name describing the data, default: empty string

    Returns
    -------
    xarray.DataArray

    Notes
    -----
    All floating datasets are converted to Python type 'float'
    """
    coords = __set_coords(data, None, dims)
    attrs = {'units': '1' if units is None else units,
             'long_name': '' if long_name is None else long_name}

    return xr.DataArray(data,
                        coords=coords, name=name, attrs=attrs)
