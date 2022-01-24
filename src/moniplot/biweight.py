"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Python implementation of the Tukey's biweight algorithm.

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
import gc
import warnings

import numpy as np


# ----- local functions -------------------------
def __biweight(data, spread=False):
    """
    Calculate biweight parameters for the whole dataset
    """
    # calculate median and median absolute difference
    if np.isnan(data).any():
        biweight_median = np.nanmedian(data)
        delta = data - biweight_median
        delta_median = np.nanmedian(np.abs(delta))
    else:
        biweight_median = np.median(data)
        delta = data - biweight_median
        delta_median = np.median(np.abs(delta))
    if delta_median == 0.:
        if spread:
            return (biweight_median, 0.)
        return biweight_median

    # calculate biweight median
    wmx = np.clip(1 - (delta / (6 * delta_median)) ** 2, 0, None) ** 2
    biweight_median += np.nansum(wmx * delta) / np.nansum(wmx)
    del wmx
    gc.collect()
    if not spread:
        return biweight_median

    # calculate biweight variance
    umn = np.clip((delta / (9 * delta_median)) ** 2, None, 1)
    biweight_var = np.nansum(delta ** 2 * (1 - umn) ** 4)
    del delta
    gc.collect()
    biweight_var /= np.nansum((1 - umn) * (1 - 5 * umn)) ** 2
    biweight_var *= np.sum(np.isfinite(data))

    return (biweight_median, np.sqrt(biweight_var))


def __biweight_axis(data, axis, spread=False):
    """
    Calculate biweight parameters, along a given axis

    Notes
    -----
    Data should not contain any invalid value
    """
    # calculate median and median absolute difference
    if np.isnan(data).any():
        all_nan = np.isnan(data).all(axis=axis)
        biweight_median = np.nanmedian(data, axis=axis, keepdims=True)
        delta = data - biweight_median
        delta_median = np.nanmedian(np.abs(delta), axis=axis, keepdims=True)
        _mm = delta_median != 0.
        delta_median[~_mm] = np.nan
        _mm = np.squeeze(_mm) & ~all_nan
    else:
        biweight_median = np.median(data, axis=axis, keepdims=True)
        delta = data - biweight_median
        delta_median = np.median(np.abs(delta), axis=axis, keepdims=True)
        _mm = delta_median != 0.
        delta_median[~_mm] = np.nan
        _mm = np.squeeze(_mm)
    biweight_median = np.squeeze(biweight_median)

    # calculate biweight median
    wmx = np.clip(1 - (delta / (6 * delta_median)) ** 2, 0, None) ** 2
    biweight_median[_mm] += \
        np.nansum(wmx * delta, axis=axis)[_mm] / np.nansum(wmx, axis=axis)[_mm]
    del wmx
    gc.collect()
    if not spread:
        return biweight_median

    # calculate biweight variance
    umn = np.clip((delta / (9 * delta_median)) ** 2, None, 1)
    del delta_median
    biweight_var = np.nansum(delta ** 2 * (1 - umn) ** 4, axis=axis)
    del delta
    gc.collect()
    biweight_var[_mm] /= \
        np.nansum((1 - umn) * (1 - 5 * umn), axis=axis)[_mm] ** 2
    biweight_var[_mm] *= np.sum(np.isfinite(data), axis=axis)[_mm]

    return (biweight_median, np.sqrt(biweight_var))


# ----- main function -------------------------
def biweight(data, axis=None, spread=False):
    """
    Calculate Tukey's biweight.
    Implementation based on Eqn. 7.6 and 7.7 in the SWIR OCAL ATBD.

    Parameters
    ----------
    data   :   array_like
       input array
    axis   :   int, optional
       axis along which the biweight medians are computed.
        - Note that axis will be ignored when data is a 1-D array.
    spread :   bool, optional
       if True, then return also the biweight spread.

    Returns
    -------
    out    :   ndarray or tuple
       biweight median and biweight spread if parameter "spread" is True
    """
    if axis is None or data.ndim == 1:
        mask = np.isfinite(data)
        if np.all(~mask):
            if spread:
                return (np.nan, 0.)
            return np.nan
        return __biweight(data, spread)

    # only a single axis!
    if not isinstance(axis, int):
        raise TypeError('axis not an integer')
    if not 0 <= axis < data.ndim:
        raise ValueError('axis out-of-range')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        return __biweight_axis(data, axis, spread)
