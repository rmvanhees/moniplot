#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2019-2024 SRON - Netherlands Institute for Space Research
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
"""Python implementation of the Tukey's biweight algorithm."""

from __future__ import annotations

__all__ = ["Biweight", "biweight"]

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray


# ----- class Biweight -------------------------
class Biweight:
    """Python implementation of the Tukey's biweight algorithm.

    Parameters
    ----------
    data : array_like
       input array
    axis : int, optional
       axis along which the biweight medians are computed.

    Notes
    -----
    Parameter `data` should not contain any invalid value.

    Parameter `axis` will be ignored when data is a 1-D array.

    Raises
    ------
    TypeError
       If axis is not an integer.
    ValueError
       If axis is invalid, e.g. axis > data.ndim.

    Examples
    --------
    Calculate biweight median, spread and unbiased estimator::

       > from moniplot.biweight import Biweight
       > Biweight((1, 2, 2.5, 1.75, 2)).median
       1.962936462507155
       > Biweight((1, 2, 2.5, 1.75, 2)).spread
       0.5042069490893494
       > Biweight((1, 2, 2.5, 1.75, 2)).unbiased_std
       0.6131156500926488

    """

    def __init__(
        self: Biweight, data: ndarray | Iterable, axis: int | None = None
    ) -> None:
        """Initialize a Biweight object."""
        data = np.asarray(data)

        self.axis = axis
        self.__mask = None
        self.nr_valid = np.sum(np.isfinite(data), axis=axis)
        if axis is None or data.ndim == 1:
            if self.nr_valid == data.size:
                self.__med_data = np.median(data)
                self.__delta = data - self.__med_data
                self.__med_delta = np.median(np.abs(self.__delta))
            elif self.nr_valid == 0:
                self.__med_data = np.nan
                self.__delta = 0.0
                self.__med_delta = 0.0
            else:
                self.__med_data = np.nanmedian(data)
                self.__delta = data - self.__med_data
                self.__med_delta = np.nanmedian(np.abs(self.__delta))
        else:
            if np.isnan(data).any():
                all_nan = self.nr_valid == 0
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", r"All-NaN (slice|axis) encountered"
                    )
                    self.__med_data = np.nanmedian(data, axis=axis, keepdims=True)
                    self.__delta = data - self.__med_data
                    self.__med_delta = np.nanmedian(
                        np.abs(self.__delta), axis=axis, keepdims=True
                    )
                _mm = self.__med_delta != 0.0
                self.__med_delta[~_mm] = np.nan
                _mm = np.squeeze(_mm) & ~all_nan
            else:
                self.__med_data = np.median(data, axis=axis, keepdims=True)
                self.__delta = data - self.__med_data
                self.__med_delta = np.median(
                    np.abs(self.__delta), axis=axis, keepdims=True
                )
                _mm = self.__med_delta != 0.0
                self.__med_delta[~_mm] = np.nan
                _mm = np.squeeze(_mm)

            self.__mask = _mm
            self.__med_data = np.squeeze(self.__med_data)

    @property
    def median(self: Biweight) -> float | ndarray:
        """Return biweight median."""
        if self.axis is None:
            if self.__med_delta == 0:
                return self.__med_data

            wmx = (
                np.clip(1 - (self.__delta / (6 * self.__med_delta)) ** 2, 0, None) ** 2
            )
            self.__med_data += np.nansum(wmx * self.__delta) / np.nansum(wmx)
        else:
            wmx = (
                np.clip(1 - (self.__delta / (6 * self.__med_delta)) ** 2, 0, None) ** 2
            )
            self.__med_data[self.__mask] += (
                np.nansum(wmx * self.__delta, axis=self.axis)[self.__mask]
                / np.nansum(wmx, axis=self.axis)[self.__mask]
            )

        return self.__med_data

    @property
    def spread(self: Biweight) -> float | ndarray:
        """Return biweight spread."""
        if self.axis is None:
            if self.__med_delta == 0:
                return 0.0

            # calculate biweight variance
            umn = np.clip((self.__delta / (9 * self.__med_delta)) ** 2, None, 1)
            biweight_var = np.nansum(self.__delta**2 * (1 - umn) ** 4)
            biweight_var /= np.nansum((1 - umn) * (1 - 5 * umn)) ** 2
            biweight_var *= self.nr_valid
        else:
            umn = np.clip((self.__delta / (9 * self.__med_delta)) ** 2, None, 1)
            biweight_var = np.nansum(self.__delta**2 * (1 - umn) ** 4, axis=self.axis)
            _mm = self.__mask
            biweight_var[_mm] /= (
                np.nansum((1 - umn) * (1 - 5 * umn), axis=self.axis)[_mm] ** 2
            )
            biweight_var[_mm] *= self.nr_valid[_mm]

        return np.sqrt(biweight_var)

    @property
    def unbiased_std(self: Biweight) -> float | ndarray:
        """Return unbiased estimator."""
        count = self.nr_valid
        if self.axis is None:
            if count == 2:
                fact = 1.684 * self.spread
            elif count == 3:
                fact = 1.595 * self.spread
            else:
                fact = 0.9909 + (0.5645 + 2.805 / count) / count
            return fact * self.spread

        fact = np.full(count.shape, np.nan)
        fact[count == 2] = 1.684
        fact[count == 3] = 1.595
        mask = count > 3
        fact[mask] = 0.9909 + (0.5645 + 2.805 / count[mask]) / count[mask]
        while fact.ndim < self.spread.ndim:
            fact = fact[:, np.newaxis]

        return fact * self.spread


# ----- main function -------------------------
def biweight(
    data: ndarray | Iterable, axis: int | None = None, spread: bool = False
) -> ndarray | tuple[ndarray, ndarray]:
    """Python implementation of the Tukey's biweight algorithm.

    Parameters
    ----------
    data : array_like
       input array
    axis : int, optional
       axis along which the biweight medians are computed.
       Note that axis will be ignored when data is a 1-D array.
    spread : bool, optional
       if True, then return also the biweight spread.

    Returns
    -------
    out : ndarray or tuple
       biweight median and biweight spread if parameter "spread" is True

    """
    bwght = Biweight(data, axis=axis)
    if spread:
        return bwght.median, bwght.spread

    return bwght.median
