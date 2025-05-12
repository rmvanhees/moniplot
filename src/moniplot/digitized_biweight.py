#
# This file is part of Python package: `moniplot`
#
#     https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2025 SRON
#    All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Calculate Perform a (fancy) median smoothing on your data (x, y)."""

from __future__ import annotations

__all__ = ["digitized_biweight"]

from typing import TYPE_CHECKING

import numpy as np

from .biweight import Biweight

if TYPE_CHECKING:
    from numpy.typing import NDArray

ONE_DAY = np.timedelta64(1, "D")


# - main function ----------------------------------
def digitized_biweight(
    x_in: NDArray[np.datetime64],
    y_in: NDArray[float],
    *,
    bin_size: int = 1,
    y_in_bins: bool = False,
    remove_empty_bins: bool = False,
) -> tuple[NDArray, ...] | None:
    """Perform a (fancy) median smoothing on your data (x_in, y_in).

    The x-values within the interval "x_range" are divided in "x_samples" bins.
    The number of samples within each bin and biweight median of all
    (x,y) values within each bin is returned.

    Parameters
    ----------
    x_in :   array_like
      x-coordinates of the sample points, dtype='datetime64[s]'
    y_in :   array_like
      y-coordinates of the sample points, NaN values are discarded
    bin_size : int, default=1
      size of the bins in days, will be neglected when x_bins is provided
    y_in_bins :   bool, default=False
      when True then return array with y-samples per bin
    remove_empty_bins :   bool, default=True
      when True, then remove empty bins from the returned arrays

    Returns
    -------
     xbinned   :   ndarray
     ybinned   :   ndarray
     count     :   ndarray
     y_per_bin :   list of ndarrays, optional

    """
    mask = np.isfinite(y_in)
    xarr = np.copy(x_in[mask])
    yarr = np.copy(y_in[mask])
    xbinned = np.arange(
        xarr.min(), xarr.max() + ONE_DAY, bin_size, dtype="datetime64[D]"
    )

    # allocate memory
    ybinned = np.full(xbinned.size, np.nan)
    count = np.zeros(xbinned.size, dtype="u4")
    y_per_bin = []

    # collect data per bin
    indx = np.searchsorted(xbinned, xarr, side="right")
    for ii in range(xbinned.size):
        mask = indx == ii + 1
        if mask.sum() > 0:
            count[ii] = mask.sum()
            ybinned[ii] = Biweight(yarr[mask]).median
            if y_in_bins:
                y_per_bin.append(yarr[mask])
        else:
            if y_in_bins and not remove_empty_bins:
                y_per_bin.append(np.empty(0, dtype=float))

    # remove empty bins
    res = (xbinned, ybinned, count)
    if remove_empty_bins:
        mask = count > 0
        res = (xbinned[mask], ybinned[mask], count[mask])

    return (*res, y_per_bin) if y_in_bins else res
