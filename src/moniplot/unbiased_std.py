#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
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
Calculate the unbiased estimator for the standard deviation.
"""
__all__ = ['unbiased_std']

from math import pi, sqrt

import numpy as np
# pylint: disable=no-member
import scipy.special as sc


def unbiased_std(data):
    """Returns the unbiased estimator for the standard deviation.
    """
    nval = data.shape[0]
    unbias = sqrt((nval - 1) / (2 * pi)) * sc.beta((nval - 1) / 2, 1 / 2)

    return unbias * np.std(data, ddof=1, axis=0)
