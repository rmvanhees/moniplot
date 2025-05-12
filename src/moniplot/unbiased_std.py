#
# This file is part of Python package: `moniplot`
#
#     https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2019-2025 SRON
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
"""Calculate the unbiased estimator for the standard deviation."""

from __future__ import annotations

__all__ = ["unbiased_std"]

from math import pi, sqrt

import numpy as np

# pylint: disable=no-member
import scipy.special as sc


def unbiased_std(data: np.ndarray) -> np.ndarray:
    """Return the unbiased estimator for the standard deviation."""
    nval = data.shape[0]
    unbias = sqrt((nval - 1) / (2 * pi)) * sc.beta((nval - 1) / 2, 1 / 2)

    return unbias * np.std(data, ddof=1, axis=0)
