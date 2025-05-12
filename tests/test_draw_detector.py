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
"""..."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from moniplot.draw_detector import DrawDetImage, DrawDetQuality

def test(delta: float = 0.025) -> None:
    """..."""

    # generate image data
    xx = yy = np.arange(-3, 3, delta)
    xdata, ydata = np.meshgrid(xx, yy)
    zz1 = np.exp(-xdata ** 2 - ydata**2)
    zz2 = np.exp(-(xdata - 1) ** 2 - (ydata - 1) **2)
    zz = (zz1 - zz2) * 2

    with DrawDetImage(zz, side_panels=True) as plot:
        plot.set_caption("This is a test figure")
        plot.set_title("simple bivariate normal distribution")
        plot.add_side_panels()
        plot.add_fig_info()
        plot.add_copyright()

    plt.show()


def qtest(delta: float = 0.025) -> None:
    """..."""

    # generate image data
    xx = yy = np.arange(-3, 3, delta)
    xdata, ydata = np.meshgrid(xx, yy)
    zz1 = np.exp(-xdata ** 2 - ydata**2)
    zz2 = np.exp(-(xdata - 1) ** 2 - (ydata - 1) **2)
    zz = np.abs(zz1 - zz2).clip(0, 1)

    attrs = {
        "colors": ["#BBBBBB", "#EE6677", "#CCBB44", "#FFFFFF"],
        "long_name": "Pixel Quality",
        "thres_bad": 0.8,
        "thres_worst": 0.1,
        "flag_meanings": ["unusable", "worst", "bad", "good"],
        "flag_values": (-1, 0, 0.1, 0.8, 1),
    }
    
    with DrawDetQuality(zz, attrs=attrs, side_panels=True) as plot:
        plot.set_caption("This is a test figure")
        plot.set_title()
        plot.add_side_panels()
        plot.add_fig_info()
        plot.add_copyright()

    plt.show()


if __name__ == "__main__":
    test()
    qtest()
