#
# This file is part of Python package: `moniplot`
#
#     https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2025 SRON
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
"""Perform a unit test on MONplot::draw_lplot."""

from __future__ import annotations

import numpy as np


def test_lplot() -> None:
    """Extensive test with MONplot::draw_lplot()."""
    from moniplot.mon_plot import MONplot

    plot = MONplot("test_lplot.pdf")
    plot.set_cset("muted")  # Note the default is 'bright'
    for ii in range(5):
        plot.draw_lplot(np.arange(10), np.arange(10) * (ii + 1), label=f"label {ii}")
    plot.draw_lplot(xlabel="x-axis", ylabel="y-axis", title='draw_lplot [cset="muted"]')

    for ii, clr in enumerate("rgbym"):
        plot.draw_lplot(
            np.arange(10), np.arange(10) * (ii + 1), color=clr, label=f"label {ii}"
        )
    plot.draw_lplot(xlabel="x-axis", ylabel="y-axis", title='draw_lplot [cset="rgbym"]')

    plot.set_cset("high-contrast")  # Note the default is 'bright'
    for ii in range(5):
        plot.draw_lplot(ydata=np.arange(10) * (ii + 1), label=f"label {ii}")
    plot.draw_lplot(
        xlabel="x-axis", ylabel="y-axis", title='draw_lplot [cset="high-contrast"]'
    )

    plot.set_cset("rainbow_PuBr", 35)
    for ii in range(35):
        plot.draw_lplot(ydata=np.arange(10) * (ii + 1), label=f"label {ii}")
    plot.draw_lplot(
        xlabel="x-axis", ylabel="y-axis", title='draw_lplot [cset="rainbow_PyBr"]'
    )

    for ii in range(35):
        plot.draw_lplot(ydata=np.arange(10) * (ii + 1), label=f"label {ii}")
    plot.draw_lplot(
        xlabel="x-axis",
        ylabel="y-axis",
        title='draw_lplot [cset="rainbow_PyBr"]',
        kwlegend={
            "fontsize": "x-small",
            "loc": "upper left",
            "bbox_to_anchor": (0.975, 1),
        },
    )
    plot.close()
