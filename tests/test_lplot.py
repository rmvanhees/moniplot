#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022-2024 SRON - Netherlands Institute for Space Research
# All rights reserved.
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
