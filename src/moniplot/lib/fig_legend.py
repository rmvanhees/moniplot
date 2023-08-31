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
"""This module contains `blank_legend_handle`."""

from __future__ import annotations

__all__ = ['blank_legend_handle']

from matplotlib.patches import Rectangle


def blank_legend_handle() -> Rectangle:
    """Show only label in a legend entry, no handle.

    See Also
    --------
    matplotlib.pyplot.legend : Place a legend on the Axes.
    """
    return Rectangle((0, 0), 0, 0,
                     fill=False,
                     edgecolor='none',
                     visible=False)
