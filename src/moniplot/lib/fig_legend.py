"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Create a blank rectangle matplotlib patch

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
import matplotlib as mpl


def blank_legend_key():
    """
    Show only text in matplotlib legenda, no key
    """
    return mpl.patches.Rectangle((0, 0), 0, 0, fill=False,
                                 edgecolor='none', visible=False)
