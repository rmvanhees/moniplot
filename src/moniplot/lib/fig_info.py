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
"""Definition of the moniplot class `FIGinfo`."""

from __future__ import annotations

__all__ = ["FIGinfo"]

import datetime as dt
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# - main function -------------------------
class FIGinfo:
    """Define figure information.

    The figure information consists of [key, value] combinations which are
    to be displayed upper-right corner of the figure.

    Parameters
    ----------
    loc :  str, default='above'
        Location to draw the fig_info box: 'above' (default), 'below', 'none'
    info_dict :  dict, optional
        Dictionary holding information to be displayed in the fig_info box

    Notes
    -----
    The figure-box can only hold a limited number of entries, because it will
    grow with the number of lines and overlap with the main image or its
    color bar. You may try loc='below', which is only available for image plots.

    """

    def __init__(
        self: FIGinfo, loc: str = "above", info_dict: dict | None = None
    ) -> None:
        """Create FIGinfo instance to hold information on the current plot."""
        self.fig_info = {} if info_dict is None else info_dict
        self._location = None
        self.set_location(loc)

    def __bool__(self: FIGinfo) -> bool:
        """Return True when an instance of fig_info exists."""
        return bool(self.fig_info)

    def __len__(self: FIGinfo) -> int:
        """Return the length of the instance fig_info."""
        return len(self.fig_info)

    def copy(self: FIGinfo) -> FIGinfo:
        """Return a deep copy of the current object."""
        return deepcopy(self)

    @property
    def location(self: FIGinfo) -> str:
        """Return location of the fig_info box."""
        return self._location

    def set_location(self: FIGinfo, loc: str) -> None:
        """Set the location of the fig_info box.

        Parameters
        ----------
        loc : str
          Location of the fig_info box

        """
        if loc not in ("above", "none"):
            raise KeyError("location should be: 'above' or 'none'")

        self._location = loc

    def add(self: FIGinfo, key: str, value: ..., fmt: str = "{}") -> None:
        r"""Extent fig_info by adding a new line.

        Parameters
        ----------
        key : str
          Name of the fig_info key
        value : Any python variable
          Value of the fig_info key. A tuple will be formatted as \*value
        fmt : str, default='{}'
          Convert value to a string, using the string format method

        """
        if isinstance(value, tuple):
            self.fig_info[key] = fmt.format(*value)
        else:
            self.fig_info[key] = fmt.format(value)

    def as_str(self: FIGinfo) -> str:
        """Return figure information as one long string."""
        info_str = ""
        for key, value in self.fig_info.items():
            info_str += f"{key}: {value}\n"

        # add timestamp
        res = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        info_str += f"created: {res.replace('+00:00', 'Z')}"

        return info_str

    def draw(self: FIGinfo, fig: Figure) -> None:
        """Display content of fig_info as a text block in the upper-right corner.

        Parameters
        ----------
        fig :  matplotlib.figure.Figure
           Matplotlib object Figure

        """
        if not self.fig_info or self.location != "above":
            return

        fig.text(
            1 - 0.4 / fig.get_figwidth(),
            1 - 0.25 / fig.get_figheight(),
            self.as_str(),
            fontsize="x-small" if len(self) > 5 else "small",
            style="normal",
            verticalalignment="top",
            horizontalalignment="right",
            multialignment="left",
            bbox={"facecolor": "white", "pad": 5},
        )
