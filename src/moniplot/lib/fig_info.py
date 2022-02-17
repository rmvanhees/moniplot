"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

This module contains the definition of the class FIGinfo

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
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime


# - main function -------------------------
class FIGinfo:
    """
    The figure information constists of [key, value] combinations which are
    to be displayed upper-right corner of the figure.

    Attributes
    ----------
    location : string
       Location to draw the fig_info box: 'above' (default), 'none'
    fig_info : OrderedDict
       Dictionary holding the information for the fig_info box

    Methods
    -------
    add(key, value, fmt='{}')
       Extent fig_info with a new line.
    as_str()
       Return figure information as one long string.
    copy()
       Return a deep copy of the current object.
    set_location(loc)
       Set the location of the fig_info box.

    Notes
    -----
    The box with the figure information can only hold a limited number of keys:
      'above' :  The figure information is displayed in a small box. This box
                 grows with the number of lines and will overlap with the main
                 image or its colorbar at about 7+ entries.
    """
    def __init__(self, loc='above', info_dict=None) -> None:
        self.fig_info = OrderedDict() if info_dict is None else info_dict
        self.set_location(loc)

    def __bool__(self) -> bool:
        return bool(self.fig_info)

    def __len__(self) -> int:
        return len(self.fig_info)

    def copy(self):
        """
        Return a deep copy of the current object
        """
        return deepcopy(self)

    def set_location(self, loc: str) -> None:
        """
        Set the location of the fig_info box

        Parameters
        ----------
        loc : str
          Location of the fig_info box
        """
        if loc not in ('above', 'none'):
            raise KeyError("location should be: 'above' or 'none'")

        self.location = loc

    def add(self, key: str, value, fmt='{}') -> None:
        """
        Extent fig_info with a new line

        Parameters
        ----------
        key : str
          Name of the fig_info key.
        value : Any python variable
          Value of the fig_info key. A tuple will be formatted as *value.
        fmt : str
          Convert value to a string, using the string format method.
          Default: '{}'.
        """
        if isinstance(value, tuple):
            self.fig_info.update({key: fmt.format(*value)})
        else:
            self.fig_info.update({key: fmt.format(value)})

    def as_str(self) -> str:
        """
        Return figure information as one long string
        """
        info_str = ''
        for key, value in self.fig_info.items():
            info_str += f'{key} : {value}\n'

        # add timestamp
        res = datetime.utcnow().isoformat(timespec='seconds')
        info_str += f'created : {res}'

        return info_str
