"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

This module contains the definition of the class FIGinfo

Copyright (c) 2020-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime


# - main function -------------------------
class FIGinfo:
    """
    The figure information constists of key, value combinations which are
    to be displayed 'above' or 'right' from the main image.

    Attributes
    ----------
    location : string
       Location to draw the fig_info box: 'above' (default), 'right', 'none'
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
      'right' :  Depending on the aspect ratio of the main image, the number of
                 lines are limited to 70 (aspect-ratio=1), 50 (aspect-ratio=2),
                 or 40 (aspect-ratio > 2).
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
        if loc not in ('above', 'right', 'none'):
            raise KeyError('location should be: above, right or none')

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
