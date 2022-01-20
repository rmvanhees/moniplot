"""
This file is part of moniplot

https://github.com/rmvanhees/moniplot.git

Create a blank rectangle matplotlib patch

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  GNU GPL v3.0
"""
import matplotlib as mpl


def blank_legend_key():
    """
    Show only text in matplotlib legenda, no key
    """
    return mpl.patches.Rectangle((0, 0), 0, 0, fill=False,
                                 edgecolor='none', visible=False)
