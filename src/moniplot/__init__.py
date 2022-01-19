# This file is part of moniplot
#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  GNU GPL v3.0

""" Moniplot is a Python data visualization library for (satellite)
instrument monitoring.
"""

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass
