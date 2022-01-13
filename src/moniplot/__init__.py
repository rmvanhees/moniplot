# This file is part of moniplot
#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2022 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  GNU GPL v3.0

""" The pys5p package contains software to read S5p Tropomi L1B products.
    And contains plotting routines to display your data beautifully."""

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass
