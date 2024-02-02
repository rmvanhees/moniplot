#
# https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2023-2024 SRON - Netherlands Institute for Space Research
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
"""Read SAA definition from YAML file."""

from importlib.resources import files

import yaml


def saa_region() -> tuple[tuple[float, float], ...]:
    """Return SAA polygon."""
    yaml_fl = files("moniplot.data").joinpath("saa_region.yaml")
    if not yaml_fl.is_file():
        raise FileNotFoundError(f"{yaml_fl} not found")

    with yaml_fl.open("r", encoding="ascii") as fid:
        try:
            res = yaml.safe_load(fid)
        except yaml.YAMLError as exc:
            raise RuntimeError("failed to read YAML file") from exc

    return tuple(zip(res[0]["saa_lon"], res[1]["saa_lat"], strict=True))
