#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
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
