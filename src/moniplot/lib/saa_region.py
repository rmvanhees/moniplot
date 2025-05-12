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
"""Read SAA definition from YAML file."""

from importlib.resources import files

import yaml


def saa_region() -> tuple[tuple[float, float], ...]:
    """Return SAA polygon."""
    yaml_fl = files("moniplot.Data").joinpath("saa_region.yaml")
    if not yaml_fl.is_file():
        raise FileNotFoundError(f"{yaml_fl} not found")

    with yaml_fl.open("r", encoding="ascii") as fid:
        try:
            res = yaml.safe_load(fid)
        except yaml.YAMLError as exc:
            raise RuntimeError("failed to read YAML file") from exc

    return tuple(zip(res[0]["saa_lon"], res[1]["saa_lat"], strict=True))
