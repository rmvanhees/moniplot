#
# This file is part of Python package: `moniplot`
#
#     https://github.com/rmvanhees/moniplot.git
#
# Copyright (c) 2019-2025 SRON
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
"""..."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np

from moniplot.draw_hist import DrawHist


def main(n_panel: int = 1) -> None:
    """..."""

    mu, sigma = 0, 0.1 # mean and standard deviation
    
    plot = DrawHist()
    fig, axx = plot.subplots(
        n_panel,
        xlim=[-0.425, 0.425],
        xlabel="noise",
        ylim=[0, 100],
        ylabel="number",
    )
    for ii in range(n_panel):
        plot.add_hist(np.random.normal(mu, sigma, 1000), clip=[-0.4, 0.4], bins=40)
        plot.draw(axx[ii])
        axx[ii].set_title(f"Figure {ii+1}")
    plt.show()



if __name__ == "__main__":
    main(4)
