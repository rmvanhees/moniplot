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
"""Perform a limited unit test on the methods of the class `MONplot`."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from moniplot.image_to_xarray import data_to_xr
from moniplot.lib.fig_info import FIGinfo
from moniplot.mon_plot import MONplot


def get_test_data(
    data_sel: list[tuple, tuple] | None = None,
    xy_min: float = -5,
    xy_max: float = 5,
    delta: float = 0.01,
    error: float = 0,
) -> xr.DataArray:
    """Generate synthetic data to simulate a square-detector image."""
    if data_sel is None:
        data_sel = [(), ()]

    res = np.arange(xy_min, xy_max, delta)
    xmesh, ymesh = np.meshgrid(res[data_sel[1]], res[data_sel[0]])
    zz1 = np.exp(-(xmesh**2) - ymesh**2)
    zz2 = np.exp(-((xmesh - 1) ** 2) - (ymesh - 1) ** 2)
    data = (zz1 - zz2) * 2
    data += np.random.default_rng().normal(0.0, error, data.shape)

    return data_to_xr(data, long_name="bogus data", units="Volt")


def test_lplot() -> xr.DataArray:
    """Run unit tests on MONplot::draw_lplot."""
    print("Run unit tests on MONplot::draw_lplot")
    plot = MONplot("mon_plot_draw_lplot-1.png")
    plot.set_institute("SRON")
    plot.set_cset("muted")
    for ii in range(5):
        plot.draw_lplot(np.arange(10), np.arange(10) * (ii + 1))
    plot.draw_lplot(xlabel="x-axis", ylabel="y-axis", title='draw_lplot [cset="muted"]')
    plot.close()

    plot = MONplot("mon_plot_draw_lplot-2.png")
    plot.set_institute("SRON")
    for ii, clr in enumerate("rgbym"):
        plot.draw_lplot(np.arange(10), np.arange(10) * (ii + 1), color=clr)
    plot.draw_lplot(xlabel="x-axis", ylabel="y-axis", title='draw_lplot [cset="rgbym"]')
    plot.close()

    plot = MONplot("mon_plot_draw_lplot-3.png")
    plot.set_institute("SRON")
    plot.set_cset("mute")
    for ii in range(5):
        plot.draw_lplot(ydata=np.arange(10) * (ii + 1))
    plot.draw_lplot(xlabel="x-axis", ylabel="y-axis", title='draw_lplot [cset="mute"]')
    plot.close()

    plot = MONplot("mon_plot_draw_lplot-4.png")
    plot.set_institute("SRON")
    plot.set_cset("rainbow_PuBr", 25)
    for ii in range(25):
        plot.draw_lplot(ydata=np.arange(10) * (ii + 1))
    plot.draw_lplot(
        xlabel="x-axis", ylabel="y-axis", title='draw_lplot [cset="rainbow_PyBr"]'
    )
    plot.close()

    # time axis
    plot = MONplot("mon_plot_draw_lplot-5.png")
    plot.set_institute("SRON")
    customdate = datetime(2016, 1, 1, 13, 0, 0)
    yval = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    xval = [customdate + timedelta(hours=i, minutes=4 * i) for i in range(len(yval))]
    plot.draw_lplot(xval, yval, label="mydata", marker="o", linestyle="-")
    plot.draw_lplot(title="draw_lplot [time_axis]", xlabel="t-axis", ylabel="y-axis")
    plot.close()


def test_qhist() -> xr.Dataset:
    """Run unit tests on MONplot::draw_qhist."""
    print("Run unit tests on MONplot::draw_qhist")
    plot = MONplot("mon_plot_draw_qhist-1.png")
    plot.set_institute("SRON")
    rng = np.random.default_rng()
    buff0 = np.repeat(0.9 + rng.random(1000) / 10, 56)
    buff1 = np.repeat(rng.random(1000) / 10, 10)
    buff2 = 0.1 + rng.random(1000) / 10
    buff3 = np.repeat(0.2 + rng.random(1000) / 10, 2)
    buff4 = np.repeat(0.3 + rng.random(1000) / 10, 3)
    buff5 = np.repeat(0.4 + rng.random(1000) / 10, 4)
    buff6 = np.repeat(0.5 + rng.random(1000) / 10, 8)
    buff7 = np.repeat(0.6 + rng.random(1000) / 10, 12)
    buff8 = np.repeat(0.7 + rng.random(1000) / 10, 20)
    buff9 = np.repeat(0.8 + rng.random(1000) / 10, 40)
    buffa = np.repeat(0.9 + rng.random(1000) / 10, 100)
    frame = np.concatenate(
        (buff0, buff1, buff2, buff3, buff4, buff5, buff6, buff7, buff8, buff9, buffa)
    ).reshape(256, 1000)
    msm = xr.merge(
        [
            data_to_xr(frame, name="dpqm", long_name="pixel-quality map"),
            data_to_xr(frame, name="dpqm_dark", long_name="pixel-quality map (dark)"),
            data_to_xr(
                frame, name="dpqm_noise", long_name="pixel-quality map (noise average)"
            ),
            data_to_xr(
                frame,
                name="dpqm_noise_var",
                long_name="pixel-quality map (noise variance)",
            ),
        ]
    )
    plot.draw_qhist(msm, data_sel=np.s_[11:228, 16:991], title="my histogram")
    plot.close()


def test_quality() -> None:
    """Run unit tests on MONplot::draw_quality."""
    print("Run unit tests on MONplot::draw_quality")
    row = np.linspace(0, 1.0, 1000)
    ref_data = np.repeat(row[None, :], 256, axis=0)

    data = ref_data.copy()
    data[125:175, 30:50] = 0.4
    data[125:175, 50:70] = 0.95
    data[75:125, 250:300] = 0.05
    data[125:175, 600:650] = 0.95
    data[75:125, 900:925] = 0.05
    data[75:125, 825:875] = 0.4

    figinfo_in = FIGinfo()
    figinfo_in.add("orbits", (17, [23662, 23707]), fmt="{} in {}")
    figinfo_in.add("coverage", ("2022-05-08", "2022-05-10"), fmt="{} / {}")

    plot = MONplot("mon_plot_draw_quality-1.png")
    plot.set_institute("SRON")
    plot.draw_quality(
        data,
        data_sel=np.s_[11:228, 16:991],
        title="no reference",
        fig_info=figinfo_in.copy(),
    )
    plot.close()
    plot = MONplot("mon_plot_draw_quality-2.png")
    plot.set_institute("SRON")
    plot.draw_quality(
        data,
        ref_data=ref_data,
        data_sel=np.s_[11:228, 16:991],
        title="with reference",
        fig_info=figinfo_in.copy(),
    )
    plot.close()
    plot = MONplot("mon_plot_draw_quality-3.png")
    plot.set_institute("SRON")
    plot.draw_quality(
        data,
        data_sel=np.s_[11:228, 16:991],
        side_panels="none",
        title="no reference",
        fig_info=figinfo_in.copy(),
    )
    plot.close()


def test_signal() -> None:
    """Run unit tests on MONplot::draw_signal."""
    print("Run unit tests on MONplot::draw_signal")
    msm = get_test_data(error=0.1)
    # msm_ref = get_test_data(error=0.025)

    # image aspect=4
    plot = MONplot(
        "mon_plot_draw_signal-01.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(
        msm,
        zscale="linear",
        side_panels="none",
        title="method=linear; aspect=1; fig_pos=above",
    )
    plot.close()
    plot = MONplot(
        "mon_plot_draw_signal-02.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(
        msm, zscale="linear", title="method=linear; aspect=1; fig_pos=above"
    )
    plot.close()
    plot = MONplot(
        "mon_plot_draw_signal-03.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(msm, zscale="diff", title="method=diff; aspect=1; fig_pos=above")
    plot.close()
    msm1 = msm.copy()
    msm1.values = (msm1.values + 3) / 3
    msm1.attrs["units"] = "1"
    plot = MONplot(
        "mon_plot_draw_signal-04.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(
        msm1, zscale="ratio", title="method=ratio; aspect=1; fig_pos=above"
    )
    plot.close()
    plot = MONplot(
        "mon_plot_draw_signal-05.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(
        np.abs(msm),
        zscale="log",
        side_panels="none",
        title="method=error; aspect=1; fig_pos=above",
    )
    plot.close()

    # image aspect=2
    msm = get_test_data(data_sel=np.s_[500 - 250 : 500 + 250, :], error=0.1)
    plot = MONplot(
        "mon_plot_draw_signal-06.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(
        msm, side_panels="none", title="method=linear; aspect=2; fig_pos=above"
    )
    plot.close()
    plot = MONplot(
        "mon_plot_draw_signal-07.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(msm, title="method=linear; aspect=2; fig_pos=above")
    plot.close()

    # image aspect=3
    msm = get_test_data(
        data_sel=np.s_[500 - 125 : 500 + 125, 500 - 375 : 500 + 375], error=0.1
    )
    plot = MONplot(
        "mon_plot_draw_signal-08.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(
        msm, side_panels="none", title="method=linear; aspect=3; fig_pos=above"
    )
    plot.close()
    plot = MONplot(
        "mon_plot_draw_signal-09.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(msm, title="method=linear; aspect=3; fig_pos=above")
    plot.close()

    # image aspect=4
    msm = get_test_data(data_sel=np.s_[500 - 128 : 500 + 128, :], error=0.1)
    plot = MONplot(
        "mon_plot_draw_signal-10.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(
        msm, side_panels="none", title="method=linear; aspect=4; fig_pos=above"
    )
    plot.close()
    plot = MONplot(
        "mon_plot_draw_signal-11.png", caption="Unit test of MONplot [draw_signal]"
    )
    plot.set_institute("SRON")
    plot.draw_signal(msm, title="method=linear; aspect=4; fig_pos=above")
    plot.close()


def test_trend() -> None:
    """Run unit tests on MONplot::draw_trend."""
    print("Run unit tests on MONplot::draw_trend")
    n_elmnt = 200
    xval = np.arange(n_elmnt) / 100

    res = []
    hk_dtype = np.dtype([("mean", "f8"), ("err1", "f8"), ("err2", "f8")])
    buff = np.empty(len(xval), dtype=hk_dtype)
    data = 140.0 + (100 - np.arange(n_elmnt)) / 1000
    buff["mean"] = data
    buff["err1"] = data - 0.0125
    buff["err2"] = data + 0.0075
    hk_attrs = {"long_name": "SWIR detector temperature", "units": "K"}
    res.append(
        xr.DataArray(
            buff,
            name="detector_temp",
            attrs=hk_attrs,
            coords={"orbit": np.arange(n_elmnt)},
        )
    )

    buff = np.empty(len(xval), dtype=hk_dtype)
    data = 202.1 + (100 - np.arange(n_elmnt)) / 1000
    buff["mean"] = data
    buff["err1"] = data - 0.15
    buff["err2"] = data + 0.175
    hk_attrs = {"long_name": "SWIR grating temperature", "units": "K"}
    res.append(
        xr.DataArray(
            buff,
            name="grating_temp",
            attrs=hk_attrs,
            coords={"orbit": np.arange(n_elmnt)},
        )
    )

    buff = np.empty(len(xval), dtype=hk_dtype)
    data = 208.2 + (100 - np.arange(n_elmnt)) / 1000
    buff["mean"] = data
    buff["err1"] = data - 0.15
    buff["err2"] = data + 0.175
    hk_attrs = {"long_name": "SWIR OBM temperature", "units": "K"}
    res.append(
        xr.DataArray(
            buff, name="obm_temp", attrs=hk_attrs, coords={"orbit": np.arange(n_elmnt)}
        )
    )
    hk_ds = xr.merge(res, combine_attrs="drop_conflicts")

    msm1 = data_to_xr(np.sin(xval * np.pi), name="msm1", dims=["orbit"])
    msm2 = data_to_xr(np.cos(xval * np.pi), name="msm2", dims=["orbit"])
    msm_ds = xr.merge((msm1, msm2), combine_attrs="drop_conflicts")

    # plot.draw_trend(msm_ds, title='one dataset, no house-keeping')
    plot = MONplot(
        "mon_plot_draw_trend-1.png", caption="Unit test of MONplot [draw_trend]"
    )
    plot.set_institute("SRON")
    plot.draw_trend(xds=msm_ds, title="two datasets, no house-keeping")
    plot.close()
    plot = MONplot(
        "mon_plot_draw_trend-1.png", caption="Unit test of MONplot [draw_trend]"
    )
    plot.set_institute("SRON")
    plot.draw_trend(hk_xds=hk_ds, title="no datasets, only house-keeping")
    plot.close()
    plot = MONplot(
        "mon_plot_draw_trend-1.png", caption="Unit test of MONplot [draw_trend]"
    )
    plot.set_institute("SRON")
    plot.draw_trend(msm_ds, hk_ds, title="two datasets and house-keeping")
    plot.close()
