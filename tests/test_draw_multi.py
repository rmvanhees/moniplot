import matplotlib.pyplot as plt
import numpy as np

from moniplot.mon_plot import MONplot
from moniplot.draw_multi import DrawMulti


def main(n_panel: int = 1) -> None:
    """..."""

    mu, sigma = 0, 0.1 # mean and standard deviation
    
    with DrawMulti(n_panel) as plot:
        for ii in range(n_panel):
            plot.hist(
                ii,
                np.random.normal(mu, sigma, 1000),
                clip=[-0.4, 0.4],
                bins=40,
            )
            plot.set_title(f"Figure {ii+1}")
    plt.show()

    with DrawMulti(n_panel, sharey=True) as plot:
        for ii in range(n_panel):
            plot.hist(
                ii,
                np.random.normal(mu, sigma, 1000),
                clip=[-0.4, 0.4],
                bins=40,
            )
            plot.set_title(f"Figure {ii+1}", ipanel=ii)
    plt.show()

    with DrawMulti(n_panel, sharex=True, sharey=True) as plot:
        for ii in range(n_panel):
            plot.hist(
                ii,
                np.random.normal(mu, sigma, 1000),
                clip=[-0.4, 0.4],
                bins=40,
            )
            plot.set_title(f"Figure {ii+1}", ipanel=ii)
        plot.set_xlim(-0.425, 0.425)
    plt.show()

    report = MONplot("multi_line.pdf")
    with DrawMulti(n_panel, sharex=True, sharey=True) as plot:
        xdata = np.arange(128)
        ydata = np.sin(xdata / np.pi)
        for ii in range(n_panel):
            plot.add_line(ii, xdata, ydata, marker="+", ls="--")
            plot.add_line(ii, xdata + 1.5, ydata, marker="x", ls="--")
            plot.add_line(ii, xdata - 1.5, ydata, marker="o", ls="--")
            plot.set_xlabel("time", ipanel=ii)
            plot.set_ylabel("value", ipanel=ii)
            plot.draw(ii)
        #plot.set_xlim(-0.5, 0.5)
        #plot.set_ylim(0, 120)
        plot.add_caption(
            f"This is a Figure of {n_panel} line-plots"
        )
        plot.add_copyright(-1)
        # fig_info.draw(plot.fig)
    report.close_this_page(plt.figure)
    report.close()

if __name__ == "__main__":
    main(5)
