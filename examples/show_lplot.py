import numpy as np

from moniplot.mon_plot import MONplot


plot = MONplot('test_lplot.pdf')
plot.set_cset(None)
for ii in range(5):
    plot.draw_lplot(np.arange(10), np.arange(10)*(ii+1),
                    label=f'label {ii}')
plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
                title='draw_lplot [cset is None]')

for ii, clr in enumerate('rgbym'):
    plot.draw_lplot(np.arange(10), np.arange(10)*(ii+1), color=clr,
                    label=f'label {ii}')
plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
                title='draw_lplot [cset="rgbym"]')

plot.set_cset('mute')   # Note the default is 'bright'
for ii in range(5):
    plot.draw_lplot(ydata=np.arange(10)*(ii+1),
                    label=f'label {ii}')
plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
                title='draw_lplot [cset="mute"]')

plot.set_cset('rainbow_PuBr', 35)
for ii in range(35):
    plot.draw_lplot(ydata=np.arange(10)*(ii+1),
                    label=f'label {ii}')
plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
                title='draw_lplot [cset="rainbow_PyBr"]')

for ii in range(35):
    plot.draw_lplot(ydata=np.arange(10)*(ii+1),
                    label=f'label {ii}')
plot.draw_lplot(xlabel='x-axis', ylabel='y-axis',
                title='draw_lplot [cset="rainbow_PyBr"]',
                kwlegend={'fontsize': 'x-small', 'loc': 'upper left',
                          'bbox_to_anchor': (0.975, 1)})
plot.close()
