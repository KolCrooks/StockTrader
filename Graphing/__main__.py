import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style, collections

import numpy as np

import csv

FILE = "NCLH_04-20-2020.csv"

#Graph settings
style.use('fivethirtyeight')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw
    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

if __name__ == "__main__":
    
    with open(FILE) as csv_file:
        reader = csv.reader(csv_file)
        ax.clear()
        lines = []
        colors = []

        n_gain = 0
        n_loss = 0
        m_gain = 0
        m_loss = 0
        hasStock = 0
        
        cnt = 0
        last = (-1, -1)
        for row in reader:
            if cnt == 0:
                cnt += 1
                continue
            elif cnt == 1:
                last = (cnt, float(row[3]))
                cnt += 1
                continue

            new = (cnt, float(row[3]))
            lines.append([last, new])

            if float(row[1]) > 0:
                colors.append((0, 0, 1, 1))
                if last[1] > new[1]:
                    n_loss += 1
                else:
                    n_gain += 1
                hasStock += 1
            else:
                colors.append((1, 0, 0, 1))
                if last[1] > new[1]:
                    m_loss += 1
                else:
                    m_gain += 1


            last = new
            cnt += 1

        c = np.array(colors)

        lc = collections.LineCollection(lines, colors=c, linewidths=2)
        ax.add_collection(lc)
        ax.autoscale()
        print(f"Loss cnt: {n_loss} \nGain cnt: {n_gain} \nPercent is Gain: {n_gain / (n_gain + n_loss)}")
        print(f"mLoss cnt: {m_loss} \nmGain cnt: {m_gain} \nPercent missed Gain: {m_gain / (m_gain + m_loss)}")
        print(f"Has Stock {hasStock/cnt}% of the time")
        scale = 1.5
        f = zoom_factory(ax, base_scale = scale)
        plt.show()
