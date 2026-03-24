from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_center, color_min, color_max = kwargs['color_range']
    else:
        color_center, color_min, color_max = np.mean(color), min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    is_perm = False
    if 'color_perm' in kwargs:
        color_perm = kwargs['color_perm']
        is_perm = True

    x_rot = kwargs.get('x_tick_rotation', 45)
    y_rot = kwargs.get('y_tick_rotation', 0)
    fontsize = kwargs.get('fontsize', 14)
    num_size_label = kwargs.get('num_size_label', 4)
    size_labels = kwargs.get('size_labels', [])
    facecolor = kwargs.get('facecolor', '#FFFFFF')

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val, size_scale = size_scale):
        if val==-1:
            return 0
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 16, hspace=1, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-2]) # Use the left 14/16ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'color_perm', 'palette', 'color_range', 'size', 'size_range', 'size_scale', \
         'marker', 'x_order', 'y_order', 'xlabel', 'ylabel', 'x_tick_rotation', \
         'y_tick_rotation','fontsize', 'num_size_label','m_color', 'size_labels', 'facecolor', \
    ]}


    
    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )

    if is_perm:
        ax.scatter(
            x=[x_to_num[v] for v in x],
            y=[y_to_num[v] for v in y],
            marker=marker,
            s=[value_to_size(v) for v in size], 
            # c=[value_to_color(v) for v in color_perm],
            facecolors="none",
            edgecolors=['k' if v!=0 else 'none' for v in color_perm],
            linewidths=3,
            **kwargs_pass_on
        )
        
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=x_rot, horizontalalignment='center', fontsize=fontsize)
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num], rotation=y_rot, fontsize = fontsize)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor(facecolor) #palette[len(palette)//2]) #'#F1F1F1')

    ax.set_xlabel(kwargs.get('xlabel', ''), fontsize=fontsize)
    ax.set_ylabel(kwargs.get('ylabel', ''), fontsize=fontsize)

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax1 = plt.subplot(plot_grid[:,-2]) # Use the rightmost-1 column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax1.barh(
            y=bar_y,
            width=[4]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0,
        )
        ax1.set_xlim(1, 2) # Bars are going from 0 to 5, so we crop the plot somewhere in the middle
        ax1.grid(False) # Hide grid
        ax1.set_facecolor('white') # Make background white
        ax1.set_xticks([]) # Remove horizontal ticks
        ax1.set_yticks(np.linspace(min(bar_y), max(bar_y), 5)) # Show vertical ticks for min, middle and max
        ax1.yaxis.tick_right() # Show vertical ticks on the right
        #ax1.set_axis_off()

    # Plot the size reference
    ax2 = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot 
    
    y_len = len(y_to_num.keys())
    size_labels = size_labels if size_labels else np.linspace(min(size), max(size), num_size_label).tolist()
    ss = size_labels + [-1]*(y_len-len(size_labels))
    ax2.scatter(
        x=[0]*y_len,
        y=np.arange(0,y_len),
        marker=marker,
        s=[value_to_size(v, size_scale=size_scale) for v in ss],
        c=kwargs.get('m_color','k'),
        **kwargs_pass_on
    )
    ax2.grid(False) # Hide grid
    ax2.set_facecolor('white')
    ax2.set_xticks([])
    ax2.set_yticks([v for k,v in y_to_num.items()][:len(size_labels)])
    ax2.yaxis.tick_right()

    return ax, ax1, ax2, size_labels


def corrplot(data, size_scale=500, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index').replace(np.nan, 0)
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

