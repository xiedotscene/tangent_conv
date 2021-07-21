import numpy as np
import matplotlib.pyplot as plt

def color_labels(results, label_names):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)

    num_classes = len(label_names)
    colors = [[255, 255, 255], [128, 128, 128], [255, 225, 25], [124, 152, 0], [170, 110, 40], [128, 0, 0], [245, 130, 48], [250, 190, 190], [0, 130, 200]]
    for i in range(len(colors)):
        for j in range(len(colors[i])):
            colors[i][j] = colors[i][j] / 255.0

    fig, ax = plt.subplots(figsize=(9.2, 1))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (label_name, color) in enumerate(zip(label_names, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=label_name, color=color)
        xcenters = starts + widths / 2

        r, g, b = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center', color=text_color)
    ax.legend(ncol=len(label_names), bbox_to_anchor=(0, 1), loc='lower left', fontsize='small')
    return fig, ax


if __name__=="__main__":
    num_classes = 9
    semantic_labels = np.ones(num_classes)

    label_names = ['unlabeled', 'man made terrain', 'natural terrain', 'high vegetation', 'low vegetation', 'building', 'hardscape',
                   'scanning artifacts', 'cars']
    results = {'labels': semantic_labels}

    color_labels(results, label_names)
    plt.show()
