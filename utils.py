from matplotlib import pyplot as plt
from matplotlib import *


def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.cm.tab10
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=10)

    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def segment_bars_with_confidence_score(save_path, confidence_score, labels=[]):
    num_pics = len(labels) + 1
    color_map = plt.cm.tab10

    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=10)
    fig = plt.figure(figsize=(15, num_pics * 1.5))

    interval = 1 / (num_pics+1)
    axes = []
    for i, label in enumerate(labels):
        i = i + 1
        axes.append(fig.add_axes([0, 1-i*interval, 1, interval]))
#         ax1.imshow([label], **barprops)
    for i, label in enumerate(labels):
        axes[i].imshow([label], **barprops)
    
    ax99 = fig.add_axes([0, 1- (i+2) * interval, 1, interval])
    ax99.set_xlim(0, len(confidence_score))
    ax99.set_ylim(0, 1)
    ax99.plot(range(len(confidence_score)), confidence_score)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()