import matplotlib as mpl
import matplotlib.pyplot as plt


def make_circles(beacons, radii):
    kwargs = dict(fill=False, color='seagreen', alpha=0.4, linestyle='dashed')
    mapfunc = lambda tup: mpl.patches.Circle(*tup, **kwargs)
    circles = map(mapfunc, zip(beacons, radii))
    return circles


def add_circles(beacons, radii, ax):
    circles = make_circles(beacons, radii)
    list(map(ax.add_patch, circles))


def set_plot_properties(fig, ax):
    fig.tight_layout()
    ax.set(xlim=[0, 200], ylim=[0, 200],
        aspect='equal', title='Range Localization')
    ax.legend()


def plot_experiment(history, position, beacons, radii):
    fig, ax = plt.subplots(figsize=(5, 5))
    add_circles(beacons, radii, ax)
    ax.scatter(*position, c='darkblue', s=50, marker='x', label='x_true')
    ax.scatter(*beacons.T, c='darkblue', s=50, marker='s', label='beacons')
    ax.scatter(*history[0], c='orange', s=20, marker='o')
    ax.plot(*history.T, c='orange', label='iterates')
    set_plot_properties(fig, ax)
    fig.savefig('../images/beacon.png', dpi=100, bbox_inches='tight')
    plt.show()
