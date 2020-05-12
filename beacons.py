import sys
import numba
import numpy as np
import numpy.linalg as la
import matplotlib as mpl
import matplotlib.pyplot as plt


spec = [
    ('criterion', numba.float32),
    ('max_iters', numba.int32),
    ('msmts', numba.float32[:]),
    ('beacons', numba.float32[:, :]),
    ('eye', numba.float32[:, :]),
    ('up', numba.float32),
    ('down', numba.float32),
    ('lbd', numba.float32),
]


@numba.jitclass(spec)
class LevenbergMarquardt:

    def __init__(self, n, msmts, beacons, criterion, max_iters):
        self.criterion = criterion
        self.max_iters = max_iters
        self.msmts = msmts.astype(np.float32)
        self.beacons = beacons.astype(np.float32)
        self.eye = np.eye(n).astype(np.float32)
        self.up, self.down, self.lbd = 2.0, 0.8, 1.0

    def jacobian(self, x):
        return 2 * (x - self.beacons)

    def func(self, x):
        norm = np.sqrt(((x - self.beacons) ** 2).sum(-1))
        return norm - self.msmts

    def _get_iterate(self, x, fval):
        """Assume linear dynamics locally."""
        J = self.jacobian(x)
        mat = J.T @ J + self.lbd * self.eye
        x_iter = x - np.ascontiguousarray(la.inv(mat)) @ J.T @ fval
        return x_iter

    def _compare(self, f0, f1, x, x_iter):
        """Accept iterate if better,
        else shrink trust region."""
        if f0 < f1:
            self.lbd *= self.up
            return x
        else:
            self.lbd *= self.down
            return x_iter

    def solve(self, x):
        """Solve non-linear least squares via LM."""
        history = [list(x)]
        for _ in range(self.max_iters):
            fval = self.func(x)
            x_iter = self._get_iterate(x, fval)
            fval_iter = self.func(x_iter)
            f0, f1 = la.norm(fval), la.norm(fval_iter)
            x = self._compare(f0, f1, x, x_iter)
            history += [list(x)]
            if f1 < self.criterion:
                exitcode = 0
                break
        else:
            exitcode = 1
        history = np.float32(history)
        return exitcode, x, history


def make_lm(beacons, distance, sigma=1e-1):
    kwargs = dict(criterion=1.0, max_iters=1000)
    m, n = beacons.shape
    noise = np.random.normal(0, sigma, size=(m,))
    msmts = distance + noise
    lm = LevenbergMarquardt(n, msmts, beacons, **kwargs)
    return lm


def make_problem():
    position = np.array([110, 70], np.float32)
    beacons = np.array([[130, 130], [90, 40], [70, 120]], np.float32)
    distance = la.norm(position[None, :] - beacons, axis=-1)
    return position, beacons, distance


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
    fig.savefig('./images/beacon.png', dpi=100, bbox_inches='tight')
    plt.show()


def main():
    position, beacons, distance = make_problem()
    lm = make_lm(beacons, distance)
    x0 = np.array([130, 160], np.float32)
    exitcode, x, history = lm.solve(x0)
    if exitcode != 0:
        print('Stopping criterion not reached.')
    plot_experiment(history, position, beacons, distance)


if __name__ == '__main__':
    main()
