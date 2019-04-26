import numba
import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import sys

class LevenbergMarquardt:

    def __init__(self, n, f, jacobian, criterion=1.0, max_iters=100):
        self.criterion = criterion
        self.max_iters = max_iters
        self.eye = np.eye(n)
        self.jacobian = jacobian
        self.func = f
        self.up = 2.0
        self.down = 0.8
        self.lbd = 1.0

    def _get_iterate(self, x, fval):
        """Assume linear dynamics locally."""
        J = self.jacobian(x)
        mat = J.T @ J + self.lbd * self.eye
        x_iter = x - la.inv(mat) @ J.T @ fval
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
class LevenbergMarquardtJit(LevenbergMarquardt):

    def __init__(self, n, msmts, beacons, criterion, max_iters):
        self.criterion = criterion
        self.max_iters = max_iters
        self.msmts = msmts.astype(np.float32)
        self.beacons = beacons.astype(np.float32)
        self.eye = np.eye(n).astype(np.float32)
        self.up = 2.0
        self.down = 0.8
        self.lbd = 1.0

    def jacobian(self, x):
        return 2 * (x - self.beacons)

    def func(self, x):
        norm = np.sqrt(((x - self.beacons) ** 2).sum(-1))
        return norm - self.msmts

def make_circles(beacons, radii):
    kwargs = dict(fill=False, color='seagreen', alpha=0.4, linestyle='dashed')
    mapfunc = lambda tup: matplotlib.patches.Circle(*tup, **kwargs)
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

def make_lm(beacons, dist, sigma=1e-1, jitted=False):
    kwargs = dict(criterion=1.0, max_iters=1000)
    m, n = beacons.shape
    noise = np.random.normal(0, sigma, size=(m,))
    msmts = dist + noise
    if jitted:
        lm = LevenbergMarquardtJit(n, msmts, beacons, **kwargs)
    else:
        jacobian = lambda x: 2 * (x - beacons)
        func = lambda x: la.norm(x - beacons, axis=-1) - msmts
        lm = LevenbergMarquardt(n, func, jacobian, **kwargs)
    return lm

def make_problem():
    position = np.array([110, 70], np.float32)
    beacons = np.array([[130, 130], [90, 40], [70, 120]], np.float32)
    dist = la.norm(position[None, :] - beacons, axis=-1)
    return position, beacons, dist

def maybe_raise(exitcode):
    if exitcode != 0:
        raise ValueError('Stopping criterion not reached.')

def parse_args():
    try:
        jitted = 'jit' in sys.argv[1].lower()
    except IndexError:
        jitted = False
    return jitted

def main():
    jitted = parse_args()
    position, beacons, dist = make_problem()
    lm = make_lm(beacons, dist, jitted=jitted)
    x0 = np.array([130, 160], np.float32)
    exitcode, x, history = lm.solve(x0)
    maybe_raise(exitcode)
    plot_experiment(history, position, beacons, dist)

if __name__ == '__main__':
    main()
