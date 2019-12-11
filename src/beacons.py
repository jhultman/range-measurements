import numba
import numpy as np
import numpy.linalg as la
import sys

from plotting import plot_experiment


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


def make_lm(beacons, dist, sigma=1e-1, jitted=False):
    kwargs = dict(criterion=1.0, max_iters=1000)
    m, n = beacons.shape
    noise = np.random.normal(0, sigma, size=(m,))
    msmts = dist + noise
    lm = LevenbergMarquardt(n, msmts, beacons, **kwargs)
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
