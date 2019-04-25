import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt

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
        history = []
        for _ in range(self.max_iters):
            history += [x]
            fval = self.func(x)
            x_iter = self._get_iterate(x, fval)
            fval_iter = self.func(x_iter)
            f0, f1 = map(la.norm, (fval, fval_iter))
            x = self._compare(f0, f1, x, x_iter)
            if f1 < self.criterion:
                history = np.array(history)
                return x, history
        msg = f'Stopping crit {self.criterion} not \
            reached in {self.max_iters} iters.'
        raise ValueError(msg)

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

def make_lm(position, beacons, dist, sigma=1e-1):
    m, n = beacons.shape
    noise = np.random.normal(0, sigma, size=(m,))
    msmts = dist + noise
    jacobian = lambda x: 2 * (x - beacons)
    func = lambda x: la.norm(x - beacons, axis=-1) - msmts
    lm = LevenbergMarquardt(n, func, jacobian, criterion=1, max_iters=1000)
    return lm

def main():
    position = np.array([110, 70])
    beacons = np.array([[130, 130], [90, 40], [70, 120]])
    dist = la.norm(position[None, :] - beacons, axis=-1)
    lm = make_lm(position, beacons, dist)
    x0 = np.array([130, 160])
    x, history = lm.solve(x0)
    plot_experiment(history, position, beacons, dist)

if __name__ == '__main__':
    main()
