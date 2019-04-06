import functools
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches
np.random.seed(21)

EPS = 1e-9


def get_residuals(x, msmt, beacons):
    diff = ((x - beacons) ** 2).sum(-1)
    residuals = diff - msmt
    return residuals


class Experiment:
    
    def __init__(self):
        self.init_locations()
        self.init_params()

    def init_locations(self):
        self.position = np.array([110, 70])
        self.beacons = np.array([
            [100, 140],
            [90, 30],
            [70, 120],
        ])
        self.dist_to_beacons = np.linalg.norm(
            self.beacons - self.position, axis=1,
        )
       
    def init_params(self):
        sigma_sensor = 2
        self.noise = functools.partial(
            np.random.normal, 
            loc=0,
            scale=sigma_sensor,
        )
   
    def make_circles(self):
        kwargs = dict(fill=False, color='seagreen', alpha=0.6, linestyle='dashed')
        mapfunc = lambda tup: matplotlib.patches.Circle(*tup, **kwargs)
        circles = map(mapfunc, zip(self.beacons, self.dist_to_beacons))     
        return circles
    
    def plot_experiment(self, pred_location):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set(xlim=[0, 200], ylim=[0, 200], aspect='equal', title='Localization')
        ax.scatter(*pred_location, c='purple', s=50, alpha=0.5, marker='x')
        ax.scatter(*self.position, c='red', s=50, alpha=0.5, marker='o')
        ax.scatter(*self.beacons.T, c='darkblue', s=50, marker='s')
        circles = self.make_circles()
        for circle in circles:
            ax.add_patch(circle)
        fig.tight_layout()
        fig.savefig('../images/beacon.png', dpi=100, bbox_inches='tight')

    def get_msmt(self):
        msmt = self.dist_to_beacons + self.noise()
        return msmt
    
    def run(self):
        msmt = self.get_msmt()
        residual_func = functools.partial(
            get_residuals,
            msmt=msmt,
            beacons=self.beacons,
        )
        x0 = np.array([50, 150])
        result = scipy.optimize.least_squares(residual_func, x0)
        self.plot_experiment(result.x)


def main():
    experiment = Experiment()
    experiment.run()


if __name__ == '__main__':
    main()
