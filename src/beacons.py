import functools
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.patches
np.random.seed(21)

EPS = 1e-9


class ParticleGroup:
    
    def __init__(self, beacons, n_particles, gridsize, sigma_est=1, sigma_perturb=1):
        self.beacon_locs = beacons
        self.particle_locs = gridsize * np.random.rand(n_particles, 2)
        self.loglikelihood = np.full((n_particles,), 0.)
        self.pdf = functools.partial(scipy.stats.norm.pdf, loc=0.0, scale=sigma_est)
        self.noise = functools.partial(np.random.normal, size=(n_particles, 2), loc=0.0, scale=sigma_perturb)
        
    def update_weights(self, msmt):
        diff = self.beacon_locs[None, :] - self.particle_locs[:, None]
        distance = np.linalg.norm(diff, axis=2)
        likelihood = self.pdf(msmt - distance) + EPS
        self.loglikelihood += np.log(likelihood).sum(-1)
    
    def resample(self):
        n = self.loglikelihood.shape[0]
        p = np.exp(self.loglikelihood)
        p /= p.sum()
        inds = np.random.choice(n, n, p=p, replace=True)
        self.particle_locs = self.particle_locs[inds] + self.noise()
        self.loglikelihood = np.zeros_like(self.loglikelihood)
    
    def forward(self, msmt):
        self.update_weights(msmt)
        self.resample()


class Simulator:
    
    def __init__(self, beacons, position, sigma):
        self.beacons = beacons
        self.position = position
        self.sigma = sigma
        self.noise = functools.partial(np.random.normal, loc=0, scale=sigma, size=beacons.shape[0])
        
    def get_msmt(self):
        diff = self.beacons - self.position
        distance = np.linalg.norm(diff, axis=1)
        msmt = distance + self.noise()
        return msmt


class Experiment:
    
    def __init__(self):
        self.init_params()
        self.init_position()
        self.init_beacons()
        self.init_pg()
        self.init_sim()
    
    def init_params(self):
        self.n_particles = 1000
        self.sigma_true = 5
        self.sigma_est = 7
        self.sigma_perturb = 1
        self.gridsize = 200
        self.n_trials = 4
    
    def init_position(self):
        self.position = np.array([100, 100])

    def init_beacons(self):
        self.beacons = np.array([
            [120, 140],
            [100, 30],
            [70, 120],
        ])
        self.dist_to_beacons = np.linalg.norm(
            self.beacons - self.position, axis=1,
        )
       
    def init_pg(self):
        self.pg = ParticleGroup(
            self.beacons, 
            self.n_particles, 
            self.gridsize, 
            self.sigma_est, 
            self.sigma_perturb,
        )
    
    def init_sim(self):
        self.sim = Simulator(
            self.beacons, 
            self.position, 
            self.sigma_true,
        )
    
    def make_circles(self):
        kwargs = dict(fill=False, color='seagreen', alpha=0.6, linestyle='dashed')
        mapfunc = lambda tup: matplotlib.patches.Circle(*tup, **kwargs)
        circles = map(mapfunc, zip(self.beacons, self.dist_to_beacons))     
        return circles
    
    def make_size(self, likelihood):
        k = int(0.80 * self.n_particles)
        top = np.argpartition(likelihood, kth=k)[k:]
        size = np.full((self.n_particles,), 5)
        size[top] = 40    
        return size
    
    def single_scatter(self, ax, locations, weights):
        size = self.make_size(weights)
        ax.scatter(*locations.T, c='grey', s=size, alpha=0.2)
        ax.scatter(*self.position, c='red', s=50, alpha=0.6, marker='o')
        ax.scatter(*self.beacons.T, c='darkblue', s=50, marker='s')
        circles = self.make_circles()
        for circle in circles:
            ax.add_patch(circle)
    
    def plot_experiment(self, locations, weights):
        ncols = max(1, int(np.sqrt(self.n_trials)))
        nrows = self.n_trials - ncols
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * nrows, 3 * nrows))
        for i, (a, l, w) in enumerate(zip(ax.ravel(), locations, weights)):
            self.single_scatter(a, l, w)
            a.set(xlim=[0, self.gridsize], ylim=[0, self.gridsize], aspect='equal', title=f'iter {i}')
        fig.tight_layout()
        fig.savefig('../images/beacon.png', dpi=300, bbox_inches='tight')

    def _run(self):
        loc_history = []
        weight_history = []
        for _ in range(self.n_trials):
            loc_history += [self.pg.particle_locs]
            weight_history += [np.exp(self.pg.loglikelihood)]
            msmt = self.sim.get_msmt()
            self.pg.forward(msmt)
        return loc_history, weight_history
    
    def run(self):
        locations, weights = self._run()
        self.plot_experiment(locations, weights)


def main():
    experiment = Experiment()
    experiment.run()


if __name__ == '__main__':
    main()
