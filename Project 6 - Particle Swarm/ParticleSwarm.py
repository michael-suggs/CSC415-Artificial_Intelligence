import math, random
import numpy as np

__author__ = "Michael J. Suggs // mjs3607@uncw.edu"


class ParticleSwarm:

    def __init__(self, pop_size, iter_lim, city_dict):
        self.pop_size = pop_size
        self.iter_lim = iter_lim
        self.city_dict = city_dict
        self.pop = {}

        self.gbest = None

    def gen_population(self):
        city_list = list(self.city_dict.keys())
        for i in range(self.pop_size):
            # generate particle's initial position with uniformly distributed random vector
            # generate particle's best known position to its initial position
            # update swarm's best known position
            # generate the particle's velocity
            random.shuffle(city_list)
            part = Particle(''.join(city_list), grid_max, grid_min)


    def swarm(self):
        for particle in self.population:
            # for each dimension:
                # pick random numbers
                # update particle velocity
                # update particle position
                # if f(x) < f(p)
                    # update particle's best known position
                    # if f(p) < f(g)
                        # update swarm's best known position
            pass

    def distance(self, individual):
        total_dist = 0
        for i in range(len(individual)):
            total_dist += math.sqrt(
                ((self.city_dict[individual[(i + 1) % len(individual)]][0] -
                  self.city_dict[individual[i]][0]) ** 2) +
                ((self.city_dict[individual[(i + 1) % len(individual)]][1] -
                  self.city_dict[individual[i]][1]) ** 2))
        return total_dist

    def evaluate(self, individual):
        return 1 / self.distance(individual)


class Particle:

    def __init__(self, soln, grid_max, grid_min):
        self.soln = soln
        self.position = np.zeros(1, 2)
        self.velocity = np.zeros(1, 2)
        self.pbest = np.zeros(1, 2)
        self.gbest = np.zeros(1, 2)

    def init_pos(self, grid_max, grid_min):
        for i in self.position.size[1]:
            self.position[i] = (grid_max - grid_min) *

    def update_velocity(self, gbest, vel_max, c1, c2):
        self.velocity = vel_max

    def update_position(self, bounds):
        pass

    def update_best_position(self, best):
        pass
