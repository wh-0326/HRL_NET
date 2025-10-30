import random
import copy
import numpy as np
import time
from matplotlib import pyplot as plt
import torch
from matplotlib.patches import Circle

class Chromosome:
    """
    Class Chromosome represents one chromosome which consists of genetic code and value of
    fitness function.
    Genetic code represents potential solution to problem - the list of locations that are selected
    as medians.
    """

    def __init__(self, content, fitness):
        self.content = content
        self.fitness = fitness

    def __str__(self): return "%s f=%f" % (self.content, self.fitness)

    def __repr__(self): return "%s f=%f" % (self.content, self.fitness)


class GeneticAlgorithm:

    def __init__(self, n, m, p, cost_matrix, r, demand, w=0.8):
        """
        n: number of users
        m: number of candidate facilities
        p: number of facilities to select
        cost_matrix: a distance matrix of shape [m, n]
        r: coverage radius
        demand: demand for each user, shape [n]
        w: weight for the BCLP objective function
        """
        self.time = None
        self.user_num = n
        self.fac_num = m
        self.p = p
        self.r = r
        self.w = w  # BCLP 权重
        self.cost_matrix = cost_matrix
        self.demand = demand
        self.iterations = 200  # Maximal number of iterations
        self.current_iteration = 0
        self.generation_size = 50  # Number of individuals in one generation
        self.reproduction_size = 20  # Number of individuals for reproduction
        self.mutation_prob = 0.1  # Mutation probability
        self.top_chromosome = None  # Chromosome that represents solution of optimization process

    def mutation(self, chromosome):
        """
        Applies mutation over chromosome with probability self.mutation_prob
        In this process, a randomly selected median is replaced with a randomly selected demand point.
        """
        mp = random.random()
        if mp < self.mutation_prob:
            i = random.randint(0, len(chromosome) - 1)
            # Candidate facilities not in the current chromosome
            available_facilities = [fac for fac in range(self.fac_num) if fac not in chromosome]
            if not available_facilities:  # If all facilities are in the chromosome, skip mutation
                return chromosome
            # Replace selected median with a randomly selected new facility
            chromosome[i] = random.choice(available_facilities)
        return chromosome

    def crossover(self, parent1, parent2):
        identical_elements = [element for element in parent1 if element in parent2]
        if len(identical_elements) == len(parent1):
            return parent1, None
        exchange_vector_for_parent1 = [element for element in parent1 if element not in identical_elements]
        exchange_vector_for_parent2 = [element for element in parent2 if element not in identical_elements]
        c = random.randint(0, min(len(exchange_vector_for_parent1), len(exchange_vector_for_parent2)) - 1)
        for i in range(c):
            exchange_vector_for_parent1[i], exchange_vector_for_parent2[i] = exchange_vector_for_parent2[i], \
            exchange_vector_for_parent1[i]
        child1 = identical_elements + exchange_vector_for_parent1
        child2 = identical_elements + exchange_vector_for_parent2
        return child1, child2

    def fitness(self, chromosome):
        """
        Calculates the fitness of a given chromosome based on the BCLP objective.
        The function calculates Z = w * sum(a_i * y_i) + (1-w) * sum(a_i * u_i),
        and returns -Z, because the genetic algorithm is set to MINIMIZE the fitness value.
        - chromosome: A list of p selected facility indices.
        """
        # 1. Get distances for selected facilities (from chromosome) to all users.
        # self.cost_matrix shape: [n_facilities, n_users]
        selected_distances = self.cost_matrix[chromosome, :]

        # 2. Determine binary coverage matrix for the selected facilities.
        # coverage_matrix shape: [p, n_users]
        coverage_matrix = (selected_distances <= self.r).astype(int)

        # 3. Calculate coverage count for each user (how many facilities cover each user).
        # coverage_count shape: [n_users]
        coverage_count = np.sum(coverage_matrix, axis=0)

        # 4. Determine y_i (covered by >= 1 facility) and u_i (covered by >= 2 facilities).
        y = (coverage_count >= 1).astype(float)
        u = (coverage_count >= 2).astype(float)

        # 5. Calculate the two terms of the BCLP objective function.
        term1 = np.sum(self.demand * y)
        term2 = np.sum(self.demand * u)

        # 6. Combine terms using the weight w.
        objective_value = self.w * term1 + (1 - self.w) * term2

        # 7. Return the negative of the objective, as the GA minimizes the fitness value.
        return -objective_value

    def initial_random_population(self):
        """ Creates initial population """
        init_population = []
        for _ in range(self.generation_size):
            # Randomly select p unique facilities
            rand_medians = random.sample(range(self.fac_num), self.p)
            init_population.append(rand_medians)

        init_population = [Chromosome(content, self.fitness(content)) for content in init_population]
        self.top_chromosome = min(init_population, key=lambda chromo: chromo.fitness)
        print("Initial top solution: %s" % self.top_chromosome)
        return init_population

    def selection(self, chromosomes):
        """Ranking-based selection method"""
        chromosomes.sort(key=lambda x: x.fitness)
        L = self.reproduction_size
        selected_chromosomes = []
        for i in range(self.reproduction_size):
            j = L - np.floor((-1 + np.sqrt(1 + 4 * random.uniform(0, 1) * (L ** 2 + L))) / 2)
            selected_chromosomes.append(chromosomes[int(j)])
        return selected_chromosomes

    def create_generation(self, for_reproduction):
        """ Creates new generation """
        new_generation = []
        while len(new_generation) < self.generation_size:
            parents = random.sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0].content, parents[1].content)
            self.mutation(child1)
            new_generation.append(Chromosome(child1, self.fitness(child1)))
            if child2 is not None and len(new_generation) < self.generation_size:
                self.mutation(child2)
                new_generation.append(Chromosome(child2, self.fitness(child2)))
        return new_generation

    def optimize(self):
        start_time = time.time()
        chromosomes = self.initial_random_population()
        while self.current_iteration < self.iterations:
            for_reproduction = self.selection(chromosomes)
            chromosomes = self.create_generation(for_reproduction)
            self.current_iteration += 1
            chromosome_with_min_fitness = min(chromosomes, key=lambda chromo: chromo.fitness)
            if chromosome_with_min_fitness.fitness < self.top_chromosome.fitness:
                self.top_chromosome = chromosome_with_min_fitness
        end_time = time.time()
        self.time = end_time - start_time
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nFinal top solution: %s" % self.top_chromosome)
        print('Time: {:0>2}:{:0>2}:{:05.4f}'.format(int(hours), int(minutes), seconds))