import numpy as np
import matplotlib.pyplot as plt
import time


class GA:

    class _Individual:

        def __init__(self):

            self.position = None
            self.solution = None
            self.cost = None

    def __init__(self,
                 cost_function,
                 max_iteration,
                 number_population,
                 number_variables,
                 crossover_percentage,
                 mutation_percentage,
                 mutation_rate,
                 selection_pressure,
                 tournament_size,
                 selection_probs,
                 crossover_probs):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_population = number_population
        self._number_variables = number_variables
        self._crossover_percentage = crossover_percentage
        self._mutation_percentage = mutation_percentage
        self._mutation_rate = mutation_rate
        self._selection_pressure = selection_pressure
        self._tournament_size = tournament_size
        self._selection_probs = selection_probs
        self._crossover_probs = crossover_probs
        self._best_costs = []
        self._best_solution = None
        self._worst_cost = None
        self._population_main = [self._Individual() for _ in range(self._number_population)]
        self._number_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_population / 2))
        self._number_mutation = int(np.ceil(self._mutation_percentage * self._number_population))
        self._population_crossover = None
        self._population_mutation = None
        self._population_probs = None

    def _initialization(self):

        for i in range(self._number_population):

            self._population_main[i].position = np.random.randint(0, 2, (1, self._number_variables))
            self._population_main[i].solution, \
            self._population_main[i].cost = self._cost_function(self._population_main[i].position)

        self._worst_cost = np.max([pop.cost for pop in self._population_main])

    def _sort_population(self, population):

        population_cost_argsort = np.argsort([pop.cost for pop in population])

        population_sorted = [population[i] for i in population_cost_argsort]

        self._best_solution = population_sorted[0]
        self._worst_cost = np.max([self._worst_cost, population_sorted[-1].cost])

        return population_sorted

    def _calc_probs(self):

        population_cost = [pop.cost for pop in self._population_main]
        population_cost_exp = [np.exp(-self._selection_pressure * i / self._worst_cost) for i in population_cost]
        population_cost_exp_sum = np.sum(population_cost_exp)
        self._population_probs = np.divide(population_cost_exp, population_cost_exp_sum)

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.random()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number <= probs_cumsum)[0][0])

    def _tournament_selection(self):

        selected_index = np.random.choice(range(self._number_population), self._tournament_size, replace=False)

        population_cost_argmin = np.argmin([self._population_main[i].cost for i in selected_index])

        return int(selected_index[population_cost_argmin])

    def _selection(self):

        method = self._roulette_wheel_selection(self._selection_probs)

        if method == 0:

            return self._roulette_wheel_selection(self._population_probs)

        elif method == 1:

            return self._tournament_selection()

    def _single_point_crossover(self, first_position, second_position):

        cut_point = np.random.choice(range(1, self._number_variables))

        offspring_first = np.concatenate((first_position[:, :cut_point], second_position[:, cut_point:]), axis=1)
        offspring_second = np.concatenate((second_position[:, :cut_point], first_position[:, cut_point:]), axis=1)

        return offspring_first, offspring_second

    def _double_point_crossover(self, first_position, second_position):

        cut_points = np.random.choice(range(1, self._number_variables), 2, replace=False)

        cut_first, cut_second = min(cut_points), max(cut_points)

        offspring_first = np.concatenate((first_position[:, :cut_first],
                                          second_position[:, cut_first: cut_second],
                                          first_position[:, cut_second:]), axis=1)

        offspring_second = np.concatenate((second_position[:, :cut_first],
                                           first_position[:, cut_first: cut_second],
                                           second_position[:, cut_second:]), axis=1)

        return offspring_first, offspring_second

    @staticmethod
    def _uniform_crossover(first_position, second_position):

        alpha = np.random.randint(0, 2, first_position.shape)

        offspring_first = np.multiply(first_position, alpha) + np.multiply(second_position, 1 - alpha)
        offspring_second = np.multiply(second_position, alpha) + np.multiply(first_position, 1 - alpha)

        return offspring_first, offspring_second

    def _crossover(self, first_ind, second_ind):

        first_pos = self._population_main[first_ind].position
        second_pos = self._population_main[second_ind].position

        method = self._roulette_wheel_selection(self._crossover_probs)

        if method == 0:

            return self._single_point_crossover(first_pos, second_pos)

        elif method == 1:

            return self._double_point_crossover(first_pos, second_pos)

        elif method == 2:

            return self._uniform_crossover(first_pos, second_pos)

    def _mutation(self, ind):

        parent_pos = self._population_main[ind].position

        offspring = np.copy(parent_pos)

        num_mutation = int(np.ceil(self._mutation_rate * self._number_variables))

        selected_indices = np.random.choice(range(self._number_variables), num_mutation, replace=False)

        offspring[:, selected_indices] = 1 - offspring[:, selected_indices]

        return offspring

    def run(self):

        tic = time.time()

        self._initialization()

        self._population_main = self._sort_population(self._population_main)

        for iter_main in range(self._max_iteration):

            self._calc_probs()

            self._population_crossover = [self._Individual() for _ in range(self._number_crossover)]
            self._population_mutation = [self._Individual() for _ in range(self._number_mutation)]

            for iter_crossover in range(0, self._number_crossover, 2):

                first_parent_index = self._selection()
                second_parent_index = self._selection()
                while first_parent_index == second_parent_index:
                    second_parent_index = self._selection()

                self._population_crossover[iter_crossover].position, \
                self._population_crossover[iter_crossover + 1].position = \
                self._crossover(first_parent_index, second_parent_index)

                self._population_crossover[iter_crossover].solution, \
                self._population_crossover[iter_crossover].cost = \
                self._cost_function(self._population_crossover[iter_crossover].position)

                self._population_crossover[iter_crossover + 1].solution, \
                self._population_crossover[iter_crossover + 1].cost = \
                self._cost_function(self._population_crossover[iter_crossover + 1].position)


            for iter_mutation in range(self._number_mutation):

                parent_index = self._selection()

                self._population_mutation[iter_mutation].position = self._mutation(parent_index)
                self._population_mutation[iter_mutation].solution, \
                self._population_mutation[iter_mutation].cost = \
                self._cost_function(self._population_mutation[iter_mutation].position)

            self._population_main.extend(self._population_crossover)
            self._population_main.extend(self._population_mutation)

            self._population_main = self._sort_population(self._population_main)
            self._population_main = self._population_main[:self._number_population]
            self._best_costs.append(self._best_solution.cost)
        toc = time.time()

        plt.figure(dpi=600, figsize=(10, 6))
        plt.plot(range(self._max_iteration), self._best_costs)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("./cost_function.png")

        return self._best_solution, toc - tic
