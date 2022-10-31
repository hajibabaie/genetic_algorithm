import numpy as np
import matplotlib.pyplot as plt
import time


class GA:

    class _Individual:

        def __init__(self):

            self.position = None
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
        self._num_population = number_population
        self._num_variables = number_variables
        self._crossover_percentage = crossover_percentage
        self._num_crossover = 2 * int(np.ceil(self._crossover_percentage * self._num_population / 2))
        self._mutation_percentage = mutation_percentage
        self._num_mutation = int(np.ceil(self._mutation_percentage * self._num_population))
        self._mutation_rate = mutation_rate
        self._selection_pressure = selection_pressure
        self._tournament_size = tournament_size
        self._selection_probs = selection_probs
        self._crossover_probs = crossover_probs
        self._population_main = [self._Individual() for _ in range(self._num_population)]
        self._population_crossover = None
        self._population_mutation = None
        self._population_probs = None
        self._best_costs = []
        self._best_solution = None
        self._worst_cost = 1 * self._num_variables

    def _initialize_population(self):

        for i in range(self._num_population):

            self._population_main[i].position = np.random.randint(0, 2, (1, self._num_variables))
            self._population_main[i].cost = self._cost_function(self._population_main[i].position)

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

        out = np.argwhere(random_number <= probs_cumsum)[0][0]

        return int(out)

    def _tournament_selection(self):

        selected_indices = np.random.choice(range(self._num_population), self._tournament_size, replace=False)

        selected_population_cost_argmin = np.argmin([self._population_main[i].cost for i in selected_indices])

        out = selected_indices[selected_population_cost_argmin]

        return int(out)

    def _selection(self):

        method = self._roulette_wheel_selection(self._selection_probs)

        if method == 0:

            return self._roulette_wheel_selection(self._population_probs)

        elif method == 1:

            return self._tournament_selection()

    def _single_point_crossover(self, first_parent_pos, second_parent_pos):

        cut_point = np.random.randint(1, self._num_variables)

        offspring_first = np.concatenate((first_parent_pos[:, :cut_point], second_parent_pos[:, cut_point:]), axis=1)
        offspring_second = np.concatenate((second_parent_pos[:, :cut_point], first_parent_pos[:, cut_point:]), axis=1)

        return offspring_first, offspring_second

    def _double_point_crossover(self, first_parent_pos, second_parent_pos):

        cut_points = np.random.choice(range(1, self._num_variables), 2, replace=False)
        first_cut, second_cut = min(cut_points), max(cut_points)

        offspring_first = np.concatenate((first_parent_pos[:, :first_cut],
                                          second_parent_pos[:, first_cut:second_cut],
                                          first_parent_pos[:, second_cut:]), axis=1)

        offspring_second = np.concatenate((second_parent_pos[:, :first_cut],
                                           first_parent_pos[:, first_cut:second_cut],
                                           second_parent_pos[:, second_cut:]), axis=1)

        return offspring_first, offspring_second

    @staticmethod
    def _uniform_crossover(first_parent_pos, second_parent_pos):

        alpha = np.random.randint(0, 2, first_parent_pos.shape)

        offspring_first = np.multiply(first_parent_pos, alpha) + np.multiply(second_parent_pos, 1 - alpha)
        offspring_second = np.multiply(second_parent_pos, alpha) + np.multiply(first_parent_pos, 1 - alpha)

        return offspring_first, offspring_second

    def _crossover(self, first_parent_ind, second_parent_ind):

        first_parent_position = self._population_main[first_parent_ind].position
        second_parent_position = self._population_main[second_parent_ind].position

        method = self._roulette_wheel_selection(self._crossover_probs)

        if method == 0:

            return self._single_point_crossover(first_parent_position, second_parent_position)

        elif method == 1:

            return self._double_point_crossover(first_parent_position, second_parent_position)

        elif method == 2:

            return self._uniform_crossover(first_parent_position, second_parent_position)

    def _mutation(self, parent_ind):

        parent_position = self._population_main[parent_ind].position

        num_mutation = int(np.ceil(self._mutation_rate * self._num_variables))

        mutate_ind = np.random.choice(range(self._num_variables), num_mutation, replace=False)

        offspring = np.copy(parent_position)

        offspring[:, mutate_ind] = 1 - parent_position[:, mutate_ind]

        return offspring

    def run(self):

        tic = time.time()

        self._initialize_population()

        self._population_main = self._sort_population(self._population_main)

        for iter_main in range(self._max_iteration):

            self._calc_probs()

            self._population_crossover = [self._Individual() for _ in range(self._num_crossover)]
            self._population_mutation = [self._Individual() for _ in range(self._num_mutation)]

            for iter_crossover in range(0, self._num_crossover, 2):

                first_parent_index = self._selection()
                second_parent_index = self._selection()
                while first_parent_index == second_parent_index:
                    second_parent_index = self._selection()

                self._population_crossover[iter_crossover].position, \
                self._population_crossover[iter_crossover + 1].position = self._crossover(first_parent_index,
                                                                                          second_parent_index)

                self._population_crossover[iter_crossover].cost = \
                self._cost_function(self._population_crossover[iter_crossover].position)

                self._population_crossover[iter_crossover + 1].cost = \
                self._cost_function(self._population_crossover[iter_crossover + 1].position)

            for iter_mutation in range(self._num_mutation):

                parent_index = self._selection()

                self._population_mutation[iter_mutation].position = self._mutation(parent_index)

                self._population_mutation[iter_mutation].cost = \
                self._cost_function(self._population_mutation[iter_mutation].position)


            self._population_main.extend(self._population_crossover)
            self._population_main.extend(self._population_mutation)

            self._population_main = self._sort_population(self._population_main)
            self._population_main = self._population_main[:self._num_population]
            self._best_costs.append(self._best_solution.cost)

        toc = time.time()

        plt.figure(dpi=600, figsize=(10, 6))
        plt.plot(range(self._max_iteration), self._best_costs)
        plt.title("Min One Problem Using Binary Genetic Algorithm", fontweight="bold")
        plt.xlabel("Number of Iteration")
        plt.ylabel("Cost")
        plt.savefig("./cost_function.png")

        return self._best_solution, toc - tic
