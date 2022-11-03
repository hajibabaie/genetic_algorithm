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
                 number_of_population,
                 variables_shape,
                 crossover_percentage,
                 mutation_percentage,
                 selection_pressure,
                 selection_probs,
                 mutation_probs,
                 tournament_size):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_population = number_of_population
        self._variables_shape = variables_shape
        self._crossover_percentage = crossover_percentage
        self._number_of_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_population / 2))
        self._mutation_percentage = mutation_percentage
        self._number_of_mutation = int(np.ceil(self._mutation_percentage * self._number_population))
        self._selection_pressure = selection_pressure
        self._selection_probs = selection_probs
        self._population_main = [self._Individual() for _ in range(self._number_population)]
        self._population_crossover = None
        self._population_mutation = None
        self._population_probs = None
        self._best_solution = None
        self._best_costs = []
        self._worst_cost = None
        self._mutation_probs = mutation_probs
        self._tournament_size = tournament_size

    def _initialization(self):

        for i in range(self._number_population):

            self._population_main[i].position = np.random.permutation(self._variables_shape).reshape((1, self._variables_shape))
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

        selected_indices = np.random.choice(range(self._number_population), self._tournament_size, replace=False)

        selected_population_cost_argmin = np.argmin([self._population_main[i].cost for i in selected_indices])

        return int(selected_indices[selected_population_cost_argmin])

    def _selection(self):

        method = self._roulette_wheel_selection(self._selection_probs)

        if method == 0:

            return self._roulette_wheel_selection(self._population_probs)

        elif method == 1:

            return self._tournament_selection()

    def _crossover(self, first_ind, second_ind):

        first_position = self._population_main[first_ind].position
        second_position = self._population_main[second_ind].position

        cut_point = np.random.choice(range(int(first_position.shape[1])))

        first_position_1st = np.copy(first_position[:, :cut_point])
        first_position_2nd = np.copy(first_position[:, cut_point:])

        second_position_1st = np.copy(second_position[:, :cut_point])
        second_position_2nd = np.copy(second_position[:, cut_point:])

        a, b, c = np.intersect1d(first_position_1st, second_position_2nd, return_indices=True)
        x, y, z = np.intersect1d(second_position_1st, first_position_2nd, return_indices=True)

        first_position_1st[:, b] = x
        second_position_1st[:, y] = a

        offspring1 = np.concatenate((first_position_1st, second_position_2nd), axis=1)
        offspring2 = np.concatenate((second_position_1st, first_position_2nd), axis=1)

        return offspring1, offspring2

    @staticmethod
    def _swap(parent_position, first_cut, second_cut):


        offspring = np.concatenate((parent_position[:, :first_cut],
                                    parent_position[:, second_cut: second_cut + 1],
                                    parent_position[:, first_cut + 1: second_cut],
                                    parent_position[:, first_cut: first_cut + 1],
                                    parent_position[:, second_cut + 1:]), axis=1)

        return offspring

    @staticmethod
    def _reversion(parent_position, first_cut, second_cut):

        offspring = np.concatenate((parent_position[:, :first_cut],
                                    np.flip(parent_position[:, first_cut:second_cut+1]),
                                    parent_position[:, second_cut+1:]), axis=1)

        return offspring

    def _insertion(self, parent_position, first_cut, second_cut):

        method = self._roulette_wheel_selection([0.5, 0.5])

        if method == 0:

            offspring = np.concatenate((parent_position[:, :first_cut],
                                        parent_position[:, first_cut + 1: second_cut],
                                        parent_position[:, second_cut: second_cut + 1],
                                        parent_position[:, first_cut: first_cut + 1],
                                        parent_position[:, second_cut + 1:]), axis=1)

        else:

            offspring = np.concatenate((parent_position[:, :first_cut],
                                        parent_position[:, second_cut: second_cut + 1],
                                        parent_position[:, first_cut: second_cut],
                                        parent_position[:, second_cut + 1:]), axis=1)

        return offspring

    def _mutation(self, parent_ind):

        parent_position = self._population_main[parent_ind].position

        method = self._roulette_wheel_selection(self._mutation_probs)

        cut_points = np.random.choice(range(parent_position.shape[1]), 2, replace=False)
        cut1, cut2 = min(cut_points), max(cut_points)

        if method == 0:

            return self._swap(parent_position, cut1, cut2)

        elif method == 1:

            return self._reversion(parent_position, cut1, cut2)

        elif method == 2:

            return self._insertion(parent_position, cut1, cut2)



    def run(self):

        tic = time.time()

        self._initialization()

        self._population_main = self._sort_population(self._population_main)

        for iter_main in range(self._max_iteration):

            self._calc_probs()

            self._population_crossover = [self._Individual() for _ in range(self._number_of_crossover)]
            self._population_mutation = [self._Individual() for _ in range(self._number_of_mutation)]

            for iter_crossover in range(0, self._number_of_crossover, 2):

                first_index = self._selection()
                second_index = self._selection()
                while first_index == second_index:
                    second_index = self._selection()

                self._population_crossover[iter_crossover].position, \
                self._population_crossover[iter_crossover + 1].position = \
                self._crossover(first_index, second_index)

                self._population_crossover[iter_crossover].cost = \
                self._cost_function(self._population_crossover[iter_crossover].position)

                self._population_crossover[iter_crossover + 1].cost = \
                self._cost_function(self._population_crossover[iter_crossover + 1].position)


            for iter_mutation in range(self._number_of_mutation):

                parent_index = self._selection()

                self._population_mutation[iter_mutation].position = self._mutation(parent_index)

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
        plt.title("quadratic assignment problem using integer genetic algorithm".title(), fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("./cost_function.png")

        return self._best_solution, toc - tic
