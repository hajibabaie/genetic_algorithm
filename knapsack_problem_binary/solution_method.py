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
                 number_of_population,
                 number_of_variables,
                 crossover_percentage,
                 crossover_probs,
                 mutation_percentage,
                 mutation_rate,
                 selection_pressure,
                 selection_probs,
                 tournament_size):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_of_population = number_of_population
        self._number_of_variables = number_of_variables
        self._crossover_percentage = crossover_percentage
        self._number_of_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_of_population / 2))
        self._mutation_percentage = mutation_percentage
        self._mutation_rate = mutation_rate
        self._number_of_mutation = int(np.ceil(self._mutation_percentage * self._number_of_population))
        self._number_variables_to_mutate = int(np.ceil(self._mutation_rate * self._number_of_variables))
        self._crossover_probs = crossover_probs
        self._selection_pressure = selection_pressure
        self._selection_probs = selection_probs
        self._tournament_size = tournament_size
        self._population_main = [self._Individual() for _ in range(self._number_of_population)]
        self._population_crossover = None
        self._population_mutation = None
        self._population_probs = None
        self._best_solution = None
        self._best_costs = []
        self._worst_cost = None

    def _initialization(self):

        for i in range(self._number_of_population):

            self._population_main[i].position = np.random.randint(0, 2, (1, self._number_of_variables))
            self._population_main[i].solution, self._population_main[i].cost = \
            self._cost_function(self._population_main[i].position)

        self._worst_cost = np.max([pop.cost for pop in self._population_main])

    def _sort_population(self, population):

        population_cost_argsort = np.argsort([pop.cost for pop in population])
        population_sorted = [population[i] for i in population_cost_argsort]
        self._best_solution = population_sorted[0]
        self._worst_cost = np.max([population[-1].cost, self._worst_cost])

        return population_sorted

    def _calc_probs(self):

        population_costs = [pop.cost for pop in self._population_main]
        population_costs_exp = [np.exp(-self._selection_pressure * i / self._worst_cost) for i in population_costs]
        population_costs_exp_sum = np.sum(population_costs_exp)
        self._population_probs = np.divide(population_costs_exp, population_costs_exp_sum)

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.random()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number <= probs_cumsum)[0][0])

    def _tournament_selection(self):

        selected_indices = np.random.choice(range(self._number_of_population), self._tournament_size, replace=False)

        selected_pop_costs_argmin = np.argmin([self._population_main[i].cost for i in selected_indices])

        return int(selected_indices[selected_pop_costs_argmin])

    def _selection(self):

        method = self._roulette_wheel_selection(self._selection_probs)

        if method == 0:

            return self._roulette_wheel_selection(self._population_probs)

        else:

            return self._tournament_selection()

    @staticmethod
    def _single_point_crossover(first_position, second_position):

        cut_point = np.random.choice(range(int(first_position.shape[1])))

        offspring1 = np.concatenate((first_position[:, :cut_point],
                                     second_position[:, cut_point:]), axis=1)

        offspring2 = np.concatenate((second_position[:, :cut_point],
                                     first_position[:, cut_point:]), axis=1)

        return offspring1, offspring2

    @staticmethod
    def _double_point_crossover(first_position, second_position):

        cut_points = np.random.choice(range(int(first_position.shape[1])), 2, replace=False)
        cut1, cut2 = min(cut_points), max(cut_points)

        offspring1 = np.concatenate((first_position[:, :cut1],
                                     second_position[:, cut1:cut2],
                                     first_position[:, cut2:]), axis=1)

        offspring2 = np.concatenate((second_position[:, :cut1],
                                     first_position[:, cut1:cut2],
                                     second_position[:, cut2:]), axis=1)

        return offspring1, offspring2

    @staticmethod
    def _uniform_crossover(first_position, second_position):

        alpha = np.random.randint(0, 2, first_position.shape)

        offspring1 = np.multiply(alpha, first_position) + np.multiply(1 - alpha, second_position)
        offspring2 = np.multiply(alpha, second_position) + np.multiply(1 - alpha, first_position)

        return offspring1, offspring2

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

    def _mutation(self, parent_ind):

        parent_postion = self._population_main[parent_ind].position

        selected_index = np.random.choice(range(self._number_of_variables),
                                          self._number_variables_to_mutate, replace=False)

        offspring = np.copy(parent_postion)
        offspring[:, selected_index] = 1 - offspring[:, selected_index]

        return offspring

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

                self._population_crossover[iter_crossover].solution, \
                self._population_crossover[iter_crossover].cost = \
                self._cost_function(self._population_crossover[iter_crossover].position)


                self._population_crossover[iter_crossover + 1].solution, \
                self._population_crossover[iter_crossover + 1].cost = \
                self._cost_function(self._population_crossover[iter_crossover + 1].position)

            for iter_mutation in range(self._number_of_mutation):

                parent_index = self._selection()

                self._population_mutation[iter_mutation].position = self._mutation(parent_index)

                self._population_mutation[iter_mutation].solution, \
                self._population_mutation[iter_mutation].cost = \
                self._cost_function(self._population_mutation[iter_mutation].position)

            self._population_main.extend(self._population_crossover)
            self._population_main.extend(self._population_mutation)
            self._population_main = self._sort_population(self._population_main)
            self._population_main = self._population_main[:self._number_of_population]
            self._best_costs.append(self._best_solution.cost)

        toc = time.time()

        plt.figure(dpi=600, figsize=(10, 6))
        plt.plot(range(self._max_iteration), self._best_costs)
        plt.title("Knapsack Problem Using Binary Genetic Algorithm", fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("./cost_function.png")

        return self._best_solution, toc - tic
