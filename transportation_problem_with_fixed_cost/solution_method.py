import numpy as np
import matplotlib.pyplot as plt
import time


class GA:

    class _Individual:

        def __init__(self):

            self.position = {"real": None,
                             "binary": None}
            self.solution = None
            self.cost = None

    def __init__(self,
                 cost_function,
                 max_iteration,
                 number_population,
                 variable_shape,
                 min_variables,
                 max_variables,
                 crossover_percentage,
                 mutation_percentage,
                 mutation_rate,
                 selection_pressure,
                 selection_probs,
                 tournament_size,
                 crossover_probs,
                 lambd1,
                 lambd2):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_population = number_population
        self._variables_shape = variable_shape
        self._min_variables = min_variables
        self._max_variables = max_variables
        self._crossover_percentage = crossover_percentage
        self._mutation_percentage = mutation_percentage
        self._mutation_rate = mutation_rate
        self._selection_pressure = selection_pressure
        self._selection_probs = selection_probs
        self._tournament_size = tournament_size
        self._crossover_probs = crossover_probs
        self._number_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_population / 2))
        self._number_mutation = int(np.ceil(self._mutation_percentage * self._number_population))
        self._population_main = [self._Individual() for _ in range(self._number_population)]
        self._population_crossover = None
        self._population_mutation = None
        self._population_probs = None
        self._best_solution = None
        self._best_costs = []
        self._worst_cost = None
        self._lambd1 = lambd1
        self._lambd2 = lambd2

    def _initialization(self):

        for i in range(self._number_population):

            self._population_main[i].position["real"] = np.random.uniform(self._min_variables,
                                                                          self._max_variables,
                                                                          self._variables_shape["real"])
            self._population_main[i].position["binary"] = np.random.randint(0, 2, self._variables_shape["binary"])

            self._population_main[i].solution, self._population_main[i].cost = \
            self._cost_function(self._population_main[i].position)

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

        selected_indices_argmin = np.argmin([self._population_main[i].cost for i in selected_indices])

        return int(selected_indices[selected_indices_argmin])

    def _selection(self):

        method = self._roulette_wheel_selection(self._selection_probs)

        if method == 0:

            return self._roulette_wheel_selection(self._population_probs)

        elif method == 1:

            return self._tournament_selection()

    def _single_point_crossover(self, first_position, second_position):

        cut_point = np.random.choice(range(1, self._variables_shape["binary"][1]))

        offspring1 = np.concatenate((first_position[:, :cut_point], second_position[:, cut_point:]), axis=1)
        offspring2 = np.concatenate((second_position[:, :cut_point], first_position[:, cut_point:]), axis=1)

        return offspring1, offspring2

    def _double_point_crossover(self, first_position, second_position):

        cut_points = np.random.choice(range(1, self._variables_shape["binary"][1]), 2, replace=False)

        first_cutpoint, second_cutpoint = min(cut_points), max(cut_points)

        offspring1 = np.concatenate((first_position[:, :first_cutpoint],
                                     second_position[:, first_cutpoint:second_cutpoint],
                                     first_position[:, second_cutpoint:]), axis=1)

        offspring2 = np.concatenate((second_position[:, :first_cutpoint],
                                     first_position[:, first_cutpoint:second_cutpoint],
                                     second_position[:, second_cutpoint:]), axis=1)

        return offspring1, offspring2

    @staticmethod
    def _uniform_crossover(first_position, second_position):

        alpha = np.random.randint(0, 2, first_position.shape)

        offspring1 = np.multiply(first_position, alpha) + np.multiply(second_position, 1 - alpha)
        offspring2 = np.multiply(second_position, alpha) + np.multiply(first_position, 1 - alpha)

        return offspring1, offspring2

    def _mutation_binary(self, parent_index):

        parent_position = self._population_main[parent_index].position["binary"]

        number_of_mutation = int(np.ceil(self._mutation_rate * parent_position.shape[1]))

        mutated_indices = np.random.choice(range(int(parent_position.shape[1])), number_of_mutation, replace=False)

        offspring = np.copy(parent_position)

        offspring[:, mutated_indices] = 1 - offspring[:, mutated_indices]

        return offspring

    def _arithmetic_crossover(self, first_index, second_index):

        first_position = self._population_main[first_index].position["real"]
        second_position = self._population_main[second_index].position["real"]

        alpha = np.random.uniform(-self._lambd1, 1 + self._lambd1, first_position.shape)

        offspring1 = np.multiply(alpha, first_position) + np.multiply(1 - alpha, second_position)
        offspring2 = np.multiply(alpha, second_position) + np.multiply(1 - alpha, first_position)

        offspring1 = np.clip(offspring1, self._min_variables, self._max_variables)
        offspring2 = np.clip(offspring2, self._min_variables, self._max_variables)

        return offspring1, offspring2

    def _arithmetic_mutation(self, parent_index):

        parent = self._population_main[parent_index].position["real"]

        number_of_mutation = int(np.ceil(self._mutation_rate * parent.size))

        rows = np.random.choice(range(parent.shape[0]), number_of_mutation)
        cols = np.random.choice(range(parent.shape[1]), number_of_mutation)

        offspring = np.copy(parent)

        sigma = self._lambd2 * (self._max_variables - self._min_variables)

        offspring[rows, cols] = offspring[rows, cols] + sigma * np.random.randn(number_of_mutation)

        offspring = np.clip(offspring, self._min_variables, self._max_variables)

        return offspring

    def _crossover_binary(self, first_index, second_index):

        first_parent = self._population_main[first_index].position["binary"]
        second_parent = self._population_main[second_index].position["binary"]

        method = self._roulette_wheel_selection(self._crossover_probs)

        if method == 0:

            return self._single_point_crossover(first_parent, second_parent)

        elif method == 1:

            return self._double_point_crossover(first_parent, second_parent)

        elif method == 2:

            return self._uniform_crossover(first_parent, second_parent)



    def run(self):

        tic = time.time()

        self._initialization()

        self._population_main = self._sort_population(self._population_main)

        for iter_main in range(self._max_iteration):

            self._calc_probs()

            self._population_crossover = [self._Individual() for _ in range(self._number_crossover)]

            self._population_mutation = [self._Individual() for _ in range(self._number_mutation)]

            for iter_crossover in range(0, self._number_crossover, 2):

                first_index = self._selection()
                second_index = self._selection()
                while first_index == second_index:
                    second_index = self._selection()

                self._population_crossover[iter_crossover].position["real"], \
                self._population_crossover[iter_crossover + 1].position["real"] = \
                self._arithmetic_crossover(first_index, second_index)

                self._population_crossover[iter_crossover].position["binary"], \
                self._population_crossover[iter_crossover + 1].position["binary"] = \
                self._crossover_binary(first_index, second_index)

                self._population_crossover[iter_crossover].solution, \
                self._population_crossover[iter_crossover].cost = \
                self._cost_function(self._population_crossover[iter_crossover].position)

                self._population_crossover[iter_crossover + 1].solution, \
                self._population_crossover[iter_crossover + 1].cost = \
                self._cost_function(self._population_crossover[iter_crossover + 1].position)

            for iter_mutation in range(self._number_mutation):

                parent_index = self._selection()

                self._population_mutation[iter_mutation].position["real"] = \
                self._arithmetic_mutation(parent_index)

                self._population_mutation[iter_mutation].position["binary"] = \
                self._mutation_binary(parent_index)

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
        plt.title("Transportation Problem With Fixed Cost Using Real and Binary Genetic Algorithm", fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("./cost_function.png")

        return self._best_solution, toc - tic
