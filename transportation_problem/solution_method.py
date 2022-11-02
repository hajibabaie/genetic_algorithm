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
                 variables_shape,
                 crossover_percentage,
                 mutation_percentage,
                 min_range_of_variables,
                 max_range_of_variables,
                 mutation_rate,
                 selection_pressure,
                 tournament_size,
                 selection_probs,
                 lambda1,
                 lambda2):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_of_population = number_of_population
        self._variables_shape = variables_shape
        self._crossover_percentage = crossover_percentage
        self._number_of_crossover = 2 * int(np.ceil(self._crossover_percentage * self._number_of_population / 2))
        self._mutation_percentage = mutation_percentage
        self._min_variables = min_range_of_variables
        self._max_variables = max_range_of_variables
        self._number_of_mutation = int(np.ceil(self._mutation_percentage * self._number_of_population))
        self._mutation_rate = mutation_rate
        self._selection_pressure = selection_pressure
        self._tournament_size = tournament_size
        self._selection_probs = selection_probs
        self._lambd1 = lambda1
        self._lambd2 = lambda2
        self._population_main = [self._Individual() for _ in range(self._number_of_population)]
        self._population_crossover = None
        self._population_mutation = None
        self._population_probs = None
        self._best_solution = None
        self._best_costs = []
        self._worst_cost = None


    def _initialization(self):

        for i in range(self._number_of_population):

            self._population_main[i].position = np.random.uniform(self._min_variables, self._max_variables,
                                                                  self._variables_shape)

            self._population_main[i].solution, \
            self._population_main[i].cost = \
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

        population_cost_sum = np.sum(population_cost_exp)

        self._population_probs = np.divide(population_cost_exp, population_cost_sum)

    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.random()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number <= probs_cumsum)[0][0])

    def _tournament_selection(self):

        selected_index = np.random.choice(range(self._number_of_population), self._tournament_size, replace=False)

        selected_population_cost_argmin = np.argmin([self._population_main[i].cost for i in selected_index])

        return int(selected_index[selected_population_cost_argmin])

    def _selection(self):

        method = self._roulette_wheel_selection(self._selection_probs)

        if method == 0:

            return self._roulette_wheel_selection(self._population_probs)

        elif method == 1:

            return self._tournament_selection()

    def _crossover(self, first_index, second_index):

        first_parent_position = self._population_main[first_index].position
        second_parent_position = self._population_main[second_index].position

        alpha = np.random.uniform(-self._lambd1, 1 + self._lambd1, first_parent_position.shape)

        offspring1 = np.multiply(alpha, first_parent_position) + np.multiply(1 - alpha, second_parent_position)
        offspring2 = np.multiply(alpha, second_parent_position) + np.multiply(1 - alpha, first_parent_position)

        offspring1 = np.clip(offspring1, self._min_variables, self._max_variables)
        offspring2 = np.clip(offspring2, self._min_variables, self._max_variables)

        return offspring1, offspring2

    def _mutation(self, parent_inx):

        parent_position = self._population_main[parent_inx].position

        number_of_mutation = int(np.ceil(self._mutation_rate * parent_position.size))

        rows = np.random.choice(range(parent_position.shape[0]), number_of_mutation)
        cols = np.random.choice(range(parent_position.shape[1]), number_of_mutation)

        offspring = np.copy(parent_position)

        sigma = self._lambd2 * (self._max_variables - self._min_variables)

        offspring[rows, cols] = offspring[rows, cols] + sigma * np.random.randn(number_of_mutation)

        offspring = np.clip(offspring, self._min_variables, self._max_variables)

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
        plt.title("Transportation Problem Using Real Genetic Algorithm", fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("./cost_function.png")

        return self._best_solution, toc - tic
