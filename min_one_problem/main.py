from cost_function import min_one
from solution_method import GA


def main():

    cost_func = min_one

    solution_method = GA(cost_function=cost_func,
                         max_iteration=100,
                         number_population=20,
                         number_variables=100,
                         crossover_percentage=0.8,
                         mutation_percentage=0.4,
                         mutation_rate=0.03,
                         selection_pressure=7,
                         tournament_size=3,
                         selection_probs=[0.5, 0.5],
                         crossover_probs=[0.3, 0.3, 0.4])

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
