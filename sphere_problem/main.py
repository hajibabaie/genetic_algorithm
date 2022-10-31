from genetic_algorithm.sphere_problem.cost_function import sphere
from genetic_algorithm.sphere_problem.solution_method import GA


def main():


    cost_func = sphere

    solution_method = GA(cost_function=cost_func,
                         max_iteration=500,
                         number_population=20,
                         number_variables=20,
                         max_range_variables=10,
                         min_range_variables=-10,
                         crossover_percentage=0.8,
                         mutation_percentage=0.4,
                         mutation_rate=0.03,
                         tournament_size=3,
                         selection_pressure=7,
                         selection_probs=[0.5, 0.5],
                         lambd1=0.1,
                         lambd2=0.1)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
