from genetic_algorithm.knapsack_problem_binary.data import Data
from genetic_algorithm.knapsack_problem_binary.cost_function import KnapsackProblem
from genetic_algorithm.knapsack_problem_binary.solution_method import GA


def main():

    # model_data = Data(number_of_items=50,
    #                   min_range_of_values=10,
    #                   max_range_of_values=90,
    #                   min_range_of_weights=100,
    #                   max_range_of_weights=900)
    #
    # model_data.create_and_save()

    data = Data.load()

    problem = KnapsackProblem()
    cost_function = problem.cost_function

    solution_method = GA(cost_function=cost_function,
                         max_iteration=40,
                         number_of_population=40,
                         number_of_variables=len(data["values"]),
                         crossover_percentage=0.8,
                         crossover_probs=[0.3, 0.3, 0.4],
                         mutation_percentage=0.4,
                         mutation_rate=0.03,
                         selection_pressure=7,
                         selection_probs=[0.5, 0.5],
                         tournament_size=5)

    solution_best, run_time = solution_method.run()

    return data, solution_best, run_time


if __name__ == "__main__":

    model_data, solution, runtime = main()
