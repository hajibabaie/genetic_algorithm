from genetic_algorithm.quadratic_assignment_problem.data import Data
from genetic_algorithm.quadratic_assignment_problem.cost_function import QuadraticAssignmentProblem
from genetic_algorithm.quadratic_assignment_problem.solution_method import GA
from genetic_algorithm.quadratic_assignment_problem.plot import plot_assignment


def main():

    # model_data = Data(number_of_locations=30,
    #                   number_of_facilities=20)
    #
    # model_data.create_and_save()

    data = Data.load()
    number_of_facilities = int(data["distances"].shape[0])

    problem = QuadraticAssignmentProblem()

    cost_function = problem.cost_function


    solution_method = GA(cost_function=cost_function,
                         max_iteration=250,
                         variables_shape=number_of_facilities,
                         number_of_population=40,
                         crossover_percentage=0.8,
                         mutation_percentage=0.4,
                         selection_pressure=7,
                         mutation_probs=[0.3, 0.3, 0.4],
                         selection_probs=[0.5, 0.5],
                         tournament_size=3)

    solution_best, run_time = solution_method.run()

    plot_assignment(solution_best, data)

    return data, solution_best, run_time


if __name__ == "__main__":

    model_data, solution, runtime = main()
