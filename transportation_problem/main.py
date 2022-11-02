from genetic_algorithm.transportation_problem.data import Data
from genetic_algorithm.transportation_problem.cost_function import TransportationProblem
from genetic_algorithm.transportation_problem.solution_method import GA


def main():

    # model_data = Data(number_of_customers=40,
    #                   number_of_suppliers=6,
    #                   customers_demand_min=10,
    #                   customers_demand_max=90)
    #
    # model_data.create_and_save()

    model_data = Data.load()

    problem = TransportationProblem()

    cost_function = problem.cost_function

    solution_method = GA(cost_function=cost_function,
                         max_iteration=6000,
                         number_of_population=100,
                         variables_shape=model_data["distances"].shape,
                         crossover_percentage=0.8,
                         mutation_percentage=0.4,
                         min_range_of_variables=0,
                         max_range_of_variables=1,
                         mutation_rate=0.05,
                         selection_pressure=100,
                         tournament_size=20,
                         selection_probs=[0.5, 0.5],
                         lambda1=0.1,
                         lambda2=0.1)

    solution_best, run_time = solution_method.run()

    return solution_best, run_time


if __name__ == "__main__":

    solution, runtime = main()
