from genetic_algorithm.hub_location_allocation_problem.data import Data
from genetic_algorithm.hub_location_allocation_problem.cost_function import HubLocationAllocation
from genetic_algorithm.hub_location_allocation_problem.solution_method import GA
from genetic_algorithm.hub_location_allocation_problem.plot import plot_assignment


def main():

    # data = Data(number_customers=100,
    #             number_servers=15,
    #             customer_min_demand=5,
    #             customer_max_demand=50,
    #             server_min_fixed_cost=8000,
    #             server_max_fixed_cost=12000)
    # data.create_and_save()

    model_data = Data.load()

    number_of_variables = model_data["server_x"].shape[0]

    problem = HubLocationAllocation()
    cost_function = problem.cost_function

    solution_method = GA(cost_function=cost_function,
                         max_iteration=100,
                         number_population=20,
                         number_variables=number_of_variables,
                         crossover_percentage=0.8,
                         mutation_percentage=0.4,
                         mutation_rate=0.05,
                         selection_pressure=7,
                         tournament_size=3,
                         selection_probs=[0.5, 0.5],
                         crossover_probs=[0.3, 0.3, 0.4])

    solution_best, runtime = solution_method.run()

    plot_assignment(solution_best, model_data)


    return model_data, solution_best, runtime


if __name__ == "__main__":
    data, solution, run_time = main()
