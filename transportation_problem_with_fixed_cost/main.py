from genetic_algorithm.transportation_problem_with_fixed_cost.data import Data
from genetic_algorithm.transportation_problem_with_fixed_cost.cost_function import TransportationProblemWithFixedCost
from genetic_algorithm.transportation_problem_with_fixed_cost.solution_method import GA


def main():

    # model_data = Data(number_of_supplier=6,
    #                   number_of_customers=40,
    #                   supplier_fixed_cost_min=5000,
    #                   supplier_fixed_cost_max=10000,
    #                   customers_demand_min=10,
    #                   customers_demand_max=90)
    #
    # model_data.create_and_save()

    data = Data.load()
    variable_shape = {"real": data["transportation_cost"].shape, "binary": (1, data["supplier_capacity"].shape[0])}
    problem = TransportationProblemWithFixedCost()
    cost_function = problem.cost_function

    solution_method = GA(cost_function=cost_function,
                         max_iteration=100,
                         number_population=40,
                         variable_shape=variable_shape,
                         min_variables=0,
                         max_variables=1,
                         crossover_percentage=0.8,
                         mutation_percentage=0.4,
                         mutation_rate=0.05,
                         selection_pressure=4,
                         selection_probs=[0.5, 0.5],
                         tournament_size=3,
                         crossover_probs=[0.3, 0.3, 0.4],
                         lambd1=0.1,
                         lambd2=0.1)

    solution_best, run_time = solution_method.run()

    return data, solution_best, run_time


if __name__ == "__main__":

    model_data, solution_best, runtime = main()