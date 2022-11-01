from genetic_algorithm.hub_location_allocation_problem.data import Data
import numpy as np



class HubLocationAllocation:


    def __init__(self):

        self._model_data = Data.load()


    def cost_function(self, solution):


        distance = self._model_data["distance"]
        customer_demand = self._model_data["customer_demand"]
        server_fixed_cost = self._model_data["server_fixed_cost"]

        distance_in_solution = np.multiply(distance, solution)
        distance_in_solution[distance_in_solution == 0] = np.inf

        solution_parsed = np.zeros((distance_in_solution.shape[0], 2))
        solution_parsed[:, 0] = np.min(distance_in_solution, axis=1)
        solution_parsed[:, 1] = np.argmin(distance_in_solution, axis=1)

        obj_func_first_part = np.dot(solution_parsed[:, 0], customer_demand)
        obj_func_second_part = np.dot(solution.ravel(), server_fixed_cost)

        cost = obj_func_first_part + obj_func_second_part

        out = {
            "objective_function_first_part": obj_func_first_part,
            "objective_function_second_part": obj_func_second_part,
            "solution_parsed": solution_parsed
        }

        return out, cost
