from genetic_algorithm.knapsack_problem_binary.data import Data
import numpy as np


class KnapsackProblem:

    def __init__(self):

        self._model_data = Data.load()

    def cost_function(self, x):

        number_of_items = x.shape[1]
        values = self._model_data["values"].reshape((1, number_of_items))
        weights = self._model_data["weights"].reshape(1, number_of_items)
        knapsack_cap = self._model_data["knapsack_capacity"]


        values_gained = np.dot(values, x.T)[0, 0]
        values_not_gained = np.dot(values, 1 - x.T)[0, 0]

        weights_gained = np.dot(weights, x.T)[0, 0]
        weights_not_gained = np.dot(weights, 1 - x.T)[0, 0]

        capacity_violation = np.maximum((weights_gained / knapsack_cap) - 1, 0)

        cost = values_not_gained * (1 + 10 * capacity_violation)

        out = {
            "values_gained": values_gained,
            "values_not_gained": values_not_gained,
            "weights_gained": weights_gained,
            "weights_not_gained": weights_not_gained,
            "capacity_violation": capacity_violation
        }

        return out, cost
