from genetic_algorithm.quadratic_assignment_problem.data import Data
import numpy as np


class QuadraticAssignmentProblem:


    def __init__(self):

        self._model_data = Data.load()

    def cost_function(self, x):

        distances = self._model_data["distances"]
        facilities_demand = self._model_data["facilities_demand"]
        number_of_facilities = facilities_demand.shape[0]
        x_parsed = x[:, :number_of_facilities]

        cost = 0
        for i in range(number_of_facilities):
            for j in range(i + 1, number_of_facilities):
                cost += distances[x_parsed[0, i], x_parsed[0, j]] * (facilities_demand[i, j] +
                                                                     facilities_demand[j, i])

        return cost
