from genetic_algorithm.transportation_problem.data import Data
import numpy as np


class TransportationProblem:

    def __init__(self):

        self._model_data = Data.load()

    def _parse_solution(self, x):

        customer_demand = self._model_data["customer_demand"]
        customer_demand = np.reshape(customer_demand, (customer_demand.shape[0], 1))

        x_sum = np.sum(x, axis=1, keepdims=True)
        x_parsed = np.divide(x, x_sum)
        x_parsed_in_demand = np.multiply(x_parsed, customer_demand)

        return x_parsed_in_demand

    def cost_function(self, x):

        x_parsed_in_demand = self._parse_solution(x)

        supplier_capacity = self._model_data["supplier_capacity"]
        supplier_capacity = np.reshape(supplier_capacity, (1, supplier_capacity.shape[0]))

        distances = self._model_data["distances"]

        supplier_sent = np.sum(x_parsed_in_demand, axis=0, keepdims=True)

        supplier_capacity_violation = np.maximum(np.divide(supplier_sent, supplier_capacity) - 1, 0)

        supplier_capacity_violation_mean = np.mean(supplier_capacity_violation)

        cost = np.sum(np.multiply(distances, x_parsed_in_demand)) * (1 + 100 * supplier_capacity_violation_mean)

        out = {
            "solution_parsed": x_parsed_in_demand,
            "supplier_sent": supplier_sent,
            "supplier_capacity_violation": supplier_capacity_violation,
            "supplier_capacity_violation_mean": supplier_capacity_violation_mean,
        }

        return out, cost
