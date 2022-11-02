from genetic_algorithm.transportation_problem_with_fixed_cost.data import Data
import numpy as np


class TransportationProblemWithFixedCost:

    def __init__(self):

        self._model_data = Data.load()

    def _parse_solution(self, x):

        if np.all(x["binary"] == 0):
            x["binary"][0, 0] = 1

        x_real = x["real"]
        x_binary = x["binary"]

        customer_demand = self._model_data["customer_demand"]
        customer_demand = np.reshape(customer_demand, (customer_demand.shape[0], 1))

        x_real_in_x_binary = np.multiply(x_real, x_binary)
        x_real_in_x_binary_sum = np.sum(x_real_in_x_binary, axis=1, keepdims=True)
        x_parsed = np.divide(x_real_in_x_binary, x_real_in_x_binary_sum)
        out = np.multiply(x_parsed, customer_demand)

        return out

    def cost_function(self, x):

        x_parsed = self._parse_solution(x)

        opened_supplier = x["binary"]

        transportation_cost = self._model_data["transportation_cost"]
        supplier_fixed_cost = self._model_data["supplier_fixed_cost"]
        supplier_capacity = self._model_data["supplier_capacity"]
        supplier_capacity = np.reshape(supplier_capacity, (1, supplier_capacity.shape[0]))

        cost_first_part = np.sum(np.multiply(transportation_cost, x_parsed))

        cost_second_part = np.sum(np.multiply(supplier_fixed_cost, opened_supplier))

        supplier_capacity_violation = np.maximum(np.divide(np.sum(x_parsed, axis=0), supplier_capacity) - 1, 0)

        supplier_capacity_violation_mean = np.mean(supplier_capacity_violation)

        cost_total = (cost_first_part + cost_second_part) * (1 + 10 * supplier_capacity_violation_mean)

        out = {
            "x_parsed": x_parsed,
            "opened_supplier": opened_supplier,
            "cost_first_part": cost_first_part,
            "cost_second_part": cost_second_part,
            "supplier_capacity_violation": supplier_capacity_violation,
            "supplier_capacity_violation_mean": supplier_capacity_violation_mean,
        }

        return out, cost_total
