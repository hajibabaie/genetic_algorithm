import numpy as np
import os


class Data:

    def __init__(self,
                 number_of_items,
                 min_range_of_values,
                 max_range_of_values,
                 min_range_of_weights,
                 max_range_of_weights):

        self._number_of_items = number_of_items
        self._min_range_of_values = min_range_of_values
        self._max_range_of_values = max_range_of_values
        self._min_range_of_weights = min_range_of_weights
        self._max_range_of_weights = max_range_of_weights

    def create_and_save(self):

        values = np.random.randint(self._min_range_of_values,
                                   self._max_range_of_values,
                                   self._number_of_items)

        weights = np.random.randint(self._min_range_of_weights,
                                    self._max_range_of_weights,
                                    self._number_of_items)

        knapsack_capacity = np.array([np.sum(weights) / 20])

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/values.csv", values)
        np.savetxt("./data/weights.csv", weights)
        np.savetxt("./data/knapsack_capacity.csv", knapsack_capacity)

    @staticmethod
    def load():

        out = {

            "values": np.loadtxt("./data/values.csv"),
            "weights": np.loadtxt("./data/weights.csv"),
            "knapsack_capacity": np.loadtxt("./data/knapsack_capacity.csv")
        }

        return out
