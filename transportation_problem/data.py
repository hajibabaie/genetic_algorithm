import numpy as np
import os


class Data:


    def __init__(self,
                 number_of_customers,
                 number_of_suppliers,
                 customers_demand_min,
                 customers_demand_max):

        self._num_customers = number_of_customers
        self._num_suppliers = number_of_suppliers
        self._customers_demand_min = customers_demand_min
        self._customers_demand_max = customers_demand_max
        self._supplier_capacity_min = None
        self._supplier_capacity_max = None

    def create_and_save(self):

        customers_x = np.random.uniform(0, 100, self._num_customers)
        customers_y = np.random.uniform(0, 100, self._num_customers)

        servers_x = np.random.uniform(0, 100, self._num_suppliers)
        servers_y = np.random.uniform(0, 100, self._num_suppliers)

        distances = np.zeros((self._num_customers, self._num_suppliers))

        for i in range(self._num_customers):
            for j in range(self._num_suppliers):

                distances[i, j] = np.sqrt(np.square(customers_x[i] - servers_x[j]) +
                                          np.square(customers_y[i] - servers_y[j]))

        customers_demand = np.random.uniform(self._customers_demand_min,
                                             self._customers_demand_max,
                                             self._num_customers)

        self._supplier_capacity_min = 0.9 * np.sum(customers_demand) / self._num_suppliers
        self._supplier_capacity_max = 1.1 * self._supplier_capacity_min

        supplier_capacity = np.random.uniform(self._supplier_capacity_min,
                                              self._supplier_capacity_max,
                                              self._num_suppliers)

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/distances.csv", distances, delimiter=",")
        np.savetxt("./data/customers_demand.csv", customers_demand)
        np.savetxt("./data/suppliers_capacity.csv", supplier_capacity)

    @staticmethod
    def load():

        out = {

            "distances": np.loadtxt("./data/distances.csv", delimiter=","),
            "customer_demand": np.loadtxt("./data/customers_demand.csv"),
            "supplier_capacity": np.loadtxt("./data/suppliers_capacity.csv")
        }

        return out
