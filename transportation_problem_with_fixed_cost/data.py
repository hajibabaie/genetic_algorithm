import numpy as np
import os


class Data:

    def __init__(self,
                 number_of_supplier,
                 number_of_customers,
                 supplier_fixed_cost_min,
                 supplier_fixed_cost_max,
                 customers_demand_min,
                 customers_demand_max):

        self._number_suppliers = number_of_supplier
        self._number_customers = number_of_customers
        self._supplier_fixed_cost_min = supplier_fixed_cost_min
        self._supplier_fixed_cost_max = supplier_fixed_cost_max
        self._customer_demand_min = customers_demand_min
        self._customer_demand_max = customers_demand_max
        self._supplier_capacity_min = None
        self._supplier_capacity_max = None

    def create_and_save(self):

        customer_x = np.random.uniform(0, 100, self._number_customers)
        customer_y = np.random.uniform(0, 100, self._number_customers)

        servers_x = np.random.uniform(0, 100, self._number_suppliers)
        servers_y = np.random.uniform(0, 100, self._number_suppliers)

        travelling_cost = np.zeros((self._number_customers, self._number_suppliers))
        for i in range(self._number_customers):
            for j in range(self._number_suppliers):
                travelling_cost[i, j] = np.sqrt(np.square(customer_x[i] - servers_x[j]) +
                                                np.square(customer_y[i] - servers_y[j]))

        customer_demand = np.random.uniform(self._customer_demand_min,
                                            self._customer_demand_max,
                                            self._number_customers)

        self._supplier_capacity_min = np.sum(customer_demand) / self._number_suppliers
        self._supplier_capacity_max = 1.2 * self._supplier_capacity_min

        supplier_capacity = np.random.uniform(self._supplier_capacity_min,
                                              self._supplier_capacity_max,
                                              self._number_suppliers)

        supplier_fixed_cost = np.random.uniform(self._supplier_fixed_cost_min,
                                                self._supplier_fixed_cost_max,
                                                self._number_suppliers)

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/transportation_cost.csv", travelling_cost, delimiter=",")
        np.savetxt("./data/customer_demand.csv", customer_demand)
        np.savetxt("./data/supplier_capacity.csv", supplier_capacity)
        np.savetxt("./data/supplier_fixed_cost.csv", supplier_fixed_cost)

    @staticmethod
    def load():

        out = {
            "transportation_cost": np.loadtxt("./data/transportation_cost.csv", delimiter=","),
            "customer_demand": np.loadtxt("./data/customer_demand.csv"),
            "supplier_capacity": np.loadtxt("./data/supplier_capacity.csv"),
            "supplier_fixed_cost": np.loadtxt("./data/supplier_fixed_cost.csv")
        }

        return out
