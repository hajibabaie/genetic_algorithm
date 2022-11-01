import numpy as np
import os


class Data:

    def __init__(self,
                 number_customers,
                 number_servers,
                 customer_min_demand,
                 customer_max_demand,
                 server_min_fixed_cost,
                 server_max_fixed_cost):

        self._num_customers = number_customers
        self._num_servers = number_servers
        self._customer_min_demand = customer_min_demand
        self._customer_max_demand = customer_max_demand
        self._server_min_fixed_cost = server_min_fixed_cost
        self._server_max_fixed_cost = server_max_fixed_cost


    def create_and_save(self):

        customer_x = np.random.uniform(0, 100, self._num_customers)
        customer_y = np.random.uniform(0, 100, self._num_customers)

        server_x = np.random.uniform(0, 100, self._num_servers)
        server_y = np.random.uniform(0, 100, self._num_servers)

        distances = np.zeros((self._num_customers, self._num_servers))

        for i in range(self._num_customers):
            for j in range(self._num_servers):
                distances[i, j] = np.sqrt(np.square(customer_x[i] - server_x[j]) +
                                          np.square(customer_y[i] - server_y[j]))

        customer_demand = np.random.uniform(self._customer_min_demand, self._customer_max_demand, self._num_customers)
        server_fixed_cost = np.random.uniform(self._server_min_fixed_cost,
                                              self._server_max_fixed_cost,
                                              self._num_servers)

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/customer_x.csv", customer_x)
        np.savetxt("./data/customer_y.csv", customer_y)
        np.savetxt("./data/server_x.csv", server_x)
        np.savetxt("./data/server_y.csv", server_y)
        np.savetxt("./data/distances.csv", distances, delimiter=",")
        np.savetxt("./data/customer_demand.csv", customer_demand)
        np.savetxt("./data/server_fixed_cost.csv", server_fixed_cost)

    @staticmethod
    def load():

        out = {
            "customer_x": np.loadtxt("./data/customer_x.csv"),
            "customer_y": np.loadtxt("./data/customer_y.csv"),
            "server_x": np.loadtxt("./data/server_x.csv"),
            "server_y": np.loadtxt("./data/server_y.csv"),
            "distance": np.loadtxt("./data/distances.csv", delimiter=","),
            "customer_demand": np.loadtxt("./data/customer_demand.csv"),
            "server_fixed_cost": np.loadtxt("./data/server_fixed_cost.csv")
        }

        return out
