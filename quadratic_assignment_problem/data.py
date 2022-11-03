import numpy as np
import os


class Data:

    def __init__(self,
                 number_of_locations,
                 number_of_facilities,
                 ):

        self._number_of_locations = number_of_locations
        self._number_of_facilities = number_of_facilities


    def create_and_save(self):

        locations_x = np.random.uniform(0, 100, self._number_of_locations)
        locations_y = np.random.uniform(0, 100, self._number_of_locations)

        distances = np.zeros((self._number_of_locations, self._number_of_locations))
        for i in range(self._number_of_locations):
            for j in range(i + 1, self._number_of_locations):
                distances[i, j] = np.sqrt(np.square(locations_x[i] - locations_x[j]) +
                                          np.square(locations_y[i] - locations_y[j]))
                distances[j, i] = distances[i, j]

        facilities_demand = np.random.uniform(5, 50, (self._number_of_facilities, self._number_of_facilities))
        facilities_demand = facilities_demand - np.diag(np.diag(facilities_demand))

        os.makedirs("./data", exist_ok=True)
        np.savetxt("./data/locations_x.csv", locations_x)
        np.savetxt("./data/locations_y.csv", locations_y)
        np.savetxt("./data/distances.csv", distances, delimiter=",")
        np.savetxt("./data/facilities_demand.csv", facilities_demand, delimiter=",")


    @staticmethod
    def load():

        out = {
            "locations_x": np.loadtxt("./data/locations_x.csv"),
            "locations_y": np.loadtxt("./data/locations_y.csv"),
            "distances": np.loadtxt("./data/distances.csv", delimiter=","),
            "facilities_demand": np.loadtxt("./data/facilities_demand.csv", delimiter=",")
        }

        return out
