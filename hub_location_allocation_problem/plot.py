import numpy as np
import matplotlib.pyplot as plt



def plot_assignment(solution, data):

    customer_x = data["customer_x"]
    customer_y = data["customer_y"]

    server_x = data["server_x"]
    server_y = data["server_y"]

    solution_parsed = solution.solution["solution_parsed"]
    opened_servers = list(set([int(i) for i in solution_parsed[:, 1]]))

    plt.figure(dpi=600, figsize=(10, 6))
    plt.scatter(customer_x, customer_y, marker="x", c="black", label="customers")
    plt.scatter(server_x, server_y, marker="o", c="red", label="servers")
    plt.scatter(server_x[opened_servers], server_y[opened_servers], marker="o", s=90, c="green", label="opened servers")

    for i in range(len(customer_x)):

        plt.plot([customer_x[i], server_x[int(solution_parsed[i, 1])]],
                 [customer_y[i], server_y[int(solution_parsed[i, 1])]], c="black")

    plt.legend()

    plt.savefig("./assignment.png")

