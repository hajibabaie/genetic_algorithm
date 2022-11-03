import matplotlib.pyplot as plt


def plot_assignment(solution, data):

    number_of_facilities = int(data["facilities_demand"].shape[0])

    locations_x = data["locations_x"]
    locations_y = data["locations_y"]

    solution_parsed = solution.position[:, :number_of_facilities]

    plt.figure(dpi=600, figsize=(10, 6))
    plt.scatter(locations_x, locations_y, marker="o", s=20, c="red")
    for i in range(number_of_facilities):
        plt.text(locations_x[solution_parsed[0, i]], locations_y[solution_parsed[0, i]], i)
    plt.scatter(locations_x[solution_parsed[0, :number_of_facilities]], locations_y[solution_parsed[0, :number_of_facilities]],
                marker="o", s=20, c="green")
    plt.savefig("./assignment.png")

