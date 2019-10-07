from ga_objects import Individual
from ga_objects import Population
import model_runner


def main():

    population = Population()
    population.initiatePopulation()
    population.printPopulation()


if __name__ == "__main__":
    print("### Running Genetic Algorithm")
    main()
    print("### Finished Running Genetic Algorithm")

