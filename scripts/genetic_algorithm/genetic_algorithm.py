from ga_objects import Individual
from ga_objects import Population
import model_runner


def main():

    stopCondition = False
    generations = []
    population = Population()
    population.initiatePopulation()
    population.calculateMetrics()
    generations.append(population)
    while(not stopCondition):
        population.calculateMetrics()
        population.printPopulation()

        generations.append(population)

        ##Selection
        population = population.selection()
        ##Crossover
        ##Mutation
        ##Compute fitness

        if(True):
            stopCondition = True


if __name__ == "__main__":
    print("### Running Genetic Algorithm")
    main()
    print("### Finished Running Genetic Algorithm")

