from ga_objects import Individual
from ga_objects import Population
import model_runner


def main():

    stop_condition = False
    gens_without_improvement_limit = 4

    generations = []
    population = Population()
    population.initiatePopulation()

    generations.append(population)
    gens_without_improvement = 0

    while(not stop_condition):

        ##Select Parents
        population.parentSelection()
        ##Generate Children
        population = population.generateChildren()
        population.applySelectionPressure()

        if(population.max_fitness > generations[-1].max_fitness):
            gens_without_improvement = 0
        else:
            gens_without_improvement += 1

        generations.append(population)

        if(gens_without_improvement > gens_without_improvement_limit or len(generations) > 10):
            stop_condition = True


    for gen in generations:
        gen.printPopulation(generations.index(gen))


    print("### Running Best Indiv simulation:")
    best_indiv = generations[-1].indivs[0]
    best_indiv.parameters["verbose"] = True
    model_runner.runConfiguration(best_indiv)

if __name__ == "__main__":
    print("### Running Genetic Algorithm")
    main()
    print("### Finished Running Genetic Algorithm")

