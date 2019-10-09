from ga_objects import Individual
from ga_objects import Population
import model_runner

def runAdditionalMethods(indiv):

    indiv.parameters["verbose"] = True
    indiv.parameters["additional_metrics"] = True
    model_runner.runConfiguration(indiv)

    return

def runSampleIndiv():

    genes = [0,0,0,0,0,1,0,1,0,0]
    indiv = Individual()
    indiv.generateRandomIndiv()
    indiv.parameters["verbose"] = True
    indiv.parameters["additional_metrics"] = True
    model_runner.runConfiguration(indiv)

    return

def main():

    stop_condition = False
    GENS_WITHOUT_IMPROVEMENT = 4
    GENS_LIMIT = 10

    generations = []
    population = Population()
    population.initiatePopulation()

    generations.append(population)
    gens_without_improvement = 0

    while(not stop_condition):
        print("Generation {}".format(len(generations)))
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

        if(gens_without_improvement > GENS_WITHOUT_IMPROVEMENT or len(generations) > GENS_LIMIT):
            stop_condition = True


    for gen in generations:
        gen.printPopulation(generations.index(gen))


    print("### Running Best Indiv simulation:")
    best_indiv = generations[-1].indivs[0]
    runAdditionalMethods(best_indiv)

if __name__ == "__main__":
    print("### Running Genetic Algorithm")
    main()
    ##runSampleIndiv()
    print("### Finished Running Genetic Algorithm")

