from ga_objects import Individual
from ga_objects import Population
import model_runner

def runAdditionalMethods(indiv):

    indiv.parameters["verbose"] = True
    indiv.parameters["additional_metrics"] = True
    model_runner.runConfiguration(indiv)
    print(indiv.parameters)

    return

def runFullFeatureSample():
    genes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    runSampleIndiv(genes)
    return

def runSampleIndiv(genes):

    indiv = Individual()
    indiv = indiv.generatedIndiv(genes)
    indiv.parameters["verbose"] = True
    indiv.parameters["additional_metrics"] = True
    model_runner.runConfiguration(indiv)

    return

def getExecutionParameters():
    parameters = {"classes_config":"standard", "model":"svm", "train_method":"standard", "fold_type":"kfold", "metric":"auc_roc", "verbose":False, "additional_metrics":False}
    return parameters

def main():

    parameters = getExecutionParameters()
    stop_condition = False
    verbose = True
    GENS_WITHOUT_IMPROVEMENT_TARGET = 20
    GENS_LIMIT = 60

    generations = []
    population = Population(params=parameters)
    population.initiatePopulation()

    generations.append(population)
    gens_without_improvement = 0

    while(not stop_condition):
        print("Generation {}\n params: {}".format(len(generations), population.parameters))
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

        if(gens_without_improvement > GENS_WITHOUT_IMPROVEMENT_TARGET or len(generations) > GENS_LIMIT):
            stop_condition = True

        if(verbose):
            population.printPopulation(len(generations) - 1)

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

