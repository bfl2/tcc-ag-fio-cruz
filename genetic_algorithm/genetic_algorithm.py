from ga_objects import Individual
from ga_objects import Population
import model_runner
import time
import json

def printElapsedTime(start_time, name):
    elapsed_time = time.time() - start_time
    time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("{} - Elapsed Time:{}".format(name, time_formatted))
    return

def runAdditionalMethods(indiv):

    indiv.parameters["verbose"] = True
    indiv.parameters["additional_metrics"] = True
    model_runner.runConfiguration(indiv)
    print(indiv.parameters)

    return

def runFullFeatureSample(parameters):
    genes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    runSampleIndiv(genes, parameters)
    return

def runSampleIndiv(genes, parameters):

    indiv = Individual(parameters)
    indiv = indiv.generatedIndiv(genes)
    indiv.parameters["verbose"] = True
    indiv.parameters["additional_metrics"] = True
    model_runner.runConfiguration(indiv)

    return

def importParametersFromJson():

    return

def getExecutionParameters(classes_config, model, balance_method, fold_type, metric, verbose, additional_metrics):
    """
    Attributes:
        classes_config:
            Description: How classes are separated into y column.
                Present classes in the dataset: (F0 = 6), (F1 = 58), (F2 = 80), (F3 = 70), (F4 = 41), (HCC = 42)
                standard separation is class HCC = 1 and the remaining classes = 0

            Possible Values: 'standard', 'F4XHCC', 'F0,F1,F2,F3XHCC', 'F0,F1,F2,F3XF4,HCC'


        model:
            Description: Classifier model to be used
            Possible Values: 'svm', 'mlp', 'random_forest', 'gradient_boosting', 'one_class_svm'

        balance_method:
            Description: Which strategy to use when balancing the training set. One Class strategy removes all ocurrences of IsHCC=0
            Possible Values: 'one_class', 'integer_balanced', 'float_balanced'(default)

        fold_type:
             Description: How the dataset is going to be split in the training/test
             Possible Values: 'kfold'(default 5 kfold), 'leave_one_out'

        metric:
             Description: Which metric to consider when scoring a model
             Possible Values: 'auc_roc', 'balanced_accuracy', 'accuracy'

        verbose:
            Description: If the program should log text
            Possible Value: Boolean

        additional_metrics:
            Description: If the program should display additional metrics
            Possible Value: Boolean
    """
    parameters = {"classes_config":classes_config, "model":model, "balance_method":balance_method, "fold_type":fold_type, "metric":metric, "verbose":verbose, "additional_metrics":additional_metrics}
    return parameters


def geneticAlgorithm(parameters):

    stop_condition = False
    verbose = True
    GENS_WITHOUT_IMPROVEMENT_TARGET = 20
    GENS_LIMIT = 60

    generations = []
    population = Population(params=parameters)
    population.initiatePopulation()

    generations.append(population)
    gens_without_improvement = 0
    print("Running Genetic Algorithm with params: {}".format(population.parameters))
    while(not stop_condition):
        print("Generation {}".format(len(generations)))
        if(population.parameters["verbose"]):
            print("params: {}", population.parameters)
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

        population.printPopulation(len(generations) - 1)


    best_indiv = generations[-1].indivs[0]

    genetic_algorithm_results = {"best_indiv":best_indiv, "generations":generations}

    return genetic_algorithm_results

def main():
    ### Run Genetic algorithm for feature selection
    params = getExecutionParameters("standard", "svm", "float_balanced", "kfold", "auc_roc", False, False)
    start_time = time.time()

    print("\n### Running Genetic Algorithm")
    ga_results = geneticAlgorithm(params)
    print("### Finished Running Genetic Algorithm")

    printElapsedTime(start_time, "Genetic Algorithm")

    start_time - time.time()
    print("\n### Running Best Indiv simulation:")
    runAdditionalMethods(ga_results["best_indiv"])
    printElapsedTime(start_time, "Best Model Execution")
    print("### Finished Best Indiv simulation:")

    ### Running classifier with full features
    print("\n### Running Full Features model")
    runFullFeatureSample(params)
    print("### Finished Full Features model")
    return


if __name__ == "__main__":
    main()

