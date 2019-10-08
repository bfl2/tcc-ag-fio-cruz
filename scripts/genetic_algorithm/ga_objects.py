import model_runner
import random

""" Global Constants """
random_state = 42

""" Individual """
class Individual:
    """ Class constants """
    SNPs = ["PTX3 rs1840680","PTX3 rs2305619","MBL -221","IL-10 -1082","IL-10 -819","IL-10 -592","TNF-308","SOD2","MPO C-463T","IL-28b rs12979860"]
    def __init__(self):
        self.gene_count = 10
        self.parameters = self.getDefaultParameters()
        self.genes = []
        self.fitness = 0
        self.fitnessCalc = False
        return

    def generateRandomIndiv(self):
        class_chances = {"class0_chance":0.5, "class1_chance":0.5}
        while (len(self.genes) < self.gene_count):
            seed = random.random()
            if(seed < class_chances["class0_chance"]):
                value = 0
            else:
                value = 1
            self.genes.append(value)
        return

    def getDefaultParameters(self):
        parameters = {"model":"random_forest", "dataset_scope_reduced":False, "verbose":False}
        return parameters

    def computeFitness(self):
        if(len(self.genes) < self.gene_count):
            self.generateRandomIndiv()
        if(not self.fitnessCalc):
            self.fitness = model_runner.runConfiguration(self)
            self.fitnessCalc = True
        return self.fitness

""" Population """
class Population:
    """ Class constants """

    def __init__(self):
        self.popSize = 10
        self.elite_frac = 0.2
        self.indivs = []
        self.avg_fitness = 0
        self.max_fitness = 0
        return

    def initiatePopulation(self):
        while(len(self.indivs) < self.popSize):
            indiv = Individual()
            indiv.generateRandomIndiv()
            indiv.computeFitness()
            self.indivs.append(indiv)
        return

    def printPopulation(self):
        for indiv in self.indivs:
            print("Index:{} Genes:{} Fitness:{}".format(self.indivs.index(indiv), indiv.genes, indiv.fitness) )

        return

    def sortPopulation(self):
        self.indivs.sort(key=lambda x: x.fitness, reverse=True)
        return

    def calculateMetrics(self):
        self.sortPopulation()
        if(len(self.indivs) > 0):
            self.max_fitness = self.indivs[0].fitness
            sum = sum(indiv.fitness for indiv in self.indivs)
            self.avg_fitness = sum/len(self.indivs)
        else:
            print("Error: Population is empty")

        return

    def selection(self):

        return