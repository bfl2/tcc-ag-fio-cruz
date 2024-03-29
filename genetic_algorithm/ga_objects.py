import model_runner
import random
import numpy as np
import copy

""" Global Constants """
random_state = 42

""" Individual """
class Individual:
    """ Class constants """
    SNPs = ["PTX3 rs1840680","PTX3 rs2305619","MBL -221","IL-10 -1082","IL-10 -819","IL-10 -592","TNF-308","SOD2","MPO C-463T","IL-28b rs12979860"]
    def __init__(self, params = None):
        self.GENE_COUNT = 10
        if(params == None):
            self.parameters = self.getDefaultParameters()
        else:
            self.parameters = params
        self.genes = []
        self.fitness = 0
        self.fitnessCalc = False
        return

    def generateRandomIndiv(self):
        CLASS_CHANCES = {"class0_chance":0.5, "class1_chance":0.5}
        CLASS_CHANCES["class0_chance"] = random.uniform(0.3, 0.7)
        CLASS_CHANCES["class1_chance"] = 1 - CLASS_CHANCES["class0_chance"]
        new_genes = []
        while (len(new_genes) < self.GENE_COUNT):
            seed = random.random()
            if(seed < CLASS_CHANCES["class0_chance"]):
                value = 0
            else:
                value = 1
            new_genes.append(value)
        if(new_genes == [0,0,0,0,0,0,0,0,0,0]):
            new_genes[random.randrange(0,self.GENE_COUNT-1)] = 1

        new_indiv = self.generateIndiv(new_genes)
        return new_indiv

    def setParameters(self, param):
        self.parameters = param
        return

    def getDefaultParameters(self):
        parameters = {"classes_config":"standard", "model":"svm", "balance_method":"standard", "fold_type":"kfold", "metric":"auc_roc", "verbose":False, "additional_metrics":False}
        return parameters

    def computeFitness(self):
        if(not self.fitnessCalc):
            self.fitness = model_runner.runConfiguration(self)
            self.fitnessCalc = True
        return self.fitness

    def generateIndiv(self, genes):
        if(len(genes)!=self.GENE_COUNT):
            print("Error: length of genes does not match GENE_COUNT")
        elif(genes == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]): ### Genes must contain at least one feature
            seed = random.randrange(0, self.GENE_COUNT - 1)
            genes[seed] = 1
        newIndiv = Individual(params=self.parameters)
        newIndiv.genes = genes

        return newIndiv

    def crossover(self, second_indiv):
        genes1 = self.genes[:]
        genes2 = second_indiv.genes
        cutoff_index = random.randrange(1, self.GENE_COUNT - 2)

        new_genes1 = []
        new_genes1.extend(genes1[:cutoff_index])
        new_genes1.extend(genes2[cutoff_index:])

        new_genes2 = []
        new_genes2.extend(genes2[:cutoff_index])
        new_genes2.extend(genes1[cutoff_index:])

        generator = Individual(params=self.parameters)

        new_indivs = [generator.generateIndiv(new_genes1), generator.generateIndiv(new_genes2)]

        return new_indivs

    def mutationA(self, genes):
        index_mut = random.randrange(0, self.GENE_COUNT - 1)
        genes[index_mut] = (genes[index_mut] + 1) % 2
        return genes

    def mutationB(self, genes):
        index_mut1 = index_mut2 = index_mut3 = 0
        while(index_mut1 == index_mut2 or index_mut1 == index_mut3 or index_mut2 == index_mut3):
            index_mut1 = random.randrange(0, self.GENE_COUNT - 1)
            index_mut2 = random.randrange(0, self.GENE_COUNT - 1)
            index_mut3 = random.randrange(0, self.GENE_COUNT - 1)

        genes[index_mut1] = (genes[index_mut1] + 1) % 2
        genes[index_mut2] = (genes[index_mut2] + 1) % 2
        genes[index_mut3] = (genes[index_mut3] + 1) % 2

        return genes

    def mutation(self):
        mutA_chance, mutB_chance = 0.5, 0.5
        genes_c = self.genes[:]

        if(mutA_chance < random.random()):
            mutated_genes = self.mutationA(genes_c)
        else:
            mutated_genes = self.mutationB(genes_c)

        mutated_indiv = self.generateIndiv(mutated_genes)

        return mutated_indiv

""" Population """
class Population:
    """ Class constants """

    def __init__(self, params = None):
        self.parameters = params

        self.POP_SIZE = 30
        self.OFFSPRING_SIZE = 2*self.POP_SIZE
        self.ELITE_FRAC = 0.2
        self.CROSSOVER_CHANCE = 0.9
        self.MUTATION_CHANCE = 0.2
        self.RANDOM_INDIV_CHANCE = 0.2

        self.indivs = []
        self.next_pop_parents = []
        self.best_indiv = None

        self.avg_fitness = 0
        self.max_fitness = 0
        return

    def initiatePopulation(self):
        while(len(self.indivs) < self.POP_SIZE):
            generator = Individual(self.parameters)
            indiv = generator.generateRandomIndiv()
            indiv.computeFitness()
            self.indivs.append(indiv)
        self.calculateMetrics()
        return

    def generateRandomChilds(self, count):
        childs =[]
        while(len(childs) < count):
            generator = Individual(self.parameters)
            indiv = generator.generateRandomIndiv()
            childs.append(indiv)
        return childs

    def printPopulation(self, generation_index, print_pop=False):
        if(self.parameters["verbose"] or print_pop):
            print("### Population")
            for indiv in self.indivs:
                print("Index:{} Genes:{} Fitness:{}".format(self.indivs.index(indiv), indiv.genes, indiv.fitness) )

        print("### Generation:{}  avg_fit:{} | max_fit:{}".format(generation_index, self.avg_fitness, self.max_fitness))
        best_indiv = self.indivs[0]
        print("Best Indiv = Model:{} Metric:{} Score:{} | Features:{}\n".format(best_indiv.parameters["model"], best_indiv.parameters["metric"], best_indiv.fitness, best_indiv.genes))

        return

    def sortPopulation(self, pop=[]):
        if(len(pop) == 0):
            pop = self.indivs
        pop.sort(key=lambda x: x.fitness, reverse=True)
        generator = Individual(self.parameters)
        self.best_indiv = generator.generateIndiv(pop[0].genes)
        self.best_indiv.computeFitness()
        return

    def calculateMetrics(self, sort=True):
        if(sort):
            self.sortPopulation()
        if(len(self.indivs) > 0):
            self.max_fitness = self.indivs[0].fitness
            sum = np.sum(indiv.fitness for indiv in self.indivs)
            self.avg_fitness = round(sum/len(self.indivs), 4)
        else:
            print("Error: Population is empty")

        return

    def tournamentSelection(self, participants, selection_size):
        ROUND_SIZE = 5
        ROUND_WINNERS_SIZE = 2
        tournament_winners = []
        while(len(tournament_winners) < selection_size):

            round_participants = random.sample(participants, ROUND_SIZE)
            round_participants.sort(key=lambda x: x.fitness, reverse=True)
            round_winners = round_participants[:ROUND_WINNERS_SIZE]
            tournament_winners.extend(round_winners)

        return tournament_winners

    def parentSelection(self):
        self.sortPopulation()
        self.next_pop_parents = []
        ELITE_COUNT = int(self.POP_SIZE*self.ELITE_FRAC)
        elite_pop = self.indivs[:ELITE_COUNT]
        self.next_pop_parents.extend(elite_pop)
        #Adding random indivs from population to parent pool via tournament selection
        tournament_pop = self.tournamentSelection(self.indivs[ELITE_COUNT:], (self.POP_SIZE*(1 - self.ELITE_FRAC))/2)
        self.next_pop_parents.extend(tournament_pop)

        return

    def getParents(self):
        parent_index1 = 0
        parent_index2 = 0
        parents_size = len(self.next_pop_parents)
        while(parent_index1 == parent_index2):
            parent_index1 = random.randrange(0, parents_size)
            parent_index2 = random.randrange(0, parents_size)
        generator = Individual(params=self.parameters)
        genes1, genes2 = self.next_pop_parents[parent_index1].genes, self.next_pop_parents[parent_index2].genes
        parent1, parent2 = generator.generateIndiv(genes1), generator.generateIndiv(genes2)
        return parent1, parent2

    def generateChildren(self):

        offspring = []
        best_indiv = self.best_indiv
        while( len(offspring) < self.OFFSPRING_SIZE - 1):
            parent1, parent2 = self.getParents()
            seed = random.random()
            ##Crossover
            if(self.CROSSOVER_CHANCE >= seed):
                childs = parent1.crossover(parent2)
            else:
                childs = [parent1, parent2]

            for c in childs:
                ##Mutation
                childsM = []
                seed = random.random()
                if (self.MUTATION_CHANCE >= seed):
                    ci = c.mutation()
                elif(self.RANDOM_INDIV_CHANCE >= seed - (self.MUTATION_CHANCE)):
                    ci = c.generateRandomIndiv()
                else:
                    ci = copy.deepcopy(c)
                ##Calculate Fitness
                ci.computeFitness()

                childsM.append(ci)

            offspring.extend(childsM)


        offspring.append(best_indiv) #Keeping best individual in population
        ## Return new population
        new_pop = Population(params=self.parameters)
        new_pop.indivs = offspring

        return new_pop

    def getRandomIndivSubset(self, indivs, subset_size):
        subset = random.sample(indivs, subset_size)
        return subset

    def getBestIndiv(self):
        generator = Individual(params=self.parameters)
        best_indiv = generator.generateIndiv(self.indivs[0].genes)
        if(self.max_fitness <= self.indivs[0].fitness):
            return best_indiv
        else:
            print("#population is not sorted properly")
            self.sortPopulation()
            return best_indiv

    def applySelectionPressure(self):
        ### Reduce population to POP_SIZE
        self.sortPopulation()
        elite_offspring = self.indivs[:int(self.POP_SIZE/2)]
        remaining_pop = self.indivs[int(self.POP_SIZE/2):]
        selected_pop = []
        selected_pop.extend(elite_offspring)
        remaining_selected = self.getRandomIndivSubset(remaining_pop, int(self.POP_SIZE/2))
        selected_pop.extend(remaining_selected)
        self.indivs = selected_pop
        self.calculateMetrics(sort=True)
