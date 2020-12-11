import random
from decimal import Decimal

class genom:

    genom_list = None
    evaluation = None

    def __init__(self, genom_list, evaluation):
        self.genom_list = genom_list
        self.evaluation = evaluation
    
    def getGenom(self):
        return self.genom_list

    def getEvalution(self):
        return self.evaluation
    
    def setGenom(self, genom_list):
        self.genom_list = genom_list
    
    def setEvalution(self, evaluation):
        self.evaluation = evaluation

GENOM_LENGTH = 50
MAX_GENOM_LIST = 50
SELECT_GENOM = 10
INDIVIDUAL_MUTATTION = 0.1
GENOM_MUTATION = 0.1
MAX_GENERATION = 100

def create_genom(length):
    genom_list = []
    for _ in range(length):
        genom_list.append(random.randint(0, 1))
    return genom(genom_list, 0)

def evaluation(ga):
    genom_total = sum(ga.getGenom())
    return Decimal(genom_total) / Decimal(len(ga.getGenom()))

def select(ga, elite_length):
    sort_result = sorted(ga, reverse=True, key=lambda u: u.evaluation)
    result = [sort_result.pop(0) for i in range(elite_length)]
    return result

def crossover(ga_one, ga_second):
    genom_list = []
    cross_one = random.randint(0, GENOM_LENGTH)
    cross_second = random.randint(cross_one, GENOM_LENGTH)
    one = ga_one.getGenom()
    second = ga_second.getGenom()
    progeny_one = one[:cross_one] + second[cross_one:cross_second] + one[cross_second:]
    progeny_second = second[:cross_one] + one[cross_one:cross_second] + second[cross_second:]
    genom_list.append(genom(progeny_one,0))
    genom_list.append(genom(progeny_second, 0))
    return genom_list

def next_generation_geno_create(ga, ga_elite, ga_progeny):
    next_generation_geno = sorted(ga, reverse=False, key=lambda u: u.evaluation)
    for _ in range(0, len(ga_elite) + len(ga_progeny)):
        next_generation_geno.pop(0)
    next_generation_geno.extend(ga_elite)
    next_generation_geno.extend(ga_progeny)
    return next_generation_geno

def mutation(ga, individual_mutation, genom_mutation):
    ga_list = []
    for i in ga:
        if individual_mutation > (random.randint(0, 100) / Decimal(100)):
            genom_list = []
            for i_ in i.getGenom():
                if genom_mutation > (random.randint(0, 100) / Decimal(100)):
                    genom_list.append(random.randint(0, 1))
                else:
                    genom_list.append(i_)
            i.setGenom(genom_list)
            ga_list.append(i)
        else:
            ga_list.append(i)
    return ga_list

if __name__ == '__main__':

    current_generation_individual_group = []
    for i in range(MAX_GENOM_LIST):
        current_generation_individual_group.append(create_genom(GENOM_LENGTH))
    
    for count_ in range(1, MAX_GENERATION + 1):
        for i in range(MAX_GENOM_LIST):
            evaluation_result = evaluation(current_generation_individual_group[i])
            current_generation_individual_group[i].setEvalution(evaluation_result)

        elite_genes = select(current_generation_individual_group, SELECT_GENOM)
        progeny_gene = []
        for i in range(0, SELECT_GENOM):
            progeny_gene.extend(crossover(elite_genes[i - 1], elite_genes[i]))
        
        next_generation_individual_group = next_generation_geno_create(current_generation_individual_group, elite_genes, progeny_gene)
        next_generation_individual_group = mutation(next_generation_individual_group, INDIVIDUAL_MUTATTION, GENOM_MUTATION)

        fits = [i.getEvalution() for i in current_generation_individual_group]
        min_ = min(fits)
        max_ = max(fits)
        avg_ = sum(fits) / Decimal(len(fits))

        print ("-----第{}世代の結果-----".format(count_))
        print ("  Min:{}".format(min_))
        print ("  Max:{}".format(max_))
        print ("  Avg:{}".format(avg_))

        current_generation_individual_group = next_generation_individual_group
    
    print ("最も優れた個体は{}".format(elite_genes[0].getGenom()))