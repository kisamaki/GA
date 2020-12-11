import random
from decimal import Decimal
import numpy as np

class genom:
    genom_list = None
    evalution = None

    def __init__(self, genom_list, evalution):
        self.genom_list = genom_list
        self.evalution = evalution
    
    def GetGenom(self):
        return self.genom_list
    
    def GetEvalution(self):
        return self.evalution
    
    def SetGenom(self, genom_list):
        self.genom_list = genom_list
    
    def SetEvalution(self, evalution):
        self.evalution = evalution

NN_INPUT = 2
NN_OUTPUT = 1
NN_HIDDEN = 5
W1_LENGTH = NN_INPUT*NN_HIDDEN
W2_LENGTH = NN_HIDDEN*NN_OUTPUT
GENOM_LENGTH = W1_LENGTH + W2_LENGTH + NN_HIDDEN + NN_OUTPUT
B_LENGTH = NN_OUTPUT + NN_HIDDEN
MAX_GENOM_LIST = 500
SELECT_GENOM = 50
INDIVIDUAL_MUTATTION = 0.1
GENOM_MUTATION = 0.1
MAX_GENERATION = 100


def create_genom(length):
    genom_list = []
    for _ in range(length-B_LENGTH):
        genom_list.append(random.random())
    for _ in range(B_LENGTH):
        genom_list.append(np.array(1))
    return genom(genom_list, 0)

def evalution(ga):
    y_list = []
    t_list = np.array([0, 0, 0, 1])
    y_list.append(predict(ga, [0,0]))
    y_list.append(predict(ga, [0,1]))
    y_list.append(predict(ga, [1,0]))
    y_list.append(predict(ga, [1,1]))
    result = 0.5 * np.sum((y_list - t_list)**2)
    return 1 - result

def predict(ga, x):
    # def Relu(x):
    #     return np.maximum(0, x)

    def sigmoid(x):
        return 1 / (1 + np.exp(x))

    genom_list = ga.GetGenom()
    w1 = genom_list[:W1_LENGTH]
    w1 = np.array(w1).reshape(NN_INPUT, NN_HIDDEN)
    w2 = genom_list[W1_LENGTH:W1_LENGTH+W2_LENGTH]
    w2 = np.array(w2).reshape(NN_HIDDEN, NN_OUTPUT)
    b1 = genom_list[W1_LENGTH+W2_LENGTH:GENOM_LENGTH-NN_OUTPUT]
    b2 = genom_list[GENOM_LENGTH-NN_OUTPUT:]
    a1 = np.dot(x, w1) - b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) - b2
    y = sigmoid(a2)
    if y > 0.5:
        y = 1
    else:
        y = 0
    return y

def select(ga, elite_length):
    sort_result = sorted(ga, reverse=True, key = lambda u: u.evalution)
    result = [sort_result.pop(0) for i in range(elite_length)]
    return result

def crossover(ga_one, ga_second):
    genom_list = []
    cross_one = random.randint(0, GENOM_LENGTH)
    cross_second = random.randint(cross_one, GENOM_LENGTH)
    one = ga_one.GetGenom()
    second = ga_second.GetGenom()
    progeny_one = one[:cross_one] + second[cross_one:cross_second] + one[cross_second:]
    progeny_second = second[:cross_one] + one[cross_one:cross_second] + second[cross_second:]
    genom_list.append(genom(progeny_one,0))
    genom_list.append(genom(progeny_second, 0))
    return genom_list

def next_generation_geno_create(ga, ga_elite, ga_progeny):
    next_generation_geno = sorted(ga, reverse=False, key=lambda u: u.evalution)
    for _ in range(0, len(ga_elite) + len(ga_progeny)):
        next_generation_geno.pop(0)
    next_generation_geno.extend(ga_elite)
    next_generation_geno.extend(ga_progeny)
    return next_generation_geno

def mutation(ga):
    ga_list = []
    for i in ga:
        if INDIVIDUAL_MUTATTION > (random.randint(1,100) / Decimal(100)):
            genom_list = []
            for i_ in i.GetGenom():
                if GENOM_MUTATION > (random.randint(1,100) / Decimal(100)):
                    genom_list.append(random.random())
                else:
                    genom_list.append(i_)
            i.SetGenom(genom_list)
            ga_list.append(i)
        else:
            ga_list.append(i)
    return ga_list

if __name__ == '__main__':
    current_generation_individual_group = []
    for _ in range(MAX_GENOM_LIST):
        current_generation_individual_group.append(create_genom(GENOM_LENGTH))
    
    for count_ in range(MAX_GENERATION):
        for i in range(MAX_GENOM_LIST):
            current_generation_individual_group[i].SetEvalution(evalution(current_generation_individual_group[i]))

        elite_genes = select(current_generation_individual_group, SELECT_GENOM)
        progeny_gene = []
        for i in range(0, SELECT_GENOM):
            progeny_gene.extend(crossover(elite_genes[i - 1], elite_genes[i]))
        
        next_generation_individual_group = next_generation_geno_create(current_generation_individual_group, elite_genes, progeny_gene)
        # next_generation_individual_group = mutation(next_generation_individual_group)

        fits = [i.GetEvalution() for i in current_generation_individual_group]
        min_ = min(fits)
        max_ = max(fits)
        avg_ = sum(fits) / len(fits)

        print ("-----第{}世代の結果-----".format(count_))
        for ga_ in current_generation_individual_group:
            print(
                    predict(ga_, [0,0]),
                    predict(ga_, [0,1]),
                    predict(ga_, [1,0]),
                    predict(ga_, [1,1]),
                )

        current_generation_individual_group = next_generation_individual_group