import random
import numpy as np
from decimal import Decimal
# OXゲーム
BOARD = [0, 0, 0, 0, 0, 0, 0, 0, 0]

def drawborad():
    for i in range(len(BOARD)):
        if BOARD[i] == 0:
            print("-", end="")
        elif BOARD[i] == 1:
            print("O", end="")
        elif BOARD[i] == 2:
            print("X", end="")
        if (i+1) % 3 == 0:
            print()
    print("**********")

def setstoneO(i):
    if(BOARD[i] == 0):
        BOARD[i] = 1
    else:
        setstoneO(int(input("設置可能な座標を選択してください: ")))
        
    
def setstoneX(i):
    if(BOARD[i] == 0):
        BOARD[i] = 2
    else:
        setstoneO(int(input("設置可能な座標を選択してください: ")))

def judge(BOARD):
    for i in [0, 3, 6]:
        if (BOARD[i] == 1) and (BOARD[i+1] == 1) and (BOARD[i+2] == 1):
            return 1
        elif (BOARD[i] == 2) and (BOARD[i+1] == 2) and (BOARD[i+2] == 2):
            return 2
    for i in [0, 1, 2]:
        if (BOARD[i] == 1) and (BOARD[i+3] == 1) and (BOARD[i+6] == 1):
            return 1
        elif (BOARD[i] == 2) and (BOARD[i+3] == 2) and (BOARD[i+6] == 2):
            return 2
    if (BOARD[0] == 1) and (BOARD[4] == 1) and (BOARD[8] == 1):
        return 1
    elif (BOARD[0] == 2) and (BOARD[4] == 2) and (BOARD[8] == 2):
        return 2
    if (BOARD[2] == 1) and (BOARD[4] == 1) and (BOARD[6] == 1):
        return 1
    elif (BOARD[2] == 2) and (BOARD[4] == 2) and (BOARD[6] == 2):
        return 2
    judge = True
    for i in range(len(BOARD)):
        if BOARD[i] == 0:
            judge = False
    if judge:
        return 3
    return 0

def cpu():
    select_list = []
    return_ = None
    BOARD_copy = BOARD.copy()
    for i in range(len(BOARD)):
        if BOARD[i] == 0:
            select_list.append(i)
    return_ = select_list[random.randrange(len(select_list))]
    for i in select_list:
        BOARD_copy[i] = 1
        if judge(BOARD_copy) == 1:
            return_ = i
        BOARD_copy[i] = 0
    for i in select_list:
        BOARD_copy[i] = 2
        if judge(BOARD_copy) == 2:
            return_ = i
        BOARD_copy[i] = 0
    
    return return_


class genom:
    genom_list = None
    evalution = None
    def __init__(self, genom_list, evalution):
        self.genom_list = genom_list
        self.evalution = evalution

    def setGenom(self, genom_list):
        self.genom_list = genom_list
    
    def setEvalution(self, evalution):
        self.evalution = evalution
    
    def GetEvalution(self):
        return self.evalution

NN_INPUT = 9
NN_OUTPUT = 9
NN_HIDDEN = 5
W1_LENGTH = NN_INPUT*NN_HIDDEN
W2_LENGTH = NN_HIDDEN*NN_OUTPUT
GENOM_LENGTH = W1_LENGTH + W2_LENGTH + NN_HIDDEN + NN_OUTPUT
B_LENGTH = NN_OUTPUT + NN_HIDDEN
MAX_GENOM_LIST = 100
SELECT_GENOM = 30
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
    result = ga.GetEvalution()
    y = game(ga)
    result += y
    return result


def game(ga):
    trouble = 0
    while(judge(BOARD) == 0):
        trouble += 1
        setstoneO(predict(ga, BOARD))
        judge_ = judge(BOARD)
        if judge_ == 1:
            return 1
        elif judge_ == 3:
            return 0
        setstoneX(cpu())
        judge_ = judge(BOARD)
        if judge_ == 2:
            return -3
        elif judge_ == 3:
            return 0

def predict(ga, x):
    # def Relu(x):
    #     return np.maximum(0, x)
    
    def softmax(x):
        c = np.max(x)
        exp_x = np.exp(x-c)
        return exp_x / np.sum(exp_x)

    def sigmoid(x):
        return 1 / (1 + np.exp(x))

    select_list = []
    for i in range(len(x)):
        if x[i] == 0:
            select_list.append(i)

    genom_list = ga.genom_list
    w1 = genom_list[:W1_LENGTH]
    w1 = np.array(w1).reshape(NN_INPUT, NN_HIDDEN)
    w2 = genom_list[W1_LENGTH:W1_LENGTH+W2_LENGTH]
    w2 = np.array(w2).reshape(NN_HIDDEN, NN_OUTPUT)
    b1 = genom_list[W1_LENGTH+W2_LENGTH:GENOM_LENGTH-NN_OUTPUT]
    b2 = genom_list[GENOM_LENGTH-NN_OUTPUT:]
    a1 = np.dot(x, w1) - b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) - b2
    y = softmax(a2)
    result = np.argmax(y[select_list])
    result = select_list[result]
    return result

def select(ga, elite_length):
    sort_result = sorted(ga, reverse=True, key = lambda u: u.evalution)
    result = [sort_result.pop(0) for i in range(elite_length)]
    return result

def crossover(ga_one, ga_second):
    genom_list = []
    cross_one = random.randint(0, GENOM_LENGTH)
    cross_second = random.randint(cross_one, GENOM_LENGTH)
    one = ga_one.genom_list
    second = ga_second.genom_list
    progeny_one = one[:cross_one] + second[cross_one:cross_second] + one[cross_second:]
    progeny_second = second[:cross_one] + one[cross_one:cross_second] + second[cross_second:]
    genom_list.append(genom(progeny_one,0))
    genom_list.append(genom(progeny_second, 0))
    return genom_list

def next_generation_geno_create(ga, ga_elite, ga_progeny):
    next_generation_geno = sorted(ga, reverse=False, key=lambda u: u.evalution)
    for _ in range(0, len(ga_progeny)):
        next_generation_geno.pop(0)
    next_generation_geno.extend(ga_progeny)
    return next_generation_geno

def mutation(ga):
    ga_list = []
    for i in ga:
        if INDIVIDUAL_MUTATTION > (random.randint(1,100) / Decimal(100)):
            genom_list = []
            for i_ in i.genom_list:
                if GENOM_MUTATION > (random.randint(1,100) / Decimal(100)):
                    genom_list.append(random.random())
                else:
                    genom_list.append(i_)
            i.setGenom(genom_list)
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
            current_generation_individual_group[i].evalution = 0
            for t in range(10):
                BOARD = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                current_generation_individual_group[i].setEvalution(evalution(current_generation_individual_group[i]))

        elite_genes = select(current_generation_individual_group, SELECT_GENOM)
        progeny_gene = []
        for i in range(0, SELECT_GENOM):
            progeny_gene.extend(crossover(elite_genes[i - 1], elite_genes[i]))
        
        next_generation_individual_group = next_generation_geno_create(current_generation_individual_group, elite_genes, progeny_gene)
        next_generation_individual_group = mutation(next_generation_individual_group)

        current_generation_individual_group = sorted(current_generation_individual_group, reverse=True, key=lambda u: u.evalution)
        fits = [i.evalution for i in current_generation_individual_group]
        win_ = 0
        defate_ = 0
        draw_ = 0
        for i in fits:
            if i > 0:
                win_ += 1
            elif i == 0:
                draw_ += 1
            elif i < 0:
                defate_ += 1 

        # win_ = fits.count(1)
        # defate_ = fits.count(-1)
        # draw_ = fits.count(0)
        # avg_ = sum(fits) / len(fits)

        print ("-----第{}世代の結果-----".format(count_))
        print(fits)
        print("勝利数: {}".format(win_))
        print("敗北数: {}".format(defate_))
        print("引き分け数: {}".format(draw_))
        # print("勝率: {}".format(avg_))

        current_generation_individual_group = next_generation_individual_group
        
    print(current_generation_individual_group[0].genom_list)