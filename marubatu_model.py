import random
import numpy as np
from decimal import Decimal
# OXゲーム
NN_INPUT = 9
NN_OUTPUT = 9
NN_HIDDEN = 5
W1_LENGTH = NN_INPUT*NN_HIDDEN
W2_LENGTH = NN_HIDDEN*NN_OUTPUT
GENOM_LENGTH = W1_LENGTH + W2_LENGTH + NN_HIDDEN + NN_OUTPUT
B_LENGTH = NN_OUTPUT + NN_HIDDEN
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
        setstoneX(int(input("設置可能な座標を選択してください: ")))

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

def cpu():
    select_list = []
    for i in range(len(BOARD)):
        if BOARD[i] == 0:
            select_list.append(i)
    return select_list[random.randrange(len(select_list))]

def cpu2():
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

def game(ga):
    while(judge(BOARD) == 0):
        setstoneO(predict(ga, BOARD))
        drawborad()
        judge_ = judge(BOARD)
        if judge_ == 1:
            print("win")
            break
        elif judge_ == 3:
            print("draw")
            break
        # setstoneX(int(input("座標を選択してください: ")))
        setstoneX(cpu2())
        drawborad()
        judge_ = judge(BOARD)
        if judge_ == 2:
            print("defate")
            break
        elif judge_ == 3:
            print("draw")
            break

def game1(ga):
    while(judge(BOARD) == 0):
        setstoneO(int(input("座標を選択してください: ")))
        drawborad()
        judge_ = judge(BOARD)
        if judge_ == 1:
            print("win")
            break
        elif judge_ == 3:
            print("draw")
            break
        setstoneX(cpu2())
        drawborad()
        judge_ = judge(BOARD)
        if judge_ == 2:
            print("defate")
            break
        elif judge_ == 3:
            print("draw")
            break

def create_genom():
    genom_list = [0.2483557790520886, 0.5158326644586578, 0.43008595004987793, 0.10299602669633401, 0.5691201271890649, 0.772681420415241, 0.6284074113188294, 0.9208784214531475, 0.7892923274564989, 0.8856846726912455, 0.737768442119516, 0.7327890725909244, 0.980097307025369, 0.8435152641272508, 0.4260307105199711, 0.9759940238255784, 0.011260826770748134, 0.6216653802157847, 0.30283610810471406, 0.7683005265376543, 0.4568368669193744, 0.884485684045496, 0.2807086230703787, 0.9610791587282975, 0.4175268649902185, 0.5051649797935635, 0.9528414474460882, 0.36756069222183874, 0.6136990695352533, 0.8582749141620961, 0.986406197618324, 0.6661105154922868, 0.39207522284046814, 0.04118747804660616, 0.2736027987772499, 
0.08081969972128333, 0.7334665645434644, 0.3542730705123387, 0.38216505190860206, 0.6098666454482797, 0.9585833570675822, 0.29498678537340006, 0.5395921022644344, 0.030889817805696218, 0.8995983013812966, 0.7655671072170701, 0.4649999898757672, 0.18088190480342425, 0.6862430931712793, 0.17437382509484123, 0.06298408111628873, 0.21654583520053305, 0.5288291523366853, 0.7118299113093888, 0.5026596055170499, 0.003258672606062829, 0.6197603625227946, 0.12768199200409036, 0.1890540363881117, 0.0307434239056793, 0.16676761765473425, 0.22403248938728992, 0.27731308377834807, 0.771445559279192, 0.6516014803416048, 0.20044039977528327, 0.5997321497652943, 0.44084410645307137, 0.6880447059022871, 0.3247992691697902, 0.813936752678525, 0.15787999923236484, 0.02135937333020621, 0.09858115120025046, 0.9218777610782977, 0.5800727069921027, 0.41176473889597776, 0.7776699242510926, 0.4944977555154252, 0.5232845313367175, 0.12266789569333458, 0.7981857015819541, 0.29796360436048264, 0.2803464510927216, 0.5322361909845195, 0.32798352465194813, 0.9961893039931171, 0.24439988032052062, 0.11759997553068624, 0.7929120092432157, 1, 0.698608179825333, 1, 0.5121948209857904, 1, 1, 1, 1, 1, 0.6817691055956624, 1, 0.15736647304267726, 1, 1]
    return genom(genom_list, 0)

if __name__ == '__main__':
    ga = create_genom()
    game(ga)