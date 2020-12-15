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
    genom_list = [0.7052490478689298, 0.5576289011716948, 0.22230947241900934, 0.4295784543874811, 0.7149421961662451, 0.14310271889244486, 0.15377391980388622, 0.242026757795486, 0.8227750231374573, 0.2746821366752531, 0.5059778829077121, 0.8798857325740351, 0.5732938496839404, 0.5336477493594843, 0.4304617166046333, 0.570691377290625, 0.17097481156664196, 0.8846066975417477, 0.31669029949912486, 0.00401813461107281, 0.9633476657694229, 0.5032459044460272, 0.5254532020439696, 0.8103437877235468, 0.41202870128918534, 0.38808353545064245, 0.5001941455243358, 0.2859988960899198, 0.36653644896791926, 0.5449806290477536, 0.9951724401367521, 0.9219280910372836, 0.6003693097344495, 0.1776575650402602, 0.703525472393269, 0.6575719770334343, 0.4937293720751258, 0.5455798522619807, 0.35907107331010923, 0.5109485529479418, 0.7604227079294251, 0.688667127386157, 0.07562406764186624, 0.3761870662566137, 0.9244763594738055, 0.08894877006679613, 0.31141387148675204, 0.10519027961862926, 0.48065669977130066, 0.5649053130441978, 0.34746642035953546, 0.3882850639934129, 0.15517977765001045, 0.09235917324969711, 0.7764972828743807, 0.2757948831939214, 0.18533095536238942, 0.3991865245983037, 0.04900764343731323, 0.25638311232822353, 0.6581145345352103, 0.07984797906730501, 0.16523690674134084, 0.3082252801940377, 0.0005982611007974148, 0.6072453037880998, 0.3626033900280924, 0.2812064884285421, 0.40247082928420097, 0.31565620874617295, 0.0995599775658097, 0.10471807369289732, 0.6074449394291495, 0.14852308630636746, 0.18178013561678652, 0.71701458855988, 0.32502901776987436, 0.06475921071432256, 0.8613183215855615, 0.27639243871941876, 0.018013035471612926, 0.5874602319557223, 0.5382943867860823, 0.06807988078201954, 0.21798817948048976, 0.9879235955956329, 0.28574224321676545, 0.4374884843026894, 0.1299341933982554, 0.28005100793506577, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return genom(genom_list, 0)

if __name__ == '__main__':
    ga = create_genom()
    game(ga)