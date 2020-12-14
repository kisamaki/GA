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
        setstoneO(int(input("設置可能な座標を選択してください: ")))

def judge():
    for i in range(0, 3, 7):
        if (BOARD[i] == 1) and (BOARD[i+1] == 1) and (BOARD[i+2] == 1):
            return 1
        elif (BOARD[i] == 2) and (BOARD[i+1] == 2) and (BOARD[i+2] == 2):
            return 2
    for i in range(0, 1, 2):
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
    while(judge() == 0):
        setstoneO(int(input("座標を選択してください: ")))
        drawborad()
        judge_ = judge()
        if judge_ == 1:
            print("win")
            break
        elif judge_ == 3:
            print("draw")
            break
        setstoneX(predict(ga, BOARD))
        drawborad()
        judge_ = judge()
        if judge_ == 2:
            print("defate")
            break
        elif judge_ == 3:
            print("draw")
            break

def create_genom():
    genom_list =[0.015440922137647806, 0.12722187128871543, 0.8402108455050965, 0.8783699684903482, 0.8799831943933879, 0.7056227043665051, 0.39074045211959696, 0.9598976019361912, 0.20508002153244598, 0.05096279378341084, 0.8281188508120553, 0.6144900441788216, 0.15925129476997035, 0.5367136328162133, 0.7679050508082292, 0.34059536923128797, 0.1838427076747149, 0.3409855807264951, 0.7606481877353878, 0.562344836669109, 0.36415456082730435, 0.7075822937813768, 0.4787907043622235, 0.7419727139231508, 0.9807310754201914, 0.20067560263942508, 0.25134884619068365, 0.7471714778340046, 0.31737968889726365, 0.38109990560182794, 0.2444417045953755, 0.6750476540706384, 0.3442940761316504, 0.2976829027077118, 0.42781399135187126, 0.8161362845711191, 0.7916679891834315, 0.3961266058794223, 0.7309430783999747, 0.42688920397264263, 0.9426374444808989, 0.0718717047896743, 0.2145249308365994, 0.9368871865085393, 0.4169092611793326, 0.9029315209590957, 0.6681174821497877, 0.8316007501978284, 0.15001580252440583, 0.7081515438370489, 0.23646666681852713, 0.06515126821745076, 0.8497389099543871, 0.4540830095823206, 0.46458841594647515, 0.36220560412935376, 0.6645398487232914, 0.922053771101808, 0.8422579924193234, 0.15402001147520494, 0.4694006902360697, 0.983900783755828, 0.1766277934550543, 0.8938084780765476, 0.8782393416875375, 0.5447612949936467, 0.8125379220185001, 0.049298934373920744, 0.3730280346168846, 0.5819609225064692, 0.178924268185238, 0.3305880402394955, 0.6612163240846035, 0.17093951545216113, 0.4320612936126912, 0.25826651852107363, 0.9148463436852607, 0.8975900145109293, 0.4918972479567232, 0.8258358554867542, 0.9557191101227124, 0.8306969438841462, 0.0602646834722923, 0.09324124802785694, 0.2413523995208754, 0.6504282512090418, 0.4121409539902966, 0.44815987180138794, 0.2554686152201131, 0.28100101987718407, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return genom(genom_list, 0)

if __name__ == '__main__':
    ga = create_genom()
    game(ga)