import random
import numpy as np
from collections import OrderedDict
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

class Affine:
    def __init__(self, W, B):
        self.W = W
        self.B = B
        self.x = None
        self.dw = None
        self.db = None
    
    def forword(self, x):
        self.x = x
        y = np.dot(x, self.W) + self.B
        return y
    
    def backword(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forword(self, x):
        y = 1 / 1 + np.exp(-x)
        self.out = y
        return y
    
    def backword(self, x):
        y = x * self.out * (1 - self.out)
        return y

class SoftmaxwithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forword(self, x, t):
        def softmax(x):
            c = np.max(x)
            y = np.exp(x-c) / sum(np.exp(x-c))
            return y
        
        def cross_entropy_error(y, t):
            delta = 1e-7
            return -np.sum(t * np.log(y + delta))

        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backword(self, dout):
        dx = (self.y - self.t) / self.t.shape[0]
        return dx

class Network:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std= 0.1):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["B1"])
        self.layers["Sigmoid1"] = Sigmoid()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["B2"])

        self.lastlayer= SoftmaxwithLoss()

    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forword(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forword(y, t)
    
    def accurary(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)

        accurary = np.sum(y == t) / float(x.shape[0])
        return accurary
    
    def gradient(self, x, t):
        self.loss(x, t)
        dout = 1
        dout = self.lastlayer.backword(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backword(dout)
        
        grads = {}
        grads["W1"] = self.layers["Affine1"].dw
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dw
        grads["b2"] = self.layers["Affine2"].db

        return grads


NN_INPUT = 9
NN_OUTPUT = 9
NN_HIDDEN = 5
BATCH_SIZE = 100
GENERATION = 50


def game(ga):
    trouble = 0
    while(judge(BOARD) == 0):
        trouble += 1
        setstoneO(cpu())
        judge_ = judge(BOARD)
        if judge_ == 1:
            if BOARD[4] == 1:
                return 2
            return 1
        elif judge_ == 3:
            return 0
        setstoneX(cpu())
        judge_ = judge(BOARD)
        if judge_ == 2:
            return -100
        elif judge_ == 3:
            return 0

if __name__ == '__main__':
    network = Network(NN_HIDDEN, NN_HIDDEN, NN_OUTPUT)

    for _ in range(GENERATION):
        for _ in range(BATCH_SIZE):
            pass