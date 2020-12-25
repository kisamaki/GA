import numpy as np
import copy
import random
import csv
from collections import OrderedDict

class OXgame:

    def __init__(self):
        self.BOARD = np.zeros(9)
        self.continue_ = True
        self.trouble_list = []

    def drawborad(self):
        for i in range(len(self.BOARD)):
            if self.BOARD[i] == 0:
                print(str(i), end="")
            elif self.BOARD[i] == 1:
                print("O", end="")
            elif self.BOARD[i] == 2:
                print("X", end="")
            if (i+1) % 3 == 0:
                print()
        print("**********")

    def setstoneO(self, i):
        if(self.BOARD[i] == 0):
            self.BOARD[i] = 1
            self.trouble_list.append(i)
        else:
            self.trouble_list.append(i)
            return False
            
        
    def setstoneX(self, i):
        if(self.BOARD[i] == 0):
            self.BOARD[i] = 2
            self.trouble_list.append(i)
        else:
            self.trouble_list.append(i)
            return False

    def judge(self, BOARD):
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
    
    def cpu(self):
        select_list = []
        return_ = None
        BOARD_copy = self.BOARD.copy()
        for i in range(len(self.BOARD)):
            if self.BOARD[i] == 0:
                select_list.append(i)
        return_ = select_list[random.randrange(len(select_list))]
        if random.randint(0, 100) > 70:
            for i in select_list:
                BOARD_copy[i] = 1
                if self.judge(BOARD_copy) == 1:
                    return_ = i
                BOARD_copy[i] = 0
            for i in select_list:
                BOARD_copy[i] = 2
                if self.judge(BOARD_copy) == 2:
                    return_ = i
                BOARD_copy[i] = 0
        
        return return_
    
    def cpu2(self):
        select_list = []
        return_ = None
        BOARD_copy = self.BOARD.copy()
        for i in range(len(self.BOARD)):
            if self.BOARD[i] == 0:
                select_list.append(i)
        return_ = select_list[random.randrange(len(select_list))]
        for i in select_list:
            BOARD_copy[i] = 1
            if self.judge(BOARD_copy) == 1:
                return_ = i
            BOARD_copy[i] = 0
        for i in select_list:
            BOARD_copy[i] = 2
            if self.judge(BOARD_copy) == 2:
                return_ = i
            BOARD_copy[i] = 0
        
        return return_
    
    def game_step(self, action):
        self.continue_ = False
        if self.setstoneO(action) == False:
                return -1
        else:
            judge_ = self.judge(self.BOARD)
            if judge_ == 1:
                return 1
            elif judge_ == 3:
                return 0
            if self.setstoneX(self.cpu()) == True:
                return 1
            else:
                judge_ = self.judge(self.BOARD)
                if judge_ == 2:
                    return -1
                elif judge_ == 3:
                    return 0
        self.continue_ = True
        return 0

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
        self.db = dout
        dx = np.dot(dout, self.W.T)
        dout = dout.reshape(1, len(dout))
        self.dw = np.dot(self.x.reshape(len(self.x), 1), dout)
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

class Network:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std= 0.1, alpha=0.2, epsilon=0.1, gamma=0.99, actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.memorys = []
        self.actions = actions
        self.ini_action = None
        self.state = str(observation)
        self.ini_state = str(observation)

        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["B1"])
        self.layers["Sigmoid1"] = Sigmoid()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["B2"])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forword(x)
        return x
    
    def act(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(self.actions))
        else:
            action = np.argmax(self.predict(state))
        self.ini_action = action
        self.ini_state = copy.deepcopy(state)
        # print(action)
        return action
    
    def act_test(self, state):
        action = np.argmax(self.predict(state))
        return action
    
    def loss(self, memory):
        loss = (memory["reward"] + max(self.predict(memory["next_state"])) - self.predict(memory["state"])[self.ini_action])
        return loss
    
    def gradient(self, memory):
        dout = []
        loss = self.loss(memory)
        for i in range(len(actions)):
            if i == memory["action"]:
                dout.append(loss)
            else:
                dout.append(0)
        dout = (self.predict(memory["state"]) - dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backword(dout)
        
        grads = {}
        grads["W1"] = self.layers["Affine1"].dw
        grads["B1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dw
        grads["B2"] = self.layers["Affine2"].db

        return grads

class Memory:
    def __init__(self):
        self.memorys = []

    def input_(self, ini_state, ini_action, next_state, reward):
        memory = {}
        memory["state"] = copy.deepcopy(ini_state)
        memory["action"] = copy.deepcopy(ini_action)
        memory["next_state"] = copy.deepcopy(next_state)
        memory["reward"] = reward
        self.memorys.append(memory)



if __name__ == "__main__":
    NN_INPUT = 9
    NN_OUTPUT = 9
    NN_HIDDEN = 5
    BATCH_SIZE = 10000
    episode = 2
    learnig_rate = 0.1
    game = OXgame()
    ini_state = game.BOARD
    actions = game.BOARD
    target_network = Network(input_size=NN_INPUT, hidden_size=NN_HIDDEN, output_size=NN_OUTPUT, actions= actions, observation= ini_state)
    Qnetwork = Network(input_size=NN_INPUT, hidden_size=NN_HIDDEN, output_size=NN_OUTPUT, actions= actions, observation= ini_state)
    for episode in range(episode):
        target_network.memorys = []
        episode_reward = []
        for i in range(BATCH_SIZE):
            episode_memory = Memory()
            game = OXgame()
            while game.continue_ == True:
                reward = game.game_step(target_network.act(game.BOARD))
                episode_memory.input_(target_network.ini_state, target_network.ini_action, game.BOARD, reward)
            episode_reward.append(reward)

        memorys = random.sample(episode_memory.memorys, BATCH_SIZE)
        for memory in memorys:
            grad = target_network.gradient(memory)
            for key in ("W1", "B1", "W2", "B2"):
                # print(grad[key].shape, target_network.params[key].shape)
                target_network.params[key] -= learnig_rate * grad[key]

        win_ = episode_reward.count(1)
        defate_ =  episode_reward.count(-1)
        draw_ = episode_reward.count(0) 

        print("---第{}世代---".format(episode))
        print("勝率{}: ".format(win_ / len(episode_reward)))
        print("勝利数{}: ".format(win_))
        print("敗北数{}: ".format(defate_))
        print("引き分け数{}: ".format(draw_))
    
    # draw = 0
    # win = 0
    # defate = 0
    # for i in range(1000):
    #     game = OXgame()
    #     while game.continue_ == True:
    #         reward = game.game_step(target_network.act_test(game.BOARD))
    #         if reward == 1:
    #             win += 1
    #         elif reward == -1:
    #             defate +=1
            
    # print(win, defate)
        

    
