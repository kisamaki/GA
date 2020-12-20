import numpy as np
import copy
import random
import csv

def write_csv(file, save_dict):
    file = open('OX_Q.csv', 'w')
    w = csv.writer(file)
    save_row = []
    for key, value in save_dict.items():
        a = []
        a.append(key)
        a.append(value)
        save_row.append(a)
    w.writerows(save_row)
    

class Agent:
    def __init__(self, alpha=0.2, epsilon=0.1, gamma=0.99, actions=None, observation=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reward_history = []
        self.actions = actions
        self.state = str(observation)
        self.ini_state = str(observation)
        self.previous_state = None
        self.previous_action = None
        self.q_values = self._init_Q_values()

    def _init_Q_values(self):
        q_values = {}
        q_values[self.state] = np.repeat(0.0, len(self.actions))
        return q_values
    
    def init_state(self):
        self.previous_state = copy.deepcopy(self.ini_state)
        self.state = copy.deepcopy(self.ini_state)
        return self.state
    
    def act(self):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, len(self.q_values[self.state]))
        else:
            action = np.argmax(self.q_values[self.state])
        
        self.previous_action = action
        return action
    
    def act_test(self, state):
        state = str(state)
        # print(self.q_values[state])
        action = np.argmax(self.q_values[state])
        return action
    
    def ovserve(self, next_state, reward=None):
        next_state = str(next_state)
        if next_state not in self.q_values:
            self.q_values[next_state] = np.repeat(0.0, len(self.actions))
        
        self.previous_state = copy.deepcopy(self.state)
        self.state = next_state

        if reward is not None:
            self.reward_history.append(reward)
            self.learn(reward)
    
    def learn(self, reward):
        q = self.q_values[self.previous_state][self.previous_action]
        max_q = max(self.q_values[self.state])
        self.q_values[self.previous_state][self.previous_action] = ((1 - self.alpha) * q) + (self.alpha * (reward + (self.gamma * max_q)))
    
    def set_qtable(self, qtable):
        self.q_values = {}
        self.q_values = qtable

class OXgame:

    def __init__(self):
        self.BOARD = [0, 0, 0, 0, 0, 0, 0, 0, 0]
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
                return -100 
        else:
            judge_ = self.judge(self.BOARD)
            if judge_ == 1:
                return 10
            elif judge_ == 3:
                return 1
            if self.setstoneX(self.cpu()) == True:
                return 100
            else:
                judge_ = self.judge(self.BOARD)
                if judge_ == 2:
                    return -80
                elif judge_ == 3:
                    return 1
        self.continue_ = True
        return 0

def model_learn():
    episode = 200
    one_game = 1000
    reward_list = []
    game = OXgame()
    ini_state = game.BOARD
    actions = game.BOARD
    OXagent = Agent(actions= actions, observation= ini_state)
    for i in range(episode):
        episode_reward = []
        for _ in range(one_game):
            game = OXgame()
            reward = None
            OXagent.ovserve(game.BOARD, reward)
            while game.continue_ == True:
                reward = game.game_step(OXagent.act())
                # if i == episode-1:
                #     if reward < 0:
                #         print(game.trouble_list)
                #         game.drawborad()
                OXagent.ovserve(game.BOARD, reward)
            episode_reward.append(reward)
        win_ = episode_reward.count(10)
        defate_ =  episode_reward.count(-80) + episode_reward.count(-100)
        draw_ = episode_reward.count(1) 
        reward_list.append(sum(episode_reward))

        print("---第{}世代---".format(i))
        print("勝率{}: ".format(win_ / len(episode_reward)))
        print("勝利数{}: ".format(win_))
        print("敗北数{}: ".format(defate_))
        print("引き分け数{}: ".format(draw_))
    
    write_csv("OX_Q.csv", OXagent.q_values)
    score = []
    for i in range(1000):
        game = OXgame()
        while(game.judge(game.BOARD) == 0):
            if game.setstoneO(OXagent.act_test(game.BOARD)) == False:
                score.append(-1)
                break
            else:
                judge_ = game.judge(game.BOARD)
                if judge_ == 1:
                    score.append(1)
                    break
                elif judge_ == 3:
                    score.append(0)
                    break
            if game.setstoneX(game.cpu2()) == False:
                score.append(1)
                break
            else:
                judge_ = game.judge(game.BOARD)
                if judge_ == 2:
                    score.append(-1)
                    break
                elif judge_ == 3:
                    score.append(0)
                    break
    win_ = score.count(1)
    defate_ =  score.count(-1)
    draw_ = score.count(0) 
    reward_list.append(sum(score))

    print("---学習後結果---")
    print("勝率{}: ".format(win_ / len(score)))
    print("勝利数{}: ".format(win_))
    print("敗北数{}: ".format(defate_))
    print("引き分け数{}: ".format(draw_))

def model_play():
    Q_table = {}
    with open("./OX_Q.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            else:
                row_values = row[1].strip("[").strip("]").split(" ")
                row_values = [a for a in row_values if a != ""]
                values = []
                for row_value in row_values:
                    values.append(float(row_value))
                Q_table[str(row[0])] = values
    
    game = OXgame()
    ini_state = game.BOARD
    actions = game.BOARD
    OXagent = Agent(actions= actions, observation= ini_state)
    OXagent.set_qtable(Q_table)

    while(game.judge(game.BOARD) == 0):
            game.drawborad()
            if game.setstoneO(OXagent.act_test(game.BOARD)) == False:
                game.drawborad()
                print("X win")
                break
            else:
                judge_ = game.judge(game.BOARD)
                if judge_ == 1:
                    game.drawborad()
                    print("O win")
                    break
                elif judge_ == 3:
                    game.drawborad()
                    print("draw")
                    break
            game.drawborad()
            if game.setstoneX(int(input("座標を入力してください: "))) == False:
                game.drawborad()
                print("O win")
                break
            else:
                judge_ = game.judge(game.BOARD)
                if judge_ == 2:
                    game.drawborad()
                    print("X win")
                    break
                elif judge_ == 3:
                    game.drawborad()
                    print("draw")
                    break



if __name__ == "__main__":
    # model_learn()
    model_play()

    
