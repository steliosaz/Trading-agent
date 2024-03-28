from environment import TradingEnvironment
from utils import ReplayMemory
from training import DQNAgent
from model import DuellingDQN
from utils import Transition

import time
import numpy as np

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.9

STATE_SPACE = 36
ACTION_SPACE = 3




start_time = time.time()        

from DataProcessing import eth

##envs
env = TradingEnvironment(eth.train_days)
test_env = TradingEnvironment(eth.trade_day)

## Agent
memory = ReplayMemory()
agent = DQNAgent(actor_net=DuellingDQN, memory=memory)

N_EPISODES = 20 # No of episodes/epochs
scores = []
eps = EPS_START
act_dict = {0:-1, 1:1, 2:0}

te_score_min = -np.Inf

test_decisions = []
train_decisions = []
for episode in range(1, 1+ N_EPISODES):
    start_time1 = time.time()  
    print(f'episode {episode} start', start_time1)
    counter = 0 
    
    episode_score = 0 
    episode_score2 = 0 
    test_score = 0 
    score = 0 
    
    state = env.reset()
    
    state = state.reshape(-1, STATE_SPACE)
    
    while True:
        actions = agent.act(state, eps)
        action = act_dict[actions]

        next_state, reward, done, _ = env.step(action)
        
        next_state = next_state.reshape(-1, STATE_SPACE)
        
        t = Transition(state, actions, reward, next_state, done)
        
        agent.memory.store(t)
        agent.learn()

        state = next_state
        score += reward
        counter += 1

        if done:
            break
    train_decisions.append(_)
    
    episode_score += score
    episode_score2 += (env.store['pnl'][-1])
    
     # Print episode information
    print(f"Episode {episode}: Score: {episode_score}, Counter: {counter}")
    print(f"Episode {episode}: Score2: {episode_score2}, Counter: {counter}")
    
    
    scores.append(episode_score)
    eps = max(EPS_END, EPS_DECAY * eps)
    
    state = test_env.reset()
    done = False
    score_te = 0
    scores_te = [score_te]
    
    test_score = 0
    test_score2 = 0
    
    end_time1 = time.time()
    ex1 = end_time1 - start_time1
    minutes, seconds = divmod(ex1, 60)
    seconds, milliseconds = divmod(seconds, 1)

    print(f"Train {episode}Execution Time: {int(minutes)} minutes, {int(seconds)} seconds")
    while True: 
        
        actions = agent.act(state)
        action = act_dict[actions]
    
        next_state, reward, done, _ = test_env.step(action)
        
        
        next_state = next_state.reshape(-1, STATE_SPACE)
        state = next_state
        score_te += reward
        scores_te.append(score_te)
        if done:
            break
    # print(_['action_store'])        
    test_decisions.append(_)
    test_score += score_te
    test_score2 += (test_env.store['pnl'][-1])
        
    print(f"Episode: {episode}, Train Score: {episode_score:.5f}, Validation Score: {test_score:.5f}")
    print(f"Episode: {episode}, Train Value: ${episode_score2:.5f}, Validation Value: ${test_score2:.5f}", "\n")
    end_time2 = time.time()
    ex2 = end_time2 - start_time1
    minutes, seconds = divmod(ex2, 60)
    seconds, milliseconds = divmod(seconds, 1)
    print(f"Train {episode}Execution Time: {int(minutes)} minutes, {int(seconds)} seconds")
end_time = time.time()

execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")