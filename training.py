import torch as T
import torch.nn as nn
import torch.optim as optim
import random 
import numpy as np
from model import DuellingDQN
from utils import ReplayMemory
from utils import Transition

LR_DQN = 5e-4
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')


MEMORY_LEN = 10000
MEMORY_THRESH = 500
BATCH_SIZE = 200

LEARN_AFTER = MEMORY_THRESH
LEARN_EVERY = 3
UPDATE_EVERY = 9

STATE_SPACE = 36
ACTION_SPACE = 3

ACTION_LOW = -1
ACTION_HIGH = 1



GAMMA = 0.9995
TAU = 1e-3


TAU = 1e-3

class DQNAgent():
    def __init__(self, actor_net=DuellingDQN, memory= ReplayMemory()):
        
        self.actor_online = actor_net(STATE_SPACE, ACTION_SPACE).to(DEVICE)
        self.actor_target = actor_net(STATE_SPACE, ACTION_SPACE).to(DEVICE)
        self.actor_target.load_state_dict(self.actor_online.state_dict())
        
        self.memory = memory
        
        self.actor_criterion = nn.MSELoss()
        self.actor_op = optim.Adam(self.actor_online.parameters(), lr=LR_DQN)
        
        self.t_step = 0
        
    def act(self, state, eps=0.):
        self.t_step += 1
        
        state = T.from_numpy(state).float().to(DEVICE).view(1, 1, -1)
        
        self.actor_online.eval()
        with T.no_grad():
            actions = self.actor_online(state)
        self.actor_online.train()
        
        if random.random() > eps:
            act = np.argmax(actions.cpu().data.numpy())
        else:
            act = random.choice(np.arange(ACTION_SPACE))
        return int(act)
    
    def learn(self):
        if len(self.memory) <= MEMORY_THRESH:
            return 0
        ## Sample experiences from the MEMORY
        if self.t_step > LEARN_AFTER and self.t_step % LEARN_EVERY == 0:
            
            
            batch = self.memory.sample(BATCH_SIZE)
            
            states = np.vstack([t.States for t in batch])
            states = T.from_numpy(states).float().to(DEVICE)
            
            actions = np.vstack([t.Actions for t in batch])
            actions = T.from_numpy(actions).float().to(DEVICE)
            
            rewards = np.vstack([t.Rewards for t in batch])
            rewards = T.from_numpy(rewards).float().to(DEVICE)
            
            next_states = np.vstack([t.NextStates for t in batch])
            next_states = T.from_numpy(next_states).float().to(DEVICE)
            
            dones = np.vstack([t.Dones for t in batch])
            dones = T.from_numpy(dones).float().to(DEVICE)

            ## Actor update 
            ## compute next state actions and state values 
            
            next_state_values = self.actor_target(next_states).max(1)[0].unsqueeze(1)
            y = rewards + (1-dones) * GAMMA * next_state_values
            state_values = self.actor_online(states).gather(1, actions.type(T.int64))
            ## td error
            
            
             ## Compute Actor loss
            actor_loss = self.actor_criterion(y, state_values)
             ## Minize Actor loss
                
            self.actor_op.zero_grad()
            actor_loss.backward()
            self.actor_op.step()
            
            if self.t_step % UPDATE_EVERY == 0:
                self.soft_update(self.actor_online, self.actor_target)

    def td_errors(self, y , state_values):
        td_errors = y - state_values
        return td_errors
                
    def soft_update(self, local_model, target_model, tau=TAU):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
              target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)