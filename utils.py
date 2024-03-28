import random 
from collections import namedtuple 
from collections import deque

MEMORY_LEN = 10000
MEMORY_THRESH = 500
BATCH_SIZE = 200


Transition = namedtuple("Transition", ["States", "Actions", "Rewards", "NextStates", "Dones"])

class ReplayMemory():
    
    def __init__(self, capacity=MEMORY_LEN):
        self.memory = deque(maxlen=capacity)
        
    def store(self, t):
        
        self.memory.append(t)
        
    def sample(self, n):
        a = random.sample(self.memory, n)
        return a
    
    def __len__(self):
        return len(self.memory)
    