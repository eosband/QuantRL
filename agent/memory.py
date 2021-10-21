import random
from collections import namedtuple

########################
# Memory functionality #
########################

# Every transition needs the following five characteristics
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    '''
    A class to easily access memory of previous events and the decisions that were made
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        '''
        Randomly samples [batch_size] events from memory
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
