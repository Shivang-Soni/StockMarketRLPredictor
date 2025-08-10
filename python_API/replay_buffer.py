import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []

    def add(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        return [self.memory[i] for i in indices]

    def size(self):
        return len(self.memory)
