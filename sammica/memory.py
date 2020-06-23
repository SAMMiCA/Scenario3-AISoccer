import random
import numpy as np
from collections import deque

class memory():
    def __init__(self, maxsize):
        self.memory = deque()
        self.maxsize = maxsize
        self.count = 0

    def __len__(self):
        return len(self.memory)

    def push(self, data):
        if self.count < self.maxsize:
            self.memory.append(data)
            self.count += 1
        else :
            self.memory.popleft()
            self.memory.append(data)

    def clear(self):
        self.memory = deque()
        self.count = 0

class ReplayBuffer():
    def __init__(self):
        self.buffer_list = [frame_buffer, reward_buffer, action_buffer] # , state_buffer, solution_buffer

    def __len__(self):
        return len(reward_buffer)

    def clear(self):
        for buffer in self.buffer_list :
            buffer.clear()

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            obs_t = frame_buffer.memory[i]
            action = action_buffer.memory[i]
            reward = reward_buffer.memory[i]
            obs_tp1 = frame_buffer.memory[i + 1]
            # done =

            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            # dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1)# , np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self)) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(len(self) - i) % buffer_size for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, self.__len__())
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)
        
buffer_size = 1e6
frame_buffer = memory(buffer_size + 1)
state_buffer = memory(buffer_size)
solution_buffer = memory(buffer_size)
action_buffer = memory(buffer_size)
reward_buffer = memory(buffer_size)
