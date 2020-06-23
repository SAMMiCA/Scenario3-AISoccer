from .memory import memory, ReplayBuffer
from .perceptionsystem import perceptionSystem
from .reasoningsystem import reasoningSystem
from .learningsystem import learningSystem, action2speed
from .rewards import get_reward

buffer_size = 1e6
frame_buffer = memory(buffer_size + 1)
state_buffer = memory(buffer_size)
solution_buffer = memory(buffer_size)
action_buffer = memory(buffer_size)
reward_buffer = memory(buffer_size)

perception : perceptionSystem = perceptionSystem()
reasoning : reasoningSystem = reasoningSystem()
learning : learningSystem = learningSystem()