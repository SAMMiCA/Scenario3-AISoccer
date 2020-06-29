import numpy as np

def get_reward(frame, replayBuffer):
    tmp_prev_frame = replayBuffer.frame_buffer.memory[-1]
    #############################################
    _reward = simple_reward(frame, tmp_prev_frame)
    #############################################
    return _reward

def simple_reward(frame, tmp_prev_frame):
    return 0

