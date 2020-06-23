import numpy as np
from sammica import frame_buffer

def get_reward(frame, prev_frame = None):
    tmp_prev_frame = frame_buffer.memory[-1] if prev_frame is None else prev_frame
    #############################################
    _reward = simple_reward(frame, tmp_prev_frame)
    #############################################
    return _reward

def simple_reward(next_frame, cur_frame):
    return 0