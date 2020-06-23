#!/usr/bin/python3
# Author(s): Luiz Felipe Vecchietti, Chansol Hong, Inbae Jeong
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Participant
except ImportError as err:
    print('player_skeleton: \'participant\' module cannot be imported:', err)
    raise

from sammica import frame_buffer, state_buffer, solution_buffer, action_buffer, reward_buffer
from sammica import perception, reasoning, learning
from sammica import get_reward, action2speed

TRAINING = False

class player(Participant):
    def init(self, info):
        self.info = info
        if TRAINING :
            perception.init(info, TRAINING)
            reasoning.init(info, TRAINING)
            learning.init(info, TRAINING)

    def update(self, frame):
        if TRAINING :
            if frame_buffer.__len__ > 0 :
                reward_buffer.push(get_reward(frame))
            frame_buffer.push(frame)

            state = perception.update(frame)
            solution = reasoning.update(frame, state)
            actions = learning.update(frame, solution)

            state_buffer.push(state)
            solution_buffer.push(solution)
            action_buffer.push(actions)
        # else:
        actions = learning.get_action(frame)

        speeds = []
        for robot_id, action in enumerate(actions):
            speeds.append(action2speed(action))
        self.set_speeds(speeds)

    # def finish(self):
    #     pass

if __name__ == '__main__':
    player = player()
    player.run()