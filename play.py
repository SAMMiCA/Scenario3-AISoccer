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

from sammica import memory, ReplayBuffer
from sammica import perceptionSystem, reasoningSystem, learningSystem
from sammica import action2speed
from sammica.rewards import get_reward

TRAINING = True

class player(Participant):
    def init(self, info):
        self.info = info
        self.perception = perceptionSystem(info, TRAINING)
        self.reasoning = reasoningSystem(info, TRAINING)
        self.learning = learningSystem(info, TRAINING)

        if TRAINING :
            buffer_size = 1e6
            self.replayBuffer = ReplayBuffer(buffer_size)

    def update(self, frame):
        if TRAINING :
            if len(self.replayBuffer.frame_buffer) > 1 :
                self.replayBuffer.reward_buffer.push(get_reward(frame, self.replayBuffer))
            self.replayBuffer.frame_buffer.push(frame)

            state = self.perception.update(frame)
            solution = self.reasoning.update(frame, state)
            actions = self.learning.update(frame, self.replayBuffer, solution)

            self.replayBuffer.state_buffer.push(state)
            self.replayBuffer.solution_buffer.push(solution)
            self.replayBuffer.action_buffer.push(actions)
        else:
            state = self.perception.get(frame)
            solution = self.reasoning.get(frame, state)
            actions = self.learning.get(frame, solution)

        speeds = []
        for robot_id, action in enumerate(actions):
            speeds += action2speed(action)
        self.set_speeds(speeds)

    # def finish(self):
    #     pass

if __name__ == '__main__':
    player = player()
    player.run()
