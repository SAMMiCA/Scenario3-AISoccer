from sammica import ReplayBuffer
from sammica import perception, reasoning

KICK = 0
PASS = 1

def action2speed(action_key):
    if action_key == KICK:
        return [0, 0, 0, 0, 0]
    elif action_key == PASS:
        return [0, 0, 0, 0, 0]
    else:
        return [0, 0, 0, 0, 0]

class learningSystem():
    def __init__(self):
        self.replay_buffer = ReplayBuffer()

    def init(self, info, training):
        self.info = info
        self.training = training
        self.batch_size = 1e1
        self.min_replay_buffer_len = 1e2# batch_size * max_episode_len
        #############################################
        # self.trainers = [model for _ in range(info['number_of_robots'])]
        #############################################

    def update(self, frame, solution = None):
        if self.training :
            if len(self.replay_buffer) < self.min_replay_buffer_len :
                return self.get_action(frame)
            #############################################
            # Training
            batch = self.replay_buffer.sample(int(self.batch_size)) # return [[obs], [act], [reward], [obs_n]]
            # all_experience = self.replay_buffer.collect()

        robotActions = self.get_action(frame)
        #############################################
        return robotActions # [KICK, PASS, ...]

    def get(self, received_frame, solution) :
        robotActions = [agent(received_frame, solution) for agent in self.trainers]
        return robotActions

    def get_action(self, received_frame):
        state = perception.get(received_frame)
        solution = reasoning.get(received_frame, state)
        return self.get(received_frame, solution)

