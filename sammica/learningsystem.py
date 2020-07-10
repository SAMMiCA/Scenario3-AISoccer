KICK = 0
PASS = 1

def action2speed(action_key):
    if action_key == KICK:
        return [0, 0, 9, 9, 0]
    elif action_key == PASS:
        return [0, 0, 0, 0, 0]
    else:
        return [0, 0, 0, 0, 0]

class learningSystem():
    def __init__(self, info, training):
        self.info = info
        self.training = training

        self.batch_size = 1e1
        self.min_replay_buffer_len = 1e2# batch_size * max_episode_len
        #############################################
        # self.trainers = [model for _ in range(info['number_of_robots'])]
        #############################################

    def update(self, frame, replay_buffer, solution = None):
        if self.training :
            if len(replay_buffer) < self.min_replay_buffer_len :
                return self.get(frame, solution)
            #############################################
            # Training
            batch = replay_buffer.sample(int(self.batch_size)) # return [[obs], [act], [reward], [obs_n]]
            # all_experience = self.replay_buffer.collect()

        robotActions = self.get(frame, solution)
        #############################################
        return robotActions # [KICK, PASS, ...]

    def get(self, received_frame, solution) :
        robotActions = [KICK, KICK, KICK, KICK, KICK] #[agent(received_frame, solution) for agent in self.trainers]
        return robotActions
