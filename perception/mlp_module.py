import os

import numpy as np
import torch


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = torch.nn.Linear(44 * 2, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, 64)
        self.out = torch.nn.Linear(64, 3)  # [3.0, 0.1, 0.5]
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        latent = self.layer3(x)
        out = self.out(latent)
        return out, latent


class PerceptionModule(object):
    def __init__(self):
        self.history_length = 3
        self.queue = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MLP().to(self.device)  # [3.0, 0.1, 0.5]
        self.model.load_state_dict(torch.load(os.path.dirname(
            os.path.realpath(__file__)) + '/mlp_perception.pth'))
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    # Reformat received frame to mlp input
    def format_data(self, frame, speeds):
        # Flattens 5x3 agents position data to 1x15
        def flatten_agent_coord(posture): return [pos[i] for pos in posture for i in range(3)]

        cur_ball = frame.coordinates[2]  # Current ball posture 1x3
        cur_spd = [speeds[6*i:6*i+6][j]
                   for i in range(5) for j in range(2)]  # Extract wheel speeds of ally agents, 1x10
        cur_pos = flatten_agent_coord(frame.coordinates[0])  # Current ally posture 1x15
        cur_pos_opp = flatten_agent_coord(frame.coordinates[1])  # Current opponent posture 1x15

        # 44 items: state + ball xyz + ally spd lr wheel + ally pos xyz + opponent pos xyz
        return [frame.game_state] + cur_ball + cur_spd + cur_pos + cur_pos_opp

    def perceive(self, frame, speeds, formatted_frame=None):
        # Convert received frame to mlp input format
        if formatted_frame is None:
            formatted_frame = self.format_data(frame, speeds)

        # A queue is used to keep 3 recent frames to be used for friction recognition
        self.queue.append(formatted_frame)
        if len(self.queue) < self.history_length:
            return None, None
        self.queue = self.queue[-self.history_length:]

        frames = np.asarray(self.queue)
        assert frames.shape[-1] == 44  # each frame should have 44 items

        x = []
        for frame_idx in range(1, self.history_length):  # convert positions to position diffs
            temp_buffer = frames[frame_idx].copy()
            temp_buffer[14:] = frames[frame_idx, 14:] - frames[frame_idx - 1, 14:]
            x.extend(temp_buffer)  # concat

        x = torch.Tensor(x).to(self.device)  # 88 dim
        logit, latent = self.model(x)
        mlp_label = logit.argmax(axis=-1).item()

        return mlp_label, latent


if __name__ == '__main__':
    data_idx = [idx for idx in range(46) if idx != 0 and idx != 2] + [2]  # 0: frame id, 2: friction
    data = np.loadtxt('./test/sample.csv', dtype=np.float32,
                      skiprows=1, usecols=data_idx, delimiter=',')

    MLP = PerceptionModule()

    num_sample = {"3.0": 0, "0.1": 0, "0.5": 0}
    num_correct = {"3.0": 0, "0.1": 0, "0.5": 0}
    for datum in data:
        label = str(datum[-1])
        # sample data is already formatted
        mlp_label, latent = MLP.perceive(None, None, formatted_frame=datum[:-1])
        if mlp_label is None:
            continue
        friction = [3.0, 0.1, 0.5][mlp_label]

        num_sample[label] += 1
        if str(friction) == label:
            num_correct[label] += 1

    sum = 0
    sum_anomaly = 0
    for case in num_sample:
        sum += num_correct[case]
        if case != "3.0":
            sum_anomaly += num_correct[case]

    print("Sample ditribution: {}".format(num_sample))
    print("Correct perception: {}".format(num_correct))
    print("Accuracy: {}".format(sum / data.shape[0]))
    print("Recall: {}".format(sum_anomaly / (num_sample["0.1"] + num_sample["0.5"])))
