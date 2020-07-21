import numpy as np

# coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
Z = 2
TH = 3
ACTIVE = 4
TOUCH = 5
BALL_POSSESSION = 6

def get_reward(info, frame, replayBuffer):
    tmp_prev_frame = replayBuffer.frame_buffer.memory[-1]
    #############################################
    _reward = simple_reward(info, frame, tmp_prev_frame)
    #############################################
    return _reward

def simple_reward(info, frame, tmp_prev_frame):
    reward = []

    #goalkeeper - don't leave the penalty area
    goalkeeper = frame.coordinates[MY_TEAM][0]
    if (goalkeeper[ACTIVE]) :
        if (goalkeeper[X] >= -info['field'][X]/2) and (goalkeeper[X] <= -info['field'][X]/2 + info['penalty_area'][X]) and (abs(goalkeeper[Y]) <= info['penalty_area'][Y]/2):
            reward.append(0.5)
        else:
            reward.append(-0.5)
    else:
        reward.append(0)

    #defenders - goto certain location
    defender_1 = frame.coordinates[MY_TEAM][1]
    defender_2 = frame.coordinates[MY_TEAM][2]

    d1g = np.array([2, 0])
    d2g = np.array([-1, 1.5])

    if (defender_1[ACTIVE]):
        reward.append(1 - 0.5*np.linalg.norm(d1g - defender_1[X:Z]))
    else:
        reward.append(0)

    if (defender_2[ACTIVE]):
        reward.append(1 - 0.5*np.linalg.norm(d2g - defender_2[X:Z]))
    else:
        reward.append(0)

    #attackers - 1: follow ball, 2: follow an enemy
    attacker_1 = frame.coordinates[MY_TEAM][3]
    attacker_2 = frame.coordinates[MY_TEAM][4]

    ball = np.array(frame.coordinates[BALL][X:Z])
    opponent = np.array(frame.coordinates[OP_TEAM][3][X:Z])

    if (attacker_1[ACTIVE]):
        reward.append(1 - 0.5*np.linalg.norm(opponent - attacker_1[X:Z]))
    else:
        reward.append(0)

    if (attacker_2[ACTIVE]):
        reward.append(1 - 0.5*np.linalg.norm(ball - attacker_2[X:Z]))
    else:
        reward.append(0)

    return np.array(reward)
