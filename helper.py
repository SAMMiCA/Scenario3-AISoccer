#!/usr/bin/env python3

# Author(s): Taeyoung Kim, Chansol Hong, Luiz Felipe Vecchietti
# Maintainer: Chansol Hong (cshong@rit.kaist.ac.kr)

import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../common')
try:
    from participant import Game, Frame
except ImportError as err:
    print('player_rulebasedB: \'participant\' module cannot be imported:', err)
    raise

import math

#reset_reason
NONE = Game.NONE
GAME_START = Game.GAME_START
SCORE_MYTEAM = Game.SCORE_MYTEAM
SCORE_OPPONENT = Game.SCORE_OPPONENT
GAME_END = Game.GAME_END
DEADLOCK = Game.DEADLOCK
GOALKICK = Game.GOALKICK
CORNERKICK = Game.CORNERKICK
PENALTYKICK = Game.PENALTYKICK
HALFTIME = Game.HALFTIME
EPISODE_END = Game.EPISODE_END

#game_state
STATE_DEFAULT = Game.STATE_DEFAULT
STATE_KICKOFF = Game.STATE_KICKOFF
STATE_GOALKICK = Game.STATE_GOALKICK
STATE_CORNERKICK = Game.STATE_CORNERKICK
STATE_PENALTYKICK = Game.STATE_PENALTYKICK

#coordinates
MY_TEAM = Frame.MY_TEAM
OP_TEAM = Frame.OP_TEAM
BALL = Frame.BALL
X = Frame.X
Y = Frame.Y
Z = Frame.Z
TH = Frame.TH
ACTIVE = Frame.ACTIVE
TOUCH = Frame.TOUCH
BALL_POSSESSION = Frame.BALL_POSSESSION

def distance(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def degree2radian(deg):
    return deg * math.pi / 180

def radian2degree(rad):
    return rad * 180 / math.pi

def wrap_to_pi(theta):
    while (theta > math.pi):
        theta -= 2 * math.pi
    while (theta < -math.pi):
        theta += 2 * math.pi
    return theta

def predict_ball(cur_ball, previous_ball):
    prediction_step = 1
    dx = cur_ball[X] - previous_ball[X]
    dy = cur_ball[Y] - previous_ball[Y]
    predicted_ball = [cur_ball[X] + prediction_step*dx, cur_ball[Y] + prediction_step*dy]
    return predicted_ball

def find_closest_robot(cur_ball, cur_posture, number_of_robots):
    min_idx = 0
    min_distance = 9999.99
    for i in range(number_of_robots):
        measured_distance = distance(cur_ball[X], cur_posture[i][X], cur_ball[Y], cur_posture[i][Y])
        if (measured_distance < min_distance):
            min_distance = measured_distance
            min_idx = i
    if (min_idx == 0):
        idx = 1
    else:
        idx = min_idx
    return idx

def ball_is_own_goal(predicted_ball, field, goal_area):
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + goal_area[X] and
            -goal_area[Y]/2 <= predicted_ball[Y] <= goal_area[Y]/2)

def ball_is_own_penalty(predicted_ball, field, penalty_area):
    return (-field[X]/2 <= predicted_ball[X] <= -field[X]/2 + penalty_area[X] and
    	-penalty_area[Y]/2 <= predicted_ball[Y] <=  penalty_area[Y]/2)

def ball_is_own_field(predicted_ball):
    return (predicted_ball[X] <= 0)

def ball_is_opp_goal(predicted_ball, field, goal_area):
    return (field[X]/2  - goal_area[X] <= predicted_ball[X] <= field[X]/2 and
            -goal_area[Y]/2 <= predicted_ball[Y] <= goal_area[Y]/2)

def ball_is_opp_penalty(predicted_ball, field, penalty_area):
    return (field[X]/2  - penalty_area[X] <= predicted_ball[X] <= field[X]/2 and
            -penalty_area[Y]/2 <= predicted_ball[Y] <= penalty_area[Y]/2)

def ball_is_opp_field(predicted_ball):
    return (predicted_ball[X] > 0)

def get_defense_kick_angle(predicted_ball, field, cur_ball):
    if predicted_ball[X] >= -field[X] / 2:
        x = -field[X] / 2 - predicted_ball[X]
    else:
        x = -field[X] / 2 - cur_ball[X]
    y = predicted_ball[Y]
    return math.atan2(y, abs(x) + 0.00001)

def get_attack_kick_angle(predicted_ball, field):
    x = field[X] / 2 - predicted_ball[X] + 0.00001
    y = predicted_ball[Y]
    angle = math.atan2(y, x)
    return -angle

def direction_angle(self, id, x, y, cur_posture):
    dx = x - cur_posture[id][X]
    dy = y - cur_posture[id][Y]

    return ((math.pi/2) if (dx == 0 and dy == 0) else math.atan2(dy, dx))

def shoot_chance(self, id, cur_posture, ball):
    d2b = distance(ball[X], cur_posture[id][X],
                                ball[Y],cur_posture[id][Y])
    dx = ball[X] - cur_posture[id][X]
    dy = ball[Y] - cur_posture[id][Y]

    gy = self.goal_area[Y]

    if (dx < 0) or (d2b > self.field[Y]/2):
        return False

    y = (self.field[X]/2 - ball[X])*dy/dx + cur_posture[id][Y]

    if (abs(y) < gy/2):
        return True
    elif (ball[X] < 2.5) and (self.field[Y] - gy/2 < abs(y) < self.field[Y] + gy/2):
        return True
    else:
        return False

def ball_coming_toward_robot(id, cur_posture, prev_ball, cur_ball):
    x_dir = abs(cur_posture[id][X] - prev_ball[X]) \
        > abs(cur_posture[id][X] - cur_ball[X])
    y_dir = abs(cur_posture[id][Y] - prev_ball[Y]) \
        > abs(cur_posture[id][Y] - cur_ball[Y])

    # ball is coming closer
    if (x_dir and y_dir):
        return True
    else:
        return False

def set_wheel_velocity(max_linear_velocity, left_wheel, right_wheel):
    ratio_l = 1
    ratio_r = 1

    if (left_wheel > max_linear_velocity or right_wheel > max_linear_velocity):
        diff = max(left_wheel, right_wheel) - max_linear_velocity
        left_wheel -= diff
        right_wheel -= diff
    if (left_wheel < -max_linear_velocity or right_wheel < -max_linear_velocity):
        diff = min(left_wheel, right_wheel) + max_linear_velocity
        left_wheel -= diff
        right_wheel -= diff

    return left_wheel, right_wheel

def printConsole(message):
    print(message)
    sys.__stdout__.flush()

def print_debug_flag(self):
    printConsole('GK:' + str(self.GK.flag))
    printConsole('D1:' + str(self.D1.flag))
    printConsole('D2:' + str(self.D2.flag))
    printConsole('F1:' + str(self.F1.flag))
    printConsole('F2:' + str(self.F2.flag))
    printConsole("--------")