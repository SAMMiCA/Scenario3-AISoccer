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
import helper

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

class ActionControl:

    def __init__(self, max_linear_velocity):
        self.max_linear_velocity = max_linear_velocity
        self.g = 9.81 # gravity
        self.damping = 0.2 # linear damping
        self.mult_fs = 0.75 
        self.max_kick_speed = 10*self.mult_fs # 7.5 m/s
        self.mult_angle = 5
        self.max_kick_angle = 10*self.mult_angle # 50 degrees

        self.cur_posture = []
        self.cur_posture_opp = []
        self.cur_ball = []
        self.previous_ball = []
        self.reset_reason = NONE

    def update_state(self, cur_posture, cur_ball):
        self.cur_posture = cur_posture
        self.cur_ball = cur_ball

    def go_to(self, robot_id, x, y):
        sign = 1
        kd = 7 if ((robot_id == 1) or (robot_id == 2)) else 5
        ka = 0.3

        tod = 0.005 # tolerance of distance
        tot = math.pi/360 # tolerance of theta

        dx = x - self.cur_posture[robot_id][X]
        dy = y - self.cur_posture[robot_id][Y]
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        desired_th = math.atan2(dy, dx)

        d_th = helper.wrap_to_pi(desired_th - self.cur_posture[robot_id][TH])
        
        if (d_th > helper.degree2radian(90)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.degree2radian(-90)):
            d_th += math.pi
            sign = -1

        if (d_e < tod):
            kd = 0
        if (abs(d_th) < tot):
            ka = 0

        if self.go_fast(robot_id):
            kd *= 5

        left_wheel, right_wheel = helper.set_wheel_velocity(self.max_linear_velocity[robot_id],
                    sign * (kd * d_e - ka * d_th), 
                    sign * (kd * d_e + ka * d_th))

        return left_wheel, right_wheel

    def go_fast(self, robot_id):
        distance2ball = helper.distance(self.cur_ball[X], self.cur_posture[robot_id][X],
                                    self.cur_ball[Y], self.cur_posture[robot_id][Y])
        d_bg = helper.distance(self.cur_ball[X], 3.9,
                                    self.cur_ball[Y], 0)
        d_rg = helper.distance(3.9, self.cur_posture[robot_id][X],
                                    0, self.cur_posture[robot_id][Y])
        
        if (distance2ball < 0.25 and d_rg > d_bg):
            if (self.cur_ball[X] > 3.7 and abs(self.cur_ball[Y]) > 0.5 and abs(self.cur_posture[robot_id][TH]) < 30 * math.pi/180):
                return False
            else:
                return True
        else:
            return False

    def turn_to(self, robot_id, angle):
        ka = 0.5
        tot = math.pi/360

        desired_th = angle
        d_th = helper.wrap_to_pi(desired_th - self.cur_posture[robot_id][TH])

        if (d_th > helper.degree2radian(90)):
            d_th -= math.pi
        elif (d_th < helper.degree2radian(-90)):
            d_th += math.pi

        if (abs(d_th) < tot):
            ka = 0
        
        left_wheel, right_wheel = helper.set_wheel_velocity(self.max_linear_velocity[robot_id],
                                                                    -ka*d_th, ka*d_th)
        
        return left_wheel, right_wheel

    def kick(self, cross, shoot, quickpass):
        # return kick speed and kick angle
        # range: 0-10
        if cross:
            return 10, 7
        elif shoot:
            return 10, 2
        elif quickpass:
            return 5, 0
        else:
            return 0, 0

    def jump(self, flag):
        # return jump speed
        # range: 0-10
        if flag:
            return 10
        else:
            return 0

    def dribble(self, flag):
        # return variable: boolean 0 (disabled) or 1 (enabled)
        if flag:
            return 1
        else:
            return 0