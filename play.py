#!/usr/bin/env python3

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

from config import parse_args
import numpy as np
import tensorflow as tf
import math
import time
import pickle
import random
import rmaddpg.misc.tf_util as U
from rmaddpg.rmaddpg import _RMADDPGAgentTrainer
from rmaddpg.network import mlp_model, lstm_fc_model

from gym import spaces
from rmaddpg.misc.multi_discrete import MultiDiscrete

TRAINING = True

# reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5
GOALKICK = 6
CORNERKICK = 7
PENALTYKICK = 8
HALFTIME = 9
EPISODE_END = 10

# game_state
STATE_DEFAULT = 0
STATE_KICKOFF = 1
STATE_GOALKICK = 2
STATE_CORNERKICK = 3
STATE_PENALTYKICK = 4

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


def get_lstm_states(_type, trainers):
    if _type == 'p':
        return [agent.p_c for agent in trainers], [agent.p_h for agent in trainers]
    if _type == 'q':
        return [agent.q_c for agent in trainers], [agent.q_h for agent in trainers]
    else:
        raise ValueError("unknown type")

def update_critic_lstm(trainers, obs_n, action_n, p_states):
    obs_n = [o[None] for o in obs_n]
    action_n = [a[None] for a in action_n]
    q_c_n = [trainer.q_c for trainer in trainers]
    q_h_n = [trainer.q_h for trainer in trainers]
    p_c_n, p_h_n = p_states if p_states else [None, None]

    for trainer in trainers:
        q_val, (trainer.q_c, trainer.q_h) = trainer.q_debug['q_values'](*(obs_n + action_n + q_c_n + q_h_n))

def get_trainers(num_agent, obs_shape_n, act_space_n, arglist):
    trainers = []
    trainer = _RMADDPGAgentTrainer

    for i in range(num_agent):
        trainers.append(trainer(
            "agent_%d" % i,  mlp_model, lstm_fc_model, obs_shape_n, act_space_n, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

def create_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

class player(Participant):
    def convert2action(self, action_n, coordinates):
        one_hot_actions = []
        msg = []
        for id in range(self.num_agent):
            one_hot_action = []
            agent_action = action_n[id][0]
            # parse the action array
            control = []
            act = []
            index = 0
            for i in range(len(self.act_discrete)):
                #separate discrete action sets to action numbers
                act.append(np.argmax(agent_action[index:(index+self.act_discrete[i])]))
                one_hot_action += [0 for j in range(self.act_discrete[i])]
                one_hot_action[index+act[i]] = 1
                index += self.act_discrete[i]

            # drive action number to actual wheel speeds
            if act[0] == 0:
                control += [0, 0]
            elif act[0] == 1: #up
                control += self.go_to(0, 2, coordinates[MY_TEAM][id][TH], self.max_linear_velocity[id])
            elif act[0] == 2: #right, up
                control += self.go_to(1.414, 1.414, coordinates[MY_TEAM][id][TH], self.max_linear_velocity[id])
            elif act[0] == 3: #right
                control += self.go_to(2, 0, coordinates[MY_TEAM][id][TH], self.max_linear_velocity[id])
            elif act[0] == 4: #right, down
                control += self.go_to(1.414, -1.414, coordinates[MY_TEAM][id][TH], self.max_linear_velocity[id])
            elif act[0] == 5: #down
                control += self.go_to(0, -2, coordinates[MY_TEAM][id][TH], self.max_linear_velocity[id])
            elif act[0] == 6: #left, down
                control += self.go_to(-1.414, -1.414, coordinates[MY_TEAM][id][TH], self.max_linear_velocity[id])
            elif act[0] == 7: #left
                control += self.go_to(-2, 0, coordinates[MY_TEAM][id][TH], self.max_linear_velocity[id])
            elif act[0] == 8: #left, up
                control += self.go_to(-1.141, 1.414, coordinates[MY_TEAM][id][TH], self.max_linear_velocity[id])
            else:
                self.printConsole("PANIC!")

            # kick action switch
            if act[1] == True:
                control += [5, 0]
            else:
                control += [0, 0]

            # jump action switch
            if act[2] == True:
                control += [5]
            else:
                control += [0]

            msg += control
            one_hot_actions.append([one_hot_action])

        return msg, one_hot_actions


    def go_to(self, dx, dy, th, maxvel):
        scale, mult_lin, mult_ang = 1.4, 3.5, 0.4
        damping = 0.35
        ka = 0
        sign = 1
        # calculate how far the target position is from the robot
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        # calculate how much the direction is off
        desired_th = (math.pi / 2) if (dx == 0 and dy == 0) else math.atan2(dy, dx)
        d_th = desired_th - th
        while (d_th > math.pi):
            d_th -= 2 * math.pi
        while (d_th < -math.pi):
            d_th += 2 * math.pi

        # based on how far the target position is, set a parameter that
        # decides how much importance should be put into changing directions
        # farther the target is, less need to change directions fastly
        if (d_e > 1):
            ka = 17 / 90
        elif (d_e > 0.5):
            ka = 19 / 90
        elif (d_e > 0.3):
            ka = 21 / 90
        elif (d_e > 0.2):
            ka = 23 / 90
        else:
            ka = 25 / 90

        # if the target position is at rear of the robot, drive backward instead
        if (d_th > math.radians(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < math.radians(-95)):
            d_th += math.pi
            sign = -1

        # if the direction is off by more than 85 degrees,
        # make a turn first instead of start moving toward the target
        if (abs(d_th) > math.radians(85)):
            left_wheel, right_wheel = self.trim_velocity(-mult_ang * d_th, mult_ang * d_th, maxvel)
        # otherwise
        else:
            # scale the angular velocity further down if the direction is off by less than 40 degrees
            if (d_e < 5 and abs(d_th) < math.radians(40)):
                ka = 0.1
            ka *= 4
            # set the wheel velocity
            # 'sign' determines the direction [forward, backward]
            # 'scale' scales the overall velocity at which the robot is driving
            # 'mult_lin' scales the linear velocity at which the robot is driving
            # larger distance 'd_e' scales the base linear velocity higher
            # 'damping' slows the linear velocity down
            # 'mult_ang' and 'ka' scales the angular velocity at which the robot is driving
            # larger angular difference 'd_th' scales the base angular velocity higher
            left_wheel, right_wheel = self.trim_velocity(sign * scale * (mult_lin * (
                                                1 / (1 + math.exp(-3 * d_e)) - damping) - mult_ang * ka * d_th),
                                    sign * scale * (mult_lin * (
                                                1 / (1 + math.exp(-3 * d_e)) - damping) + mult_ang * ka * d_th), maxvel)
        return [left_wheel, right_wheel]

    def trim_velocity(self, left_wheel, right_wheel, maxvel):
        multiplier = 1

        # wheel velocities need to be scaled so that none of wheels exceed the maximum velocity available
        # otherwise, the velocity above the limit will be set to the max velocity by the simulation program
        # if that happens, the velocity ratio between left and right wheels will be changed that the robot may not execute
        # turning actions correctly.
        if (abs(left_wheel) > maxvel or abs(right_wheel) > maxvel):
            if (abs(left_wheel) > abs(right_wheel)):
                multiplier = maxvel / abs(left_wheel)
            else:
                multiplier = maxvel / abs(right_wheel)

        return left_wheel*multiplier, right_wheel*multiplier


    def get_obs(self, frame):
        state = []
        for item in frame.coordinates[MY_TEAM]:
            state += item[X:Y+1]

        for item in frame.coordinates[OP_TEAM]:
            state += item[X:Y+1]

        state += frame.coordinates[BALL][X:Y+1]

        # x = np.array(x)/self.bound[X]
        # y = np.array(y)/self.bound[Y]
        # th = np.array(th)/math.pi

        # self.pushData(x, self.x_buffer, frame.reset_reason!=NONE)
        # self.pushData(y, self.y_buffer, frame.reset_reason!=NONE)
        # self.pushData(th, self.th_buffer, frame.reset_reason!=NONE)
        #
        # diff = np.concatenate((self.getDiff(self.x_buffer), self.getDiff(self.y_buffer), self.getDiff(self.th_buffer)))

        return np.array([state])

    def init(self, info):
        self.info = info
        self.perception = perceptionSystem(info, TRAINING)
        self.reasoning = reasoningSystem(info, TRAINING)
        self.learning = learningSystem(info, TRAINING)

        self.num_agent = info['number_of_robots']
        self.max_linear_velocity = info['max_linear_velocity']

        x_lim = 0.5 * (info['field'][X] + info['goal_area'][X])
        y_lim = 0.5 * (info['field'][Y] + info['goal_area'][Y])

        if TRAINING :
            buffer_size = 1e6
            self.replayBuffer = ReplayBuffer(buffer_size)

        self.arglist = parse_args()
        arglist = self.arglist
        if arglist.use_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.terminal = False
        self.first = True

        num_thread = 1
        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=num_thread,
            intra_op_parallelism_threads=num_thread)
        self.sess = tf.InteractiveSession(config=tf_config)

        # To make sure that training and testing are based on diff seeds
        if arglist.restore:
            create_seed(np.random.randint(2))
        else:
            create_seed(arglist.seed)

        # Create agent trainers
        self.state_dim_n = self.num_agent * 2 * 2 + 2 # agent coordinates [x, y, z, th], ball coordinate [x, y, z]
        self.obs_shape_n = [(self.state_dim_n,) for i in range(self.num_agent)]

        self.act_discrete = [9, 2, 2] # 0~8: stop + 8 directions, 0~1 kick, 0~1 jump
        self.act_space_n = [MultiDiscrete([[0, self.act_discrete[i] - 1] for i in range(len(self.act_discrete))]) for j in range(self.num_agent)]
        self.trainers = get_trainers(self.num_agent, self.obs_shape_n, self.act_space_n, arglist)
        self.printConsole('Using {} agents'.format(arglist.good_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore or arglist.benchmark:
            self.printConsole('Loading previous state...')
            U.load_state(arglist.load_dir)

        self.episode_rewards = [0.0]  # sum of rewards for all agents
        self.agent_rewards = [[0.0] for _ in range(self.num_agent)]  # individual agent reward
        self.final_ep_rewards = []  # sum of rewards for training curve
        self.final_ep_ag_rewards = []  # agent rewards for training curve
        self.agent_info = [[[]]]  # placeholder for benchmarking info
        self.saver = tf.train.Saver()
        self.train_step = 0
        self.t_start = time.time()
        self.new_episode = True # start of a new episode (used for replay buffer)
        self.start_saving_comm = False

        if arglist.graph:
            self.printConsole("Setting up graph writer!")
            self.writer = tf.summary.FileWriter("../../examples/ai28_player/learning_curves/graph",self.sess.graph)

    def update(self, frame):
        if not frame.end_of_frame:
            return

        if TRAINING :
            if len(self.replayBuffer.frame_buffer) > 0 :
                rew_agent = get_reward(self.info, frame, self.replayBuffer)
                rew_n = np.array([sum(rew_agent)/5 for i in range(self.num_agent)])
                self.replayBuffer.reward_buffer.push(rew_n)
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


        # Closing graph writer
        if self.arglist.graph:
            self.writer.close()

        arglist = self.arglist
        obs = self.get_obs(frame)
        obs_n = [obs for i in range(self.num_agent)]
        # done = True if frame.reset_reason == SCORE_MYTEAM or frame.reset_reason == SCORE_OPPONENT else False
        done = True if frame.reset_reason == HALFTIME or frame.reset_reason == EPISODE_END else False
        done_n = [done for i in range(self.num_agent)]

        if self.first:
            self.first = False
        else:
            # collect experience
            for i, agent in enumerate(self.trainers):
                # do this every iteration
                if arglist.critic_lstm and arglist.actor_lstm:
                    agent.experience(self.prev_obs_n[i], self.action_n[i], rew_n[i],
                                    obs_n[i], done_n[i], # terminal,
                                    self.p_in_c_n[i][0], self.p_in_h_n[i][0],
                                    self.p_out_c_n[i][0], self.p_out_h_n[i][0],
                                    self.q_in_c_n[i][0], self.q_in_h_n[i][0],
                                    self.q_out_c_n[i][0], self.q_out_h_n[i][0], self.new_episode)
                elif arglist.critic_lstm:
                    agent.experience(self.prev_obs_n[i], self.action_n[i], rew_n[i],
                                    obs_n[i], done_n[i], # terminal,
                                    self.q_in_c_n[i][0], self.q_in_h_n[i][0],
                                    self.q_out_c_n[i][0], self.q_out_h_n[i][0],self.new_episode)
                elif arglist.actor_lstm:
                    agent.experience(self.prev_obs_n[i], self.action_n[i], rew_n[i],
                                    obs_n[i], done_n[i], # terminal,
                                    self.p_in_c_n[i][0], self.p_in_h_n[i][0],
                                    self.p_out_c_n[i][0], self.p_out_h_n[i][0],
                                    self.new_episode)
                else:
                    agent.experience(self.prev_obs_n[i], self.action_n[i], rew_n[i],
                                    obs_n[i], done_n[i], # terminal,
                                    self.new_episode)

            # Adding rewards
            if arglist.tracking:
                for i, a in enumerate(self.trainers):
                    a.tracker.record_information("ag_reward", rew_agent[i])

            for i, rew in enumerate(rew_agent):
                self.episode_rewards[-1] += rew/self.num_agent
                self.agent_rewards[i][-1] += rew

            # If an episode was finished, reset internal values
            if done:
                self.new_episode = True
                # reset trainers
                if arglist.actor_lstm or arglist.critic_lstm:
                    for agent in self.trainers:
                        agent.reset_lstm()
                if arglist.tracking:
                    for agent in self.trainers:
                        agent.tracker.reset()
                self.episode_rewards.append(0)
                for a in self.agent_rewards:
                    a.append(0)
                self.agent_info.append([[]])
            else:
                self.new_episode=False

            # increment global step counter
            self.train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    self.agent_info[-1][i].append(info_n['n'])
                if self.train_step > arglist.benchmark_iters and done:
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    self.printConsole('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(self.agent_info[:-1], fp)
                    return
            # otherwise training
            else:
                # update all trainers, if not in display or benchmark mode
                loss = None

                if done and (len(self.episode_rewards) % 10 == 0):
                    self.printConsole("Episodes Seen: {}, entering training...".format(len(self.episode_rewards)))
                    for i in range(150):
                        # get same episode sampling
                        if arglist.sync_sampling:
                            inds = [random.randint(0, len(self.trainers[0].replay_buffer._storage)-1) for i in range(arglist.batch_size)]
                        else:
                            inds = None

                        for agent in self.trainers:
                            # if arglist.lstm:
                            #     agent.preupdate(inds=inds)
                            # else:
                            agent.preupdate(inds)
                        for agent in self.trainers:
                            loss = agent.update(self.trainers)#, self.train_step)
                            if loss is None: continue
                    self.printConsole("Training round done")

                # save model, display training output
                if done and (len(self.episode_rewards) % arglist.save_rate == 0):
                    U.save_state(arglist.save_dir, saver=self.saver)
                    self.printConsole("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        self.train_step, len(self.episode_rewards), np.mean(self.episode_rewards[-arglist.save_rate:]), round(time.time()-self.t_start, 3)))
                    self.printConsole("Agent Rewards: GK {}, D1 {}, D2 {}, F1 {}, F2 {}".format(
                        np.mean(self.agent_rewards[0][-arglist.save_rate:]),
                        np.mean(self.agent_rewards[1][-arglist.save_rate:]),
                        np.mean(self.agent_rewards[2][-arglist.save_rate:]),
                        np.mean(self.agent_rewards[3][-arglist.save_rate:]),
                        np.mean(self.agent_rewards[4][-arglist.save_rate:])))
                    self.t_start = time.time()
                    # Keep track of final episode reward
                    self.final_ep_rewards.append(np.mean(self.episode_rewards[-arglist.save_rate:]))
                    for rew in self.agent_rewards:
                        self.final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                    if self.arglist.tracking:
                        for agent in self.trainers:
                            agent.tracker.save()


        if arglist.actor_lstm:
            # get critic input states
            self.p_in_c_n, self.p_in_h_n = get_lstm_states('p', self.trainers) # num_trainers x 1 x 1 x 64
        if arglist.critic_lstm:
            self.q_in_c_n, self.q_in_h_n = get_lstm_states('q', self.trainers) # num_trainers x 1 x 1 x 64


        # get action
        self.action_n = [agent.action(obs) for agent, obs in zip(self.trainers, obs_n)]
        message, self.action_n = self.convert2action(self.action_n, frame.coordinates)
        if arglist.critic_lstm:
            # get critic output states
            p_states = [self.p_in_c_n, self.p_in_h_n] if arglist.actor_lstm else []
            update_critic_lstm(self.trainers, obs_n, self.action_n, p_states)
            self.q_out_c_n, self.q_out_h_n = get_lstm_states('q', self.trainers) # num_trainers x 1 x 1 x 64
        if arglist.actor_lstm:
            self.p_out_c_n, self.p_out_h_n = get_lstm_states('p', self.trainers) # num_trainers x 1 x 1 x 64

        self.prev_obs_n = obs_n

        # speeds = []
        #
        # for robot_id, action in enumerate(actions):
        #     speeds += action2speed(action)
        self.set_speeds(message)

    def finish(self):
        arglist = self.arglist

        if not os.path.exists("rewards"):
            os.makedirs("rewards")
        rew_file_name = "rewards/" + arglist.commit_num + "_rewards.pkl"
        with open(rew_file_name, 'wb') as fp:
            pickle.dump(self.final_ep_rewards, fp)
        agrew_file_name = "rewards/" + arglist.commit_num + "_agrewards.pkl"
        # agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
        with open(agrew_file_name, 'wb') as fp:
            pickle.dump(self.final_ep_ag_rewards, fp)
        self.printConsole('...Finished total of {} episodes.'.format(len(self.episode_rewards)))

        self.sess.close()

if __name__ == '__main__':
    player = player()
    player.run()
