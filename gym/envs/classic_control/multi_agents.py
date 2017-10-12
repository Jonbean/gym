# -*- coding: utf-8 -*-
#!/user/bin/env python

"""
Multi-agents communication environment implemented by Zheng Cai(Jon Tsai).
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from random import shuffle

logger = logging.getLogger(__name__)

class MultiAgentsEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # ================================================ # 
        # init function for multiagents env
        # ================================================ # 

        self.point_length = 1.0
        self.tau = 0.02  # seconds between state updates
        self.screen_width = 600
        self.screen_height = 400


        # world_width = self.x_threshold*2
        # scale = screen_width/world_width
        self.x_threshold = 3.0
        self.y_threshold = 2.0
        self.theta_threshold = 2 * np.pi / 360.0
        self.world_width = self.x_threshold
        self.scale = self.screen_width / self.world_width
        pi = np.pi

        self.goals_colors = [(.1, .2, .3), (.4, .5, .6), (.8, .5, .8)]

        self.goal_1_position = self.scale * np.array((self.x_threshold * np.random.rand(), self.y_threshold * np.random.rand())) 

        self.goal_2_position = self.scale * np.array((self.x_threshold * np.random.rand(), self.y_threshold * np.random.rand())) 

        # define boundary at which to fail the episode
        self.low = np.array([0, 0, -pi, -pi])
        self.high = np.array([self.x_threshold, self.y_threshold, pi, pi])

        # self.action_space = spaces.MultiDiscrete([ [0,4],[0,2], [0,4], [0,2] ])

        self.vocabulary_size = 20
        self.action_space = spaces.Dict({"agent1_momentum": spaces.Box(low=-pi, high=pi, shape=(1,)), 
            "agent2_momentum": spaces.Box(low=-pi, high=pi, shape=(1,)), 
            "agent1_orientation": spaces.Discrete(3), 
            "agent2_orientation": spaces.Discrete(3),
             "agent1_utterance": spaces.Discrete(self.vocabulary_size), 
             "agent2_utterance": spaces.Discrete(self.vocabulary_size)})

        act_set = ["moving2", "gaze@"]
        goals_positions = [self.goal_1_position, self.goal_2_position]
        self.agnet1_goal = {"who": 1, "action": act_set[np.random.randint(2)], "goal":np.random.randint(2)}

        self.agent2_goal = {"who":0, "action": act_set[np.random.randint(2)], "goal":np.random.randint(2)}

        # self.observation_space = spaces.Box(self.low, self.high)
        # self.observation_space = spaces.Dict({
        #     "agents_position": spaces.Box(low=self.low, high=self.high), 
        #     "agents_utterance":spaces.MultiDiscrete([ [0,self.vocabulary_size],[0, self.vocabulary_size] ])})

        self._seed()
        self.viewer = None
        self.state = None


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def action_map(self, discrete_action):
        x_dot = 0
        y_dot = 0
        speed_scale = 0.02
        if discrete_action == 1:
            x_dot = 1
            y_dot = 0
        elif discrete_action == 2:
            x_dot = -1
            y_dot = 0
        elif discrete_action == 3:
            x_dot = 0
            y_dot = 1
        elif discrete_action == 4:
            x_dot = 0
            y_dot = -1
        else:
            x_dot = 0
            y_dot = 0
        return speed_scale * x_dot, speed_scale * y_dot

    def momentum_map(self, abs_angle):
        speed_scale = 0.02
        x_dot = np.cos(abs_angle)
        y_dot = np.sin(abs_angle)
        return speed_scale * x_dot, speed_scale * y_dot

    def angle_map(self, angle_act):
        """
        three actions of orientation change could be make, 0 stands 
        for focus, 1 stands for turning anticlockwise, 2 stands for
        turning clockwise.
        """
        angle_mom = 0

        if angle_act == 1:
            angle_mom = -1
        elif angle_act == 2:
            angle_mom = 1
        return angle_mom
    def goal_reach_criteria(self, goal):
        agent_pos = None
        goal_pos = None
        criteria = None 

        agent_pos = (self.state["agent1_pos"], self.state["agent2_pos"])[goal["who"]]
        goal_pos = (self.goal_1_position, self.goal_2_position)[goal["goal"]]
        euc_dis = np.linalg.norm(np.array(agent_pos) - np.array(goal_pos))

        if goal["action"] == "moving2":

            if euc_dis <= (self.agent_radius + self.goal_radius):
                criteria = True
            else:
                criteria = False
        else:
            agent_ori = [self.state["agent1_orientation"], self.state["agent2_orientation"]][goal["who"]]

            x_diff = goal_pos[0] - agent_pos[0]
            gaze_angle_diff = np.arccos(x_diff/euc_dis)
            if gaze_angle_diff < 0.1:
                criteria = True
            else:
                criteria = False
        return criteria

    def _step(self, action):
        # ============================= # 
        # the multiagents step #
        # ============================= # 
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        # x1, y1, x2, y2, angle1, angle2 = state[0],state[1], state[2], state[3], state[4], state[5]
        x1, y1 = state["agent1_pos"]
        x2, y2 = state["agent2_pos"]
        angle1 = state["agent1_orientation"]
        angle2 = state["agent2_orientation"]

        # x1_dot, y1_dot = self.action_map(action[0])
        x1_dot, y1_dot = self.momentum_map(action["agent1_momentum"])
        #angle1_act = action[1]
        angle1_act = action["agent1_orientation"]

        # x2_dot, y2_dot = self.action_map(action[2])
        # angle2_act = action[3]
        x2_dot, y2_dot = self.momentum_map(action["agent2_momentum"])
        angle2_act = action["agent2_orientation"]

        # print "action: ", x_dot, y_dot
        # here we recalculate env state
        x1 = x1 + x1_dot
        y1 = y1 + y1_dot

        x2 = x2 + x2_dot
        y2 = y2 + y2_dot

        angle_speed = 0.09
        angle1_mom = self.angle_map(angle1_act)
        angle2_mom = self.angle_map(angle2_act)

        angle1 = angle1 + angle1_mom * angle_speed
        angle2 = angle2 + angle2_mom * angle_speed

        # update the env state with new state variables

        # done is one of the value we need to return to judge whether we should stop

        # internal goal reached criteria 

        criteria1 = self.goal_reach_criteria(self.agnet1_goal)
        criteria2 = self.goal_reach_criteria(self.agent2_goal)

        done =  x1 < -self.x_threshold \
                or x1 > self.x_threshold \
                or y1 < -self.y_threshold \
                or y1 > self.y_threshold \
                or x2 < -self.x_threshold \
                or x2 > self.x_threshold \
                or y2 < -self.y_threshold \
                or y2 > self.y_threshold \
                or criteria1 \
                or criteria2
        done = bool(done)

        # here after we define reward of each step
        reward = -1.0
        # self.state = np.array([x1,y1,x2,y2, angle1, angle2])
        self.state = {"agent1_pos": (x1, y1),
        "agent2_pos": (x2, y2),
        "agent1_orientation": angle1,
        "agent2_orientation": angle2}
        # final return values are observation, reward, done, diag_info
        return self.state, reward, done, {} 

    def _reset(self):
        """
        initialize environment state
        """

        self.state = {"agent1_pos": (self.np_random.uniform(low=0, high=self.x_threshold), self.np_random.uniform(low=0, high=self.y_threshold)),
            "agent2_pos": (self.np_random.uniform(low=0, high=self.x_threshold), self.np_random.uniform(low=0, high=self.y_threshold)),
            "agent1_orientation": self.np_random.uniform(low=-np.pi, high=np.pi),
            "agent2_orientation": self.np_random.uniform(low=-np.pi, high=np.pi)
                      }
        shuffle(self.goals_colors)

        # self.state = (0.0, 0.0)
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        
        self.agent_radius = 20
        self.goal_radius = 8
        point_len = 100
        # if viewer hasn't been created, create one
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            # self.viewer.set_bounds(0, self.x_threshold,0, self.y_threshold)


            # # why off set?
            # axleoffset = cartheight/4.0

            # # create cart object
            # cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            agent1_circle = rendering.make_circle(self.agent_radius)
            agent1_circle.set_color(.5, .5, .5)
            agent1_circle.add_attr(rendering.Transform(translation=(1.5,1.5)))
            # so transform is an independent object
            self.agent1trans = rendering.Transform()
            # then carttrans attribute is added to cart
            agent1_circle.add_attr(self.agent1trans)

            # after cart object is created, we add it into 
            self.viewer.add_geom(agent1_circle)
            # add pointer line

            # startp1x = self.state[0]
            # startp1y = self.state[1]
            # endp1x = startp1x + point_len * np.sin(self.state[4])
            # endp1y = startp1y + point_len * np.cos(self.state[4])
            startp1x = self.state["agent1_pos"][0]
            startp1y = self.state["agent1_pos"][1]
            endp1x = startp1x + point_len * np.sin(self.state["agent1_orientation"])
            endp1y = startp1y + point_len * np.cos(self.state["agent1_orientation"])

            # pole1 = rendering.FilledPolygon([(startp1x,startp1y), (startp1x,startp1y+1), (endp1x,endp1y), (endp1x,endp1y+1)])
            pole1 = rendering.Line(start=(startp1x, startp1y), end=(endp1x, endp1y))
            # pole1 = rendering.PolyLine([(startp1x, startp1y), (endp1x, endp1y)], close=False)
            pole1.set_color(0,0,0)
            self.pole1trans = rendering.Transform()
            pole1.add_attr(self.pole1trans)
            pole1.add_attr(self.agent1trans)
            self.viewer.add_geom(pole1)

            # agent2
            agent2_circle = rendering.make_circle(self.agent_radius)
            agent2_circle.set_color(.8, .5, .5)
            agent2_circle.add_attr(rendering.Transform(translation=(1.5,1.5)))
            # so transform is an independent object
            self.agent2trans = rendering.Transform()
            # then carttrans attribute is added to cart
            agent2_circle.add_attr(self.agent2trans)

            # after cart object is created, we add it into 
            self.viewer.add_geom(agent2_circle)

            # startp2x = self.state[2]
            # startp2y = self.state[3]
            # endp2x = startp2x + point_len * np.sin(self.state[5])
            # endp2y = startp2y + point_len * np.cos(self.state[5])
            startp2x = self.state["agent2_pos"][0]
            startp2y = self.state["agent2_pos"][1]
            endp2x = startp2x + point_len * np.sin(self.state["agent2_orientation"])
            endp2y = startp2y + point_len * np.cos(self.state["agent2_orientation"])

            pole2 = rendering.Line(start = (startp2x, startp2y), end = (endp2x, endp2y))
            pole2.set_color(0,0,0)


            # self.pole2trans = rendering.Transform(rotation=self.state[5])

            self.pole2trans = rendering.Transform(rotation=self.state["agent1_orientation"])
            pole2.add_attr(self.pole2trans)
            pole2.add_attr(self.agent2trans)
            self.viewer.add_geom(pole2)

            # now we define goal1 object
            self.goal_circle1 = rendering.make_circle(self.goal_radius)
            self.goal_circle1.set_color(*(self.goals_colors[0]))
            self.goal_circle1.add_attr(rendering.Transform(translation=self.goal_1_position))
            self.goaltrans = rendering.Transform()
            self.goal_circle1.add_attr(self.goaltrans)
            self.viewer.add_geom(self.goal_circle1)            

            # define goal2 object
            self.goal_circle2 = rendering.make_circle(self.goal_radius)
            self.goal_circle2.set_color(*(self.goals_colors[1]))
            self.goal_circle2.add_attr(rendering.Transform(translation=self.goal_2_position))
            self.goaltrans = rendering.Transform()
            self.goal_circle2.add_attr(self.goaltrans)

            # so transform is an independent object
            # then carttrans attribute is added to cart

            # after cart object is created, we add it into 
            self.viewer.add_geom(self.goal_circle2)            


        if self.state is None: return None

        x1, y1 = self.state["agent1_pos"]
        x2, y2 = self.state["agent2_pos"]
        angle1 = self.state["agent1_orientation"]
        angle2 = self.state["agent2_orientation"]
        # cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        new_x1 = self.scale * x1 
        new_y1 = self.scale * y1 
        new_x2 = self.scale * x2
        new_y2 = self.scale * y2

        self.agent1trans.set_translation(new_x1, new_y1)
        self.agent2trans.set_translation(new_x2, new_y2)
        self.pole1trans.set_rotation(-angle1)
        self.pole2trans.set_rotation(-angle2)
        self.goal_circle1.set_color(*(self.goals_colors[0]))
        self.goal_circle2.set_color(*(self.goals_colors[1]))


        # goal_posx = self.scale * self.goal_1_position[0] 
        # goal_posy = self.scale * self.goal_1_position[1] 
        # self.goaltrans.set_translation(goal_posx, goal_posy)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
