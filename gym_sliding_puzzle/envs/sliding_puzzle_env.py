"""
Simple Puzzle game implemented by Sukwon Choi
"""

import logging
import math
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pyglet
import time
import os

logger = logging.getLogger(__name__)

class SlidingPuzzleEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }
    
    def __init__(self):
        self.size = 3
        self.num_of_tiles = self.size*self.size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete( [[0,8]]*9 + [[-10, 5]] )
        self.shuffle = 3

        # Reward
        self.reward_done = 100
        self.reward_useless = -20
        self.reward_repeat = -10
        self.reward_other = -1

        self._seed()
        self.state = None
        self.window = None	
        self.prev_action = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        board = self.state[:-1]

        # Get blank tile
        for idx, val in enumerate(board):
            if val == self.num_of_tiles-1: break
        blank = idx

        useless_action = False
        if action == 0: # Move Left
            if blank in [0, 3, 6]: useless_action = True
            else: board[blank], board[blank-1] = board[blank-1], board[blank]
        if action == 1: # Move Right
            if blank in [2, 5, 8]: useless_action = True
            else: board[blank], board[blank+1] = board[blank+1], board[blank]
        if action == 2: # Move Up
            if blank in [0, 1, 2]: useless_action = True
            else: board[blank], board[blank-3] = board[blank-3], board[blank]
        if action == 3: # Move Down
            if blank in [6, 7, 8]: useless_action = True
            else: board[blank], board[blank+3] = board[blank+3], board[blank]

        # Check Done Condition
        done =  board == [0, 1, 2, 3, 4, 5, 6, 7, 8]

        # Set Reward
        if done:
            reward = self.reward_done
        elif useless_action:
            reward = self.reward_useless
        elif (action + self.prev_action) in [1, 5]:
            reward = self.reward_repeat
        else:
            reward = self.reward_other

        self.state = board
        self.prev_action = action
        self.state.append(self.prev_action)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.prev_action = -10
        self.state = list(range(self.num_of_tiles))
        self.prev_action = -10
        self.state.append(-10)

        cnt = 0
        while(cnt < self.shuffle or self.state[:-1] == [0, 1, 2, 3, 4, 5, 6, 7, 8]):
            action = int(random.random()*4)
            _, reward, _, _ = self._step(action)
            if reward != self.reward_useless and reward != self.reward_repeat: cnt = cnt + 1
             
        self.prev_action = -10
        self.state[-1] = self.prev_action
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        time.sleep(1)
        print(self.state)
        if close:
            if self.window == None: 
                return
            else:
                self.window.close()
                return 

        tile_size = 88
        window_size = tile_size * self.size
        if self.window == None: # Open window and load images
            self.window = pyglet.window.Window(window_size, window_size)
            script_dir = os.path.dirname(__file__)
            self.original_images = [pyglet.image.load(os.path.join(script_dir, 'res/puzzle_' + str(x+1) + '.png')) for x in list(range(9))]
        
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Reorder images by self.state
        self.images = [self.original_images[i] for i in self.state[:-1]]

        cnt = 0
        for i in range(2, -1, -1):
            for j in range(3):
                self.images[cnt].blit(tile_size*j,tile_size*i)
                cnt = cnt + 1
        
        self.window.flip()
        return
