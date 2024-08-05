#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import gymnasium as gym
import numpy as np

from gymnasium import spaces

# Classe principale de l'env avec consigne
class GymWrap(gym.Wrapper):
    def __init__(self,
                 config = {
                     "env":'CartPole-v1',
                     "mode":0,
                     "classic":False,
                     "dim":None,
                     "is_discrete":True,
                     "N_space":3}):
      
      self.env = gym.make(config["env"])
      super().__init__(self.env)

      # Modifie les limites d'arret (all inverse of chosen state) # TO DO
      #self.env.unwrapped.model.jnt_range[0][0] = -np.inf # limite sup
      #self.env.unwrapped.model.jnt_range[0][1] = np.inf #limite inf
      # Gestion de l'action dans un tableau
      self.action_packed = False
      if isinstance(self.env.action_space.sample(),np.ndarray): self.action_packed = True
      # config
      self._max_episode_steps = self.env._max_episode_steps
      self.mode = config["mode"]  # Mode de l'env
      self.taux_r = 0
      self._elapsed_steps = 0
      self.choice_state(config["dim"])
      self.classic = config['classic']
      self.is_discrete = config['is_discrete']
      self.N_space = config['N_space']
      # parameter
      if self._isdiscrete :
            self.action_space = spaces.Discrete(self.N_space)  # {-1, 0, 1} if 3
      else :
            self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32) # [-1;1]
      self.SPRL = None
      # memory
      self.previous_state = None  # Memoire state
      self.previous_action = None # Memoire action
      # observation
      self.define_boundary()
      # define setpoint
      self.set_setpoint()

    ############ Parameter Part
    def signal_generator(self, N, epsilon=0.05):
        a = (1 - 2*epsilon)
        # smoother
        s = np.random.randint(10,N//10)
        # random
        r = np.random.random(N)
        # smooth signal
        smooth = sm.gaussian_filter1d(r, s)
        n = a*(smooth - smooth.min())/(smooth.max() - smooth.min()) + epsilon
        return n

    def define_boundary(self) :
      state_box = self.env.observation_space
      if state_box.high[self.chosen_state]-state_box.low[self.chosen_state] == np.inf:
        # explore limit
        i = 0
        cstate = []
        while i < 2*self._max_episode_steps :
          state, _ = self.env.reset()
          cstate += [state[self.chosen_state]]
          terminated = False
          i += 1
          while not(terminated) :
            action = self.env.action_space.sample()
            state, _, terminated, _, _ = self.env.step(action)
            cstate += [state[self.chosen_state]]
            i+=1
      else : cstate = [state_box.high[self.chosen_state],state_box.low[self.chosen_state]]
      ## calculate boundary
      self.min, self.max = min(cstate), max(cstate)
      self.taux_r = (self.max-self.min)*0.1
      self.setmin, self.setmax = self.min + self.taux_r, self.max - self.taux_r
      # update min/max
      self.min, self.max = self.min - self.taux_r, self.max + self.taux_r # Pourquoi ?
      ## define observation
      if self.classic :
        n = self.env.observation_space.shape[0]
        low_box, high_box = np.array(n*[-np.inf], dtype=np.float32), np.array(n*[np.inf], dtype=np.float32)
        # update
        low_box[self.chosen_state], high_box[self.chosen_state] = self.min, self.max
      else :
        # our approach
        low_box = np.array([-3., self.min, self.min, self.min], dtype=np.float32)
        high_box = np.array([3., self.max, self.max, self.max], dtype=np.float32)
      # set
      self.observation_space = spaces.Box(low=low_box, high=high_box)

    def choice_state(self, dimension=None):
      state_box = self.env.observation_space
      if dimension is None:
          self.chosen_state = np.random.randint(self.env.observation_space.shape[0])
      else :
          self.chosen_state = dimension

    def set_setpoint(self, epsilon=0.05):
        # modes
        if self.mode < 1  :
            self.setpoint = 0
        elif self.mode == 1 :
            self.setpoint = np.random.uniform(self.setmin, self.setmax)
        else :
            consigne = self.signal_generator(self._max_episode_steps+5, epsilon=0.05) # def pour nb max iter
            self.all_setpoint = (self.setmax - self.setmin)*consigne + self.setmin # rescale de  0-1 vers valeurs cibles
            self.setpoint = self.all_setpoint[self._elapsed_steps+1]

    ############ Model Part
    def obs_mode(self, state, init=False):
      # classic, Constant or setpoint
      if self.mode == 2 :
        self.setpoint = self.all_setpoint[self._elapsed_steps+1]
      if init :
        self.previous_state = state[self.chosen_state]
        if self.is_discrete : self.previous_action = 2*(self.action_space.sample() / (self.N_space -1.)) -1.
        else : self.previous_action = self.action_space.sample()
        #if self.action_packed : self.previous_action = self.previous_action[0] # unpack action if is packed
      obs = np.array([self.previous_action,self.previous_state, state[self.chosen_state], self.setpoint])
      self.previous_state = state[self.chosen_state]
      return obs

    ############ Simulation Part
    def reset(self, seed=None, options=None):
      state, _ = self.env.reset()
      if self.classic: obs = state
      else :
        self.SPRL = self.obs_mode(state, init=True)
        obs = self.SPRL
      self._elapsed_steps = 0 # reset steps
      self.set_setpoint() # reset setpoint
      return obs, {}

    def step(self, action):
      #gestion des action
      if self.is_discrete :
        action = 2*(action / (self.N_space -1.)) -1.
      else : action = float(action)
      # gestion des action encapsuler dans un tableau
      if self.action_packed:
        action = [action]
      self._elapsed_steps+=1
      state, reward, terminated, truncated, info = self.env.step(action)
      #print(state)
      self.SPRL = self.obs_mode(state)
      if self.classic: obs = state
      else : obs = self.SPRL
      #gestion de la previous action en fonction de l'etat de action
      if self.action_packed:
        self.previous_action = action[0]
      else : self.previous_action = action
      # mode -1
      var_state = state[self.chosen_state]
      # terminated
      if self.mode>-1 :
        terminated = (var_state < self.setmin-self.taux_r or var_state > self.setmax+self.taux_r)
        if self.setpoint-self.taux_r > var_state or var_state > self.setpoint + self.taux_r:
          reward = 0.1
        else : reward = 1.
      else : terminated = (var_state < self.setmin or var_state > self.setmax)
      return obs, reward, terminated, truncated, info

### basic exemple 
if __name__ == '__main__' :
    pass