#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""
import gymnasium as gym
import numpy as np

from gymnasium import spaces
import scipy.ndimage as sm

# Classe principale de l'env avec consigne
class GymWrap(gym.Wrapper):
    def __init__(self,
                 config = {
                     "env":'CartPole-v1',
                     "mode":2,
                     "classic":False,
                     "dim":None,
                     "is_discrete":False,
                     "N_space":3}):
      
      self.env = gym.make(config["env"])
      super().__init__(self.env)

      # config
      self._max_episode_steps = self.env._max_episode_steps
      self.mode = config["mode"]  # Mode de l'env
      self.taux_r = 0
      self._elapsed_steps = 0
      self.dim = config["dim"]
      self.choice_state(self.dim)
      self.classic = config['classic']
      self.is_discrete = config['is_discrete']
      self.N_space = config['N_space']
      # parameter
      if self.is_discrete :
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
        else : self.previous_action = self.action_space.sample()[0]
      obs = np.array([self.previous_action,self.previous_state, state[self.chosen_state], self.setpoint])
      self.previous_state = state[self.chosen_state]
      return obs

    ############ Simulation Part
    def reset(self, seed=None, options=None):
      state, _ = self.env.reset()
      self.set_setpoint() # reset setpoint
      if self.classic: obs = state
      else :
        if self.dim is None : 
            self.choice_state(self.dim)
            self.define_boundary()
        self.SPRL = self.obs_mode(state, init=True)
        obs = self.SPRL
      self._elapsed_steps = 0 # reset steps
      return obs, {}

    def step(self, action):
      #gestion des action
      if self.is_discrete :
        action = action / (self.N_space -1.)
        # adaptation action wrapper
        if isinstance(env.unwrapped.action_space, spaces.Discrete) :
            wrapper_action = int(np.rint(action*(env.unwrapped.action_space.n-1)))
        else :
            wrapper_action = 2*action - 1
        action = 2*action - 1
      else : 
        action = float(action[0])
        if isinstance(env.unwrapped.action_space, spaces.Discrete) :
            wrapper_action = int(np.rint((action+1)/2*(env.unwrapped.action_space.n-1)))
        else :
            wrapper_action = action
      self._elapsed_steps+=1
      # put action in wrapper
      state, reward, terminated, truncated, info = self.env.step(wrapper_action)
      self.SPRL = self.obs_mode(state)
      if self.classic: obs = state
      else : obs = self.SPRL
      # memory
      self.previous_action = action
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
    from tqdm import tqdm
    env = GymWrap()
    observation, info = env.reset()
    for _ in tqdm(range(500)): 
       action = env.action_space.sample()  # this is where you would insert your policy
       _, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           print(f"[INFO] Reset linearized Gym control env")
           observation, info = env.reset()
    env.close()