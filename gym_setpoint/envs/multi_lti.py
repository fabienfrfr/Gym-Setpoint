#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fabienfrfr
"""

import gymnasium as gym
import numpy as np, control as ct
from scipy.ndimage import gaussian_filter, zoom

from gymnasium import spaces
import cv2

class MultiLti(gym.Env):
    def __init__(self, config={
                  "env_mode":None,
                  "reset":True,
                  "n":32,
                  "t":10,
                  "N":250,}
                 ):

        self.config = config
        self.mode = config["env_mode"]
        self.n = config["n"] #spatial
        self.n_step = config["N"] #time
        self.t = config["t"] #time
        # init
        self.N = self.n*self.n # space grid
        self.T = np.linspace(0, self.t, self.n_step)
        self.sys = None
        # spatial connection
        self.isconnected = None
        self.Qd = None
        self.isdiffuse = None
        self.smooth = None
        # scaling
        self.scaling = None
        self.format = None
        # memory
        self.X0 = None
        self.X = None
        self.U = None
        self.Y = None
        # simulation
        self.action_space = spaces.Box(low=-1., high=1., shape=(self.n,self.n), dtype=np.float32) # [-1;1]
        self._elapsed_steps = 0
        # reinforcement learning
        self.multiple_reward = False # Convolution approach

    def diffusion_map(self,N,n):
        # index link
        idx = np.arange(N).reshape((n,n))
        lleft = [str(l) + '.y' for l in np.roll(idx,1,axis=1).flatten()]
        lright = [str(l) + '.y' for l in np.roll(idx,-1,axis=1).flatten()]
        lup = [str(l) + '.y' for l in np.roll(idx,1,axis=0).flatten()]
        ldown = [str(l) + '.y' for l in np.roll(idx,-1,axis=0).flatten()]
        idx = idx.flatten()
        # Local Diffusion
        Q = [[str(i+N)+'.u']+[lleft[i]]+[lright[i]]+[lup[i]]+[ldown[i]] for i in idx]
        Q += [[str(i)+'.u']+[str(i+N)+'.y'] for i in idx]
        return tuple(map(tuple, tuple(Q)))
    
    def generate_system(self, mode):
        N = int(self.N / np.power(2,2*self.format))
        # one complex system
        if mode == 0 :
          self.isconnected = False
          self.isdiffuse = False
          sys = ct.rss(states=4, outputs=N, inputs=N)
        # multiple simple system
        else :
          ss = []
          # parameter
          self.isconnected = np.random.randint(2, dtype=bool)
          n = int(self.n / np.power(2,self.format))
          # connection
          if self.isconnected :
            for i in range(N) :
              G = ct.TransferFunction([+1./(2*self.N)],[1]) # Gain loop
              G = ct.LinearIOSystem(G, name=str(N+i), inputs='u', outputs='y')
              ss += [G]
          # random system
          if mode == 1 :
            for i in range(N) :
              subsys = ct.LinearIOSystem(ct.rss(1), name=str(i), inputs='u', outputs='y')
              ss += [subsys]
          # a/s+1
          else :
            a = 2*np.random.random((n,n)) - 1
            G = cv2.blur(a,(n,n))
            g = G.flatten() #g_2D = G.reshape(G.shape)
            for i in range(N) :
              subsys = ct.LinearIOSystem(ct.TransferFunction(g[i],[1,1]), name=str(i), inputs='u', outputs='y')
              ss += [subsys]
          # interconnect (disjoint or not)
          in_ , out_ = [f'u[{i}]'  for i in range(N)], [f'y[{i}]'  for i in range(N)]
          inplist, outlist = [[f'{i}.u'] for i in range(N)], [[f'{i}.y'] for i in range(N)]
          if self.isconnected :
            self.isdiffuse = np.random.randint(2, dtype=bool)
            if self.isdiffuse : self.Qd = Q = self.diffusion_map(N,n)
            else :
              # randomized
              idx = np.arange(N)
              R = np.random.choice(idx, size=N, replace=False)
              # construct connection without Algebraic loop ([idx!=R])
              Q = [[str(i+N)+'.u']+[str(R[i]) + '.y'] for i in idx[idx!=R]]
              Q += [[str(i)+'.u']+[str(i+N)+'.y'] for i in idx[idx!=R]]
              Q = tuple(map(tuple, tuple(Q)))
          else :
            self.isdiffuse = False
            Q = None
          self.Q = Q
          sys = ct.InterconnectedSystem(ss, connections=Q, inplist=inplist, inputs=in_, outlist=outlist, outputs=out_)
        # complete system
        return sys
    
    def reset(self, seed=None, options=None) :
        self._elapsed_steps = 0
        # scaling format (A0 if not connected)
        self.format = int(np.random.randint(1,np.log2(self.n)-1))
        # generate multiple system
        self.mode = np.random.randint(3)
        self.sys = self.generate_system(self.mode)
        ### First state and all input to predict possible setpoint in t+1
        ## scaling parameter
        scaling = np.power(2, self.format)
        N = int(self.N / np.power(2,2*self.format))
        ## state part
        if self.mode == 0 :
            X0 = 2*np.random.random(self.sys.nstates) - 1
        else :
            X0 = 2*np.random.random((self.n//scaling, self.n//scaling)) - 1
        # reshape and save
        self.X0 = X0.flatten()[:,None]
        ## input part
        U_3D = 2*np.random.random((self.n//scaling, self.n//scaling, self.n_step)) - 1
        U_3D = np.repeat(np.repeat(U_3D, scaling, axis=1), scaling, axis=0)
        # smooth input in time (or not)
        self.smooth = np.random.randint(2, dtype=bool)
        d = float(self.isdiffuse)
        U_3D = gaussian_filter(U_3D, (d, d, 1.)) if self.smooth else U_3D
        # scaling input
        U_3Df = zoom(U_3D, (1./scaling, 1./scaling, 1), order=1) # spline interpolation (anti aliasing)
        # reshape for python-control and save
        self.U = U_3Df.reshape((N,self.n_step))
        #U_ = U_3D.reshape((self.N,self.n_step)) # for obs ?
        ## simulate first step
        T = self.T[:3]
        U = self.U[:,:3]
        if self.mode == 0 :
          _, self.Y, self.X = ct.forced_response(self.sys, T=T, U=U, X0=self.X0, return_x=True)
        else :
          _, self.Y, self.X = ct.input_output_response(self.sys, T=T, U=U, X0=self.X0, return_x=True)
        ## setpoint observation output (u,y,y,y)
        obs = np.concatenate([self.U[:,1][:,None], self.Y], axis=1)
        # rescale
        obs = obs.reshape((self.n//scaling, self.n//scaling, 4))
        obs = np.repeat(np.repeat(obs, scaling, axis=1), scaling, axis=0)
        # update
        self._elapsed_steps += 1
        # return (obs,info)
        info = {}
        return obs, info
    
    def step(self, action):
        done = False
        # scaling parameter
        scaling = np.power(2, self.format)
        N = int(self.N / np.power(2,2*self.format))
        # reshape action
        action = zoom(action[:,:,None], (1./scaling, 1./scaling, 1), order=1)
        action = action.reshape((N,1))
        # matrix reward : EXPERIMENTAL
        expected_action = self.U[:, self._elapsed_steps][:,None]
        reward = expected_action - action #+ 1
        if not(self.multiple_reward) : reward = reward.mean()            
        #return None, reward, True, True, None
        # update input and next action
        T = self.T[self._elapsed_steps:self._elapsed_steps+2]
        self.V = [self.U[self._elapsed_steps][:,None], action, self.U[self._elapsed_steps:,self._elapsed_steps+2][:,None]]
        # calculate
        X = self.X.copy()
        if self.mode == 0 :
          _, Y, self.X = ct.forced_response(self.sys, T=T, U=U, X0=X, return_x=True)
          _, self.Y, X = ct.forced_response(self.sys, T=T, U=self.V, X0=X, return_x=True)
        else :
          _, Y, self.X = ct.input_output_response(self.sys, T=T, U=U, X0=X, return_x=True)
          _, self.Y, X = ct.input_output_response(self.sys, T=T, U=self.V, X0=X, return_x=True)
        self._elapsed_steps += 1
        ## setpoint observation output (u,y,y,y)
        obs = np.concatenate([self.V[:,1][:,None], self.Y], axis=1)
        # rescale
        obs = obs.reshape((self.n//scaling, self.n//scaling, 4))
        obs = np.repeat(np.repeat(obs, scaling, axis=1), scaling, axis=0)
        # return , , ,
        info = {}
        return state, reward, done, done, info
    
    def sim(self) :
        ## simulate
        if self.mode == 0 :
          T, yout, self.X = ct.forced_response(self.sys, T=self.T, U=self.U, X0=self.X0, return_x=True)
        else :
          T, yout, self.X = ct.input_output_response(self.sys, T=self.T, U=self.U, X0=self.X0, return_x=True)
        # Y up reshaping
        yout_ = yout.reshape((self.n//d_scaling, self.n//d_scaling, self.n_step))
        yout_ = np.repeat(np.repeat(yout_, d_scaling, axis=1), d_scaling, axis=0)
        yout_ = yout_.reshape((self.N,self.n_step))
        return U_.T, yout_.T

### basic exemple 
if __name__ == '__main__' :
    print(ct.__version__) # 0.9.4
    env = MultiLti()
    observation, info = env.reset()
    for _ in range(1000):
       action = env.action_space.sample()  # this is where you would insert your policy
       _, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
          break
          #observation, info = env.reset()
    env.close()