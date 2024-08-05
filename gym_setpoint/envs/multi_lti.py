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
                  "N":250}
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
        # scaling format (A0 if not connected)
        self.format = int(np.random.randint(0,np.log2(self.n)-1))
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
        self.mode = np.random.randint(3)
        self.sys = self.generate_system(self.mode)
        ### First state and all input to predict possible setpoint in t+1
        ## scaling parameter
        d_scaling = np.power(2, self.format)
        N = int(self.N / np.power(2,2*self.format))
        ## state part
        if self.mode == 0 :
            X0 = 2*np.random.random(self.sys.nstates) - 1
        else :
            s_scaling = np.power(2, np.random.randint(0,np.log2(self.n)))
            X0 = 2*np.random.random((self.n//s_scaling, self.n//s_scaling)) - 1
            X0 = np.repeat(np.repeat(X0, s_scaling, axis=1), s_scaling, axis=0)
            X0 = X0[::d_scaling,::d_scaling]
        # reshape and save
        self.X0 = X0.flatten()[:,None]
        ## input part
        in_scaling = np.power(2, np.random.randint(0,np.log2(self.n)))
        U_3D = 2*np.random.random((self.n//in_scaling, self.n//in_scaling, self.n_step)) - 1
        U_3D = np.repeat(np.repeat(U_3D, in_scaling, axis=1), in_scaling, axis=0)
        # smooth input in time (or not)
        self.smooth = np.random.randint(2, dtype=bool)
        d = float(self.isdiffuse)
        U_3D = gaussian_filter(U_3D, (d, d, 1.)) if self.smooth else U_3D
        # scaling input
        U_3Df = zoom(U_3D, (1./d_scaling, 1./d_scaling, 1), order=1) # spline interpolation (anti aliasing)
        # reshape for python-control and save
        U = U_3Df.reshape((N,self.n_step))
        self.U = U_3D.reshape((self.N,self.n_step))
        ## simulate first step
        T = self.T[:2]
        U = U[:,:2]
        if self.mode == 0 :
          T, self.Y, self.X = ct.forced_response(self.sys, T=T, U=U, X0=self.X0, return_x=True)
        else :
          T, self.Y, self.X = ct.input_output_response(self.sys, T=T, U=U, X0=self.X0, return_x=True)
        ## setpoint observation output
        obs = [self.Y, self.X, U, T]
        # return (obs,info)
        info = {}
        return obs, info
    
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

    def step(self,X,T,U_3D):
        # reshaping
        us = U_3D.shape
        if us[2] != len(T) :
          print("[ERROR] input and time doesn't have same dimension..")
        d_scaling = np.power(2, self.format)
        U_3D_ = resize(U_3D, (self.n//d_scaling, self.n//d_scaling, us[2]), anti_aliasing=True)
        N = int(self.N / np.power(2,2*self.format))
        U = U_3D_.reshape((N,us[2]))
        # simulate step
        T, yout, X = ct.input_output_response(self.sys, T=T, U=U, X0=X, return_x=True)
        return T, yout, X

### basic exemple 
if __name__ == '__main__' :
    print(ct.__version__) # 0.9.4
    env = MultiLti()
    observation, info = env.reset()
    Y, X, U, T = observation
    """
    for _ in range(1000):
       action = env.action_space.sample()  # this is where you would insert your policy
       observation, reward, terminated, truncated, info = env.step(action)
    
       if terminated or truncated:
          observation, info = env.reset()
    
    env.close()
    """