class SPDL_DB():
    def __init__(self, n=32, n_step=100, t=10):
        self.n = n
        self.n_step = n_step
        self.t = t
        # init
        self.N = n*n
        self.T = np.linspace(0, t, n_step)
        self.mode = None
        self.sys = None
        # spatial connection
        self.isconnected = False
        self.Qd = None
        self.isdiffuse = False
        self.smooth = True
        # scaling
        self.scaling = None
        self.format = 0

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

    def reset(self, mode=None, scaling=None, c=None, d=None) :
        self.mode = mode if mode != None else np.random.randint(3)
        # one complex system
        if self.mode == 0 :
          self.isconnected = False
          self.isdiffuse = False
          self.sys = ct.rss(states=4, outputs=self.N, inputs=self.N)
        # multiple simple system
        else :
          ss = []
          # parameter
          self.isconnected = c if c != None else np.random.randint(2, dtype=bool)
          # scaling format (A0 if not connected)
          self.format = scaling if scaling != None else int(np.random.randint(0,np.log2(self.n)-1))
          N = int(self.N / np.power(2,2*self.format))
          n = int(self.n / np.power(2,self.format))
          # connection
          if self.isconnected :
            for i in range(N) :
              G = ct.TransferFunction([+1./(2*self.N)],[1]) # Gain loop
              G = ct.LinearIOSystem(G, name=str(N+i), inputs='u', outputs='y')
              ss += [G]
          # random system
          if self.mode == 1 :
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
            self.isdiffuse = d if d != None else np.random.randint(2, dtype=bool)
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
          self.sys = ct.InterconnectedSystem(ss, connections=Q, inplist=inplist, inputs=in_, outlist=outlist, outputs=out_)

    def sim(self, scaling=None, smooth=None) :
        # 2D initial state
        s_scaling = scaling[0] if scaling != None else np.power(2, np.random.randint(0,np.log2(self.n)))
        X0 = 2*np.random.random((self.n//s_scaling, self.n//s_scaling)) - 1
        X0 = np.repeat(np.repeat(X0, s_scaling, axis=1), s_scaling, axis=0)
        ## random input for each time
        in_scaling = scaling[1] if scaling != None else np.power(2, np.random.randint(0,np.log2(self.n)))
        U_3D = 2*np.random.random((self.n//in_scaling, self.n//in_scaling, self.n_step)) - 1
        U_3D = np.repeat(np.repeat(U_3D, in_scaling, axis=1), in_scaling, axis=0)
        # save up scaling
        self.scaling = (s_scaling,in_scaling)
        # smooth input in time (or not)
        self.smooth = smooth if smooth != None else np.random.randint(2, dtype=bool)
        d = float(self.isdiffuse)
        U_3D = gaussian_filter(U_3D, (d, d, 1.)) if self.smooth else U_3D
        # down scaling format
        d_scaling = np.power(2, self.format)
        X0 = X0[::d_scaling,::d_scaling]
        U_3Df = zoom(U_3D, (1./d_scaling, 1./d_scaling, 1), order=1) # spline interpolation (anti aliasing)
        N = int(self.N / np.power(2,2*self.format))
        # reshaping
        X0 = X0.flatten()[:,None]
        U = U_3Df.reshape((N,self.n_step))
        U_ = U_3D.reshape((self.N,self.n_step))
        ## simulate
        if self.mode == 0 :
          T, yout, self.X = ct.forced_response(self.sys, T=self.T, U=U, X0=X0[:self.sys.nstates], return_x=True)
        else :
          T, yout, self.X = ct.input_output_response(self.sys, T=self.T, U=U, X0=X0, return_x=True)
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
          break
        d_scaling = np.power(2, self.format)
        U_3D_ = resize(U_3D, (self.n//d_scaling, self.n//d_scaling, us[2]), anti_aliasing=True)
        N = int(self.N / np.power(2,2*self.format))
        U = U_3D_.reshape((N,us[2]))
        # simulate step
        T, yout, X = ct.input_output_response(self.sys, T=T, U=U, X0=X, return_x=True)
        return T, yout, X

    def animate(self, S, RGB=False):
      fig, ax = plt.subplots()
      plt.axis('off')
      ims, i = [], 0
      for s in S :
        s_2D = s.reshape((self.n,self.n,3)) if RGB else s.reshape((self.n,self.n))
        blur = cv2.blur(s_2D,(3,3))
        im = ax.imshow(s_2D, animated=True)
        if i == 0 : ax.imshow(blur); i+=1
        ims.append([im])
      anim = animation.ArtistAnimation(fig, ims, interval=50, blit=False, repeat=True)
      return anim
