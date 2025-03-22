import numpy as np
from siri.utils.logger import lprint

class mouse_pos_filter:
    def __init__(self, D_MAX=460):
        self.last = None
        self.last_delta = 0
        self.use_last_cnt = 0

        self.D_MAX = D_MAX
    def reset(self): self.last = None

    def __call__(self, *args, **kwds):
        return self.step(*args, **kwds)

    def step(self, pos):
        if self.last is None: self.last = pos

        delta_pos = pos - self.last

        
        if abs(delta_pos) > self.D_MAX:
            self.use_last_cnt += 1
            if self.use_last_cnt > 1: print(end='x'*self.use_last_cnt)
            ret = self.last_delta
        else:
            self.use_last_cnt = 0
            self.last_delta = delta_pos
            ret = delta_pos
        
        self.last = pos

        return ret
        





class mouse_filter:
    DISABLE_FILTER = False

    def __init__(self, MAX=500, D_MAX=400, half=False):
        self.half = half
        self.last = None
        self.use_last_cnt = 0

        self.MAX = MAX
        self.D_MAX = D_MAX
    def reset(self): self.last = None

    def __call__(self, *args, **kwds):
        return self.step(*args, **kwds)

    def step(self, continuous):
        if self.DISABLE_FILTER: return continuous
        if self.last is None: self.last = continuous


        if abs(continuous) > self.MAX or abs(continuous - self.last) > self.D_MAX:
            continuous = self.last
            self.last = self.last * 0.75
            self.use_last_cnt += 1
        else:
            self.last = continuous
            self.use_last_cnt = 0
        
        thre = 650
        if abs(self.last) > thre:
            print(end='\n')
            lprint(self, f"Warning: self.last > {thre}, reset")
            self.last = float(np.clip(self.last, -thre, thre)/3)
        elif self.use_last_cnt > 2:
            print(end='\n')
            lprint(self, "Warning: self.use_last_cnt > 2, set self.last continuous")
            self.last = continuous * 0.8

        
        
        return continuous if (not self.half) else continuous/2
