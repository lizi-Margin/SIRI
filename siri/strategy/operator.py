import math, random, time
import threading
# import uinput
import numpy as np
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KBController

from siri.global_config import GloablStatus
from siri.global_config import GlobalConfig as cfg
from siri.utils.logger import lprint
from siri.utils.sleeper import Sleeper





class KB:
    kb = KBController()
    kb_bt = None

def hit_kb_bt(key):
    KB.kb.press(key)
    KB.kb.release(key)

def press_kb_bt(key):
    if KB.kb_bt is None:
        print(f'[press_kb_bt] {key}')
        KB.kb.press(key)
        KB.kb_bt = key
    elif KB.kb_bt != key:
        print(f'[press_kb_bt] {key}')
        KB.kb.release(KB.kb_bt)
        KB.kb_bt = key
        KB.kb.press(key)

def unpress_kb_bt():
    if KB.kb_bt is None:
        pass
    else:
        print(f'[unpress_kb_bt] unpress {KB.kb_bt}')
        KB.kb.release(KB.kb_bt)
        KB.kb_bt = None

def is_pressing():
    return KB.kb_bt is not None


mouse = MouseController()
def move_mouse(move_x, move_y):
    if abs(move_x) < 0.1 and abs(move_y) < 0.1:
        return  # abort
    mouse.move(move_x, move_y)


# def move_mouse(move_x, move_y):
#    print(f"[move_mouse] {move_x} {move_y}")
    
#     events = (
#         uinput.REL_X,
#         uinput.REL_Y,
#     )
#     with uinput.Device(events) as device:
#         device.emit(uinput.REL_X, move_x)
#         device.emit(uinput.REL_Y, move_y)

def norm(x: float, lower_side: float=-1.0, upper_side: float=1.0):
    if (x > upper_side): x = upper_side
    if (x < lower_side): x = lower_side
    return x

class pid_ctler:
    def __init__(self):
        self.dt = cfg.tick
        self.reset()

    def reset(self):
        self.sum_err_x = 0.
        self.last_err_x = 0.

        self.sum_err_y = 0.
        self.last_err_y = 0.

        self.sum_bound = 50
        self.p_bound = 60
        self.i_bound = 50

        

    def __call__(self,
            err_x, err_y,
            kx_p, kx_i, kx_d,
            ky_p, ky_i, ky_d,
        ) -> list:
        pid_output = np.array([0, 0])

        # x
        # I
        self.sum_err_x += err_x * self.dt
        if self.sum_err_x * err_x < 0: self.sum_err_x = 0.
        # D
        d_err_x = (err_x - self.last_err_x)/self.dt 
        # output
        pid_output[0] = norm(kx_p * err_x, -self.p_bound, self.p_bound) +  norm(kx_d * d_err_x, -self.i_bound, self.i_bound)
        pid_output[0] += kx_i * self.sum_err_x
        # print(round(kx_p * err_x, 0), round(kx_i * self.sum_err_x, 0), round(kx_d * d_err_x, 0))
        self.sum_err_x = norm(self.sum_err_x, -self.sum_bound, self.sum_bound)
        
        

        # y
        # I
        self.sum_err_y += err_y * self.dt
        if self.sum_err_y * err_y < 0: self.sum_err_y = 0.
        # D
        d_err_y = (err_y - self.last_err_y)/self.dt 
        # output
        pid_output[1] = norm(ky_p * err_y, -self.p_bound, self.p_bound)  + norm(ky_d * d_err_y, -self.i_bound, self.i_bound)
        pid_output[1] += ky_i * self.sum_err_y  # 引入积分
        self.sum_err_y = norm(self.sum_err_y, -self.sum_bound, self.sum_bound)


        self.last_err_x =err_x
        self.last_err_y= err_y
        

        return pid_output
    

# class StateHandler():
#     def switch(obs: dict):
#         return None

#     def step(obs: dict):
#         pass

# class search(StateHandler):
#     def switch(obs: dict):
#         if obs != {}: 
#             return 2
#         return None

#     def step(obs: dict):
#         pass

# class search(StateHandler):
#     def switch(obs: dict):
#         if obs != {}: 
#             return 2
#         return None

#     def step(obs: dict):
#         pass

# class StateMachine:
#     def __init__(self):
#         self.state: int = 1
#         self.state_handler: list[StateHandler] = [None]
    
#     def step(self, obs: dict):
#         new_state = self.state_handler[self.state].switch(obs)
#         if new_state is not None:
#             assert 0 < new_state and new_state < len(self.state_handler)
#             self.state = new_state
        
#         self.state_handler[self.state].step(obs)


class Aimer:
    def __init__(self):
        self.pid = pid_ctler()
        self.kx_p, self.kx_i, self.kx_d = 0.175, 0.5, 0.04
        self.ky_p, self.ky_i, self.ky_d = 0.20, 0.5, 0.0125

        self.last_output = np.array([0., 0.])
        self.n_coef = 0.8
    
    def reset(self):
        self.last_output[:] = 0.
        self.pid.reset()

    def calc_error(self, target_x, target_y):
        center_x, center_y = GloablStatus.in_window_center_xy()
        ex = target_x - center_x
        ey = target_y - center_y 
        return ex, ey
    
    def calc_movement(self, ex, ey, inbound):
        if inbound:
            output = self.pid(
                ex, ey,
                self.kx_p, self.kx_i, self.kx_d,
                self.ky_p, self.ky_i, self.ky_d
            )
            self.last_output = output 
        else:
            output = self.pid(
                ex, ey,
                self.kx_p, self.kx_i, 0.,
                self.ky_p, self.ky_i, self.ky_d
            )
            self.last_output = self.n_coef * output + (1. - self.n_coef) * self.last_output

        # return norm(self.last_output[0], -150., 150.), norm(self.last_output[1], -100., 100.)
        return norm(self.last_output[0], -200., 200.), norm(self.last_output[1], -200., 200.)


class StateMachineBase:
    def __init__(self):
        self._fire_start_t_ = None
        self._last_fire_end_t_ = -100.

        self._scope_start_t_ = None
        self._scope_unsync_cnt_ = 0


        self._search_start_t_ = None
        self._last_search_end_t_ = -100.

        self._last_detect_end_t_ = -100.

    @property
    def _scope_t(self):
        assert self._scope_start_t_ is not None
        return time.time() - self._scope_start_t_

    @property
    def _fire_t(self):
        assert self._fire_start_t_ is not None
        return time.time() - self._fire_start_t_

    @property
    def _search_t(self):
        assert self._search_start_t_ is not None
        return time.time() - self._search_start_t_

    @property
    def _last_fire_t(self):
        return time.time() - self._last_fire_end_t_

    @property
    def _last_search_t(self):
        return time.time() - self._last_search_end_t_

    @property
    def _last_detect_t(self):
        return time.time() - self._last_detect_end_t_
  
    def step(self, obs: dict):
        act = {}
        return act

                        
    def _start_scope(self):
        self._scope_start_t_ = time.time()
        mouse.click(Button.right)

    def _start_fire(self):
        self._fire_start_t_ = time.time()
        mouse.press(Button.left)
        # lprint(self, "_start_fire")

    def _start_search(self):
        self._search_start_t_ = time.time()
        lprint(self, "_start_search")
    
    def _end_scope(self):
        mouse.click(Button.right)
        self._scope_start_t_ = None
                
    def _end_fire(self):
        mouse.release(Button.left)
        self._fire_start_t_ = None
        self._last_fire_end_t_ = time.time()

    def _end_search(self):
        self._search_start_t_ = None
        self._last_search_end_t_ = time.time()
        unpress_kb_bt()
        lprint(self, "_end_search")


class StateMachine(StateMachineBase):
    def __init__(self):
        super(StateMachine, self).__init__()
        self.aimer = Aimer()
        self.kb_sm_lesure = KBStateMachine()
        self.kb_sm_fight = KBStateMachine()
        self.kb_sm_lesure.add_key('4', '5', '6')
        self.kb_sm_fight.add_key('g')

        self.USE_MODEL = True
        if self.USE_MODEL:
            from imitation.net import NetActor, LSTMNet
            self.model_tick = 0.1
            self.model = NetActor(LSTMNet).to('cuda')
            self.model.load_model("./imitation_TRAIN/BC/model-LSTMNet-sample=50-pretrained-13856-augft.pt")
            self.model.eval()
            self.model.net.reset()

  
    def step(self, obs: dict):
        assert 'in_scope' in obs
        assert 'deep_frame' in obs
        no_target = (not 'xy' in obs)
        deep_frame = obs['deep_frame']
        frame = obs['frame']
        f = obs['f']
        l = obs['l']
        r = obs['r']
        obs_dict = {
            'in_scope': 1 if obs['in_scope'] else 0,
            'xy': (0, 0,) if no_target else obs['xy'],
            'deep_frame': obs['deep_frame'],
            'f': f,
            'l': l,
            'r': r
        }

        act_dict = {
            'mv_xy': (0, 0,),
            'fire': 0,
            'scope': 0,
            'w': 0,
            'a': 0,
            's': 0,
            'd': 0,
            '4': 0,
            '5': 0,
            '6': 0,
            'g': 0
        }

        if self._last_detect_t > 15:
            SEARCH_T = 15.
            SEARCH_W_T = 2.
        else:
            SEARCH_T = 4.
            SEARCH_W_T = 2.

        rand_int = random.uniform(-1, 1)

        if (self._scope_start_t_ is not None) != obs['in_scope']:
            self._scope_unsync_cnt_ += 1
            if self._scope_unsync_cnt_ > 15:
                mouse.click(Button.right)
                self._scope_unsync_cnt_ = 0
        else:
            self._scope_unsync_cnt_ = 0

        if (self._fire_start_t_ is not None) and self._fire_t > 0.6:
            self._end_fire()

        if (self._scope_start_t_ is not None)\
           and ((self._scope_t > 4.) or ((self._last_fire_t > 1.) and no_target and (self._scope_t > 1.)))\
           and self._fire_start_t_ is None:
            self._end_scope()

        in_search = False
        in_chase = False
        need_slp = False

        if no_target:
            self.aimer.pid.reset()
            ex, ey = self.aimer.calc_error(*GloablStatus.in_window_center_xy())
            mv_x, mv_y = self.aimer.calc_movement(ex, ey, False)
            if random.uniform(0, 1) < 0.02:
                act_dict = self.kb_sm_lesure.step(act_dict)

            if self._last_fire_t > 2 and self._last_search_t > SEARCH_W_T:
                if self.USE_MODEL:
                    assert self.model_tick > cfg.tick
                    slp = Sleeper(tick = self.model_tick - cfg.tick)
                    need_slp = True

                    in_search = True
                    if self._search_start_t_ is None:
                        self._start_search()
                        self.model.net.reset()
                    
                    wasd, xy = self.model.act([frame])
                    limit = 500
                    mv_x, mv_y = norm(xy[0], lower_side=-limit, upper_side=limit), norm(xy[1], lower_side=-limit, upper_side=limit)
                    # unpress_kb_bt()
                    if wasd[0] > 0:
                        act_dict['w'] = 1
                        press_kb_bt('w')
                    elif wasd[1] > 0:
                        act_dict['a'] = 1
                        press_kb_bt('a')
                    elif wasd[2] > 0:
                        act_dict['s'] = 1
                        press_kb_bt('s')
                    elif wasd[3] > 0:
                        act_dict['d'] = 1
                        press_kb_bt('d')
                else:
                    in_search = True
                    if self._search_start_t_ is None:
                        self._start_search()

                        press_kb_bt('w')
                    act_dict['w'] = 1

                    if f:
                        # unpress_kb_bt()
                        # press_kb_bt('w')
                        # hit_kb_bt('w')
                        # press_kb_bt('w')
                        act_dict['w'] = 1
                    elif l:
                        # unpress_kb_bt()
                        # press_kb_bt('a')
                        # hit_kb_bt('a')
                        act_dict['a'] = 1
                        mv_x = -120
                    elif r:
                        # unpress_kb_bt()
                        # press_kb_bt('d')
                        # hit_kb_bt('d')
                        act_dict['d'] = 1
                        mv_x = 120
            else:
                if self._last_search_t > SEARCH_W_T/2 and self._last_detect_t > 4:
                    mv_x = -130 if int(self._last_detect_end_t_) % 2 == 0 else 130
                    mv_x += rand_int * 20


        
        else:
            self._last_detect_end_t_ = time.time()

            w, h = obs['wh']; assert w > 0 and h > 0
            ex, ey = self.aimer.calc_error(*obs['xy'])
            inbound = (abs(ex) < w and abs(ey) < h)
            mv_x, mv_y = self.aimer.calc_movement(ex, ey, inbound)
            if self._fire_start_t_ is not None:
                mv_y += 1.

            
            if inbound:
                if self._scope_start_t_ is not None:
                    if self._fire_start_t_ is None:
                        self._start_fire()
                elif max(abs(w), abs(h)) > 150 :
                    if (self._fire_start_t_ is None) or self._fire_t > 0.2:
                        self._start_fire()
                elif max(abs(w), abs(h)) > 80:
                    if self._scope_start_t_ is None:
                        self._scope_start_t_ = time.time()
                        mouse.click(Button.right)

                    if self._fire_start_t_ is None and time.time() - self._scope_start_t_ > 1.:
                        self._start_fire()
                else:
                    in_chase = True
                    press_kb_bt('w')
            else:
                in_chase = True
                press_kb_bt('w')
            
            if in_chase and random.uniform(0, 1) < 0.08:
                    act_dict = self.kb_sm_fight.step(act_dict)

        
        if (not in_search or self._search_t > SEARCH_T)and self._search_start_t_ is not None:
            self._end_search()
        
        if (not in_chase) and (not in_search) and is_pressing():
            unpress_kb_bt()
        
        if need_slp:
            mv_x, mv_y = mv_x/2, mv_y/2
            move_mouse(mv_x, mv_y)
            slp.sleep()
        move_mouse(mv_x, mv_y)

        

        act_dict['mv_xy'] = (mv_x, mv_y,)
        act_dict['fire'] = 1 if (self._fire_start_t_ is not None) else 0
        act_dict['scope'] = 1 if (self._scope_start_t_ is not None) else 0

        return {
            'obs': obs_dict,
            'act': act_dict
        }




class AgentStateMachine(StateMachineBase):
    def __init__(self):
        super(AgentStateMachine, self).__init__()
        self.aimer = Aimer()
        self.kb_sm_lesure = KBStateMachine()
        self.kb_sm_fight = KBStateMachine()
        self.kb_sm_lesure.add_key('4', '5', '6')
        self.kb_sm_fight.add_key('g')

        self.model_tick = 0.1

        # from imitation.net import NetActor, LSTMNet
        # self.model = NetActor(LSTMNet).to(cfg.device)
        # self.model.load_model("./imitation_TRAIN/BC/model-LSTMNet-nav-old-pure-50000-navft-20000-frtlnavft-40000.pt")
        # self.model.load_model("./imitation_TRAIN/BC/model-LSTMNet-nav-old-pure-50000-navft-45000.pt")
        # self.model.load_model("./imitation_TRAIN/BC/model-LSTMNet-nav-old-pure-50000.pt")

        from imitation_full.net import LSTMNet
        self.model = LSTMNet().to(cfg.device)
        # self.model.load_model("./imitation_TRAIN/BC/model-LSTMB5-nav-pure-pp19-14000-fight-pp19-14123.pt")
        self.model.load_model("./imitation_TRAIN/BC/model-LSTMB5-nav-pure-pp19-14000.pt")
        # self.model.load_model("./imitation_TRAIN/BC/model-full-fight-pp19-1-reuse=1.pt")
        # self.model.load_model("./imitation_TRAIN/BC/model-full-fight-pp19.pt")



        self.model.eval()
        # self.model.net.reset()

        self.in_press = {
            'w': 0,
            'a': 0,
            's': 0,
            'd': 0,
        }


    def step(self, obs: dict):
        assert 'in_scope' in obs
        assert 'deep_frame' in obs
        no_target = (not 'xy' in obs)
        deep_frame = obs['deep_frame']
        frame = obs['frame']
        f = obs['f']
        l = obs['l']
        r = obs['r']
        obs_dict = {
            'in_scope': 1 if obs['in_scope'] else 0,
            'xy': (0, 0,) if no_target else obs['xy'],
            'deep_frame': obs['deep_frame'],
            'f': f,
            'l': l,
            'r': r
        }

        act_dict = {
            'mv_xy': (0, 0,),
            'fire': 0,
            'scope': 0,
            'w': 0,
            'a': 0,
            's': 0,
            'd': 0,
            '4': 0,
            '5': 0,
            '6': 0,
            'g': 0
        }

        if self._last_detect_t > 15:
            SEARCH_T = 100.
            SEARCH_W_T = 0.3
        else:
            SEARCH_T = 100.
            SEARCH_W_T = 0.3

        rand_int = random.uniform(-1, 1)

        if (self._scope_start_t_ is not None) != obs['in_scope']:
            self._scope_unsync_cnt_ += 1
            if self._scope_unsync_cnt_ > 15:
                mouse.click(Button.right)
                self._scope_unsync_cnt_ = 0
        else:
            self._scope_unsync_cnt_ = 0

        if (self._fire_start_t_ is not None) and self._fire_t > 0.6:
            self._end_fire()

        if (self._scope_start_t_ is not None)\
           and ((self._scope_t > 4.) or ((self._last_fire_t > 1.) and no_target and (self._scope_t > 1.)))\
           and self._fire_start_t_ is None:
            self._end_scope()

        in_search = False
        in_chase = False
        need_slp = False

        if no_target:
            self.aimer.pid.reset()
            ex, ey = self.aimer.calc_error(*GloablStatus.in_window_center_xy())
            mv_x, mv_y = self.aimer.calc_movement(ex, ey, False)
            if random.uniform(0, 1) < 0.02:
                act_dict = self.kb_sm_lesure.step(act_dict)

            if self._last_fire_t > 1 and self._last_search_t > SEARCH_W_T:
                assert self.model_tick > cfg.tick
                slp = Sleeper(tick = self.model_tick - cfg.tick)
                need_slp = True

                in_search = True
                if self._search_start_t_ is None:
                    self._start_search()
                    self.model.reset()
                
                wasd, xy = self.model.act([frame])
                limit = 500
                mv_x, mv_y = norm(xy[0], lower_side=-limit, upper_side=limit), norm(xy[1], lower_side=-limit, upper_side=limit)
                if wasd[0] > 0:
                    act_dict['w'] = 1
                elif wasd[1] > 0:
                    act_dict['a'] = 1
                elif wasd[2] > 0:
                    act_dict['s'] = 1
                elif wasd[3] > 0:
                    act_dict['d'] = 1
           
            # else:
            #     if self._last_search_t > SEARCH_W_T/2 and self._last_detect_t > 4:
            #         mv_x = -130 if int(self._last_detect_end_t_) % 2 == 0 else 130
            #         mv_x += rand_int * 20


        
        else:
            self._last_detect_end_t_ = time.time()

            w, h = obs['wh']; assert w > 0 and h > 0
            ex, ey = self.aimer.calc_error(*obs['xy'])
            inbound = (abs(ex) < w and abs(ey) < h)
            mv_x, mv_y = self.aimer.calc_movement(ex, ey, inbound)
            if self._fire_start_t_ is not None:
                mv_y += 1.

            
            if inbound:
                if self._scope_start_t_ is not None:
                    if self._fire_start_t_ is None:
                        self._start_fire()
                elif max(abs(w), abs(h)) > 150 :
                    if (self._fire_start_t_ is None) or self._fire_t > 0.2:
                        self._start_fire()
                else:
                    if self._scope_start_t_ is None:
                        self._scope_start_t_ = time.time()
                        mouse.click(Button.right)
            
            if random.uniform(0, 1) < 0.02:
                    act_dict = self.kb_sm_fight.step(act_dict)

        
        if (not in_search or self._search_t > SEARCH_T)and self._search_start_t_ is not None:
            self._end_search()
        
        # if (not in_chase) and (not in_search) and is_pressing():
        #     unpress_kb_bt()
        

        for k in ['w', 'a', 's', 'd']:
            if act_dict[k] > 0:
                if self.in_press[k] <= 0: 
                    KB.kb.press(k)
                    self.in_press[k] = 1
            else: 
                if self.in_press[k] > 0:
                    KB.kb.release(k)
                    self.in_press[k] = 0
                
        if need_slp:
            # coef = 1.4            # mv_x, mv_y = mv_x * coef, mv_y * coef
            mv_x, mv_y = mv_x/3, mv_y/3
            move_mouse(mv_x, mv_y)
            slp.sleep_half()
            move_mouse(mv_x, mv_y)
            slp.sleep()
        move_mouse(mv_x, mv_y)

        

        act_dict['mv_xy'] = (mv_x, mv_y,)
        act_dict['fire'] = 1 if (self._fire_start_t_ is not None) else 0
        act_dict['scope'] = 1 if (self._scope_start_t_ is not None) else 0

        return {
            'obs': obs_dict,
            'act': act_dict
        }


class KBStateMachine:
    def __init__(self):
        self.state: int = 0
        self.keys = []

    def add_key(self, action_key, *action_keys):
        self.keys.append(action_key)
        for key in action_keys:
            assert isinstance(key, str)
            self.keys.append(key)

    def step(self, act_dict):
        if len(self.keys) == 0: return
        idx = self.state % len(self.keys)
        hit_kb_bt(self.keys[idx])
        act_dict[self.keys[idx]] = 1
        lprint(self, f"hit_kb_bt('{self.keys[idx]}')")
        self.state += 1
        return act_dict


class Operator(threading.Thread):
    def __init__(self, draw_action_hook=None):
        super().__init__()
        self.obs: dict = None
        self.obs_ready_mutex = threading.Semaphore(value=0)
        self.draw_action_hook = draw_action_hook

        self.sm = AgentStateMachine()

        self.start_time = time.time()
        

    def run(self):
        lprint(self, "start")
        try:
            while not GloablStatus.stop_event.is_set():
                self.obs_ready_mutex.acquire(timeout = 3 * cfg.tick)
                if self.obs is None: 
                    self.draw_action_hook(None)
                    lprint(self, "Warning: obs is None, abort")
                    continue

                if time.time() - self.start_time < 5:
                    self.draw_action_hook(None)
                    continue

                obs = self.obs; self.obs = None

                data = self.sm.step(obs)

                self.draw_action_hook(data)
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            self.obs = None
        lprint(self, "finish")

    def see_obs(self, obs: dict):
        lprint(self, "see_obs called", debug=True)

        assert isinstance(obs, dict)
        last_obs = self.obs; self.obs = obs
        if last_obs is None:
            self.obs_ready_mutex.release()
