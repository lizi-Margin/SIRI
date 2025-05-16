import time, cv2, mss, os
import numpy as np
from pynput import keyboard, mouse
from pynput.keyboard import Key

from siri.utils.sleeper import Sleeper
from siri.vision.detector import ScrGrabber
from siri.global_config import GloablStatus, GlobalConfig as cfg
from siri.utils.logger import lprint, lprint_
from siri.vision.preprocess import crop
from UTIL.colorful import *
from .traj import trajectory
from .utils import safe_dump_traj_pool, iterable_eq
from .filter import mouse_filter, mouse_pos_filter


# Define keys and mouse buttons to track
KEY_INPUTS = [
    'w', 'a', 's', 'd', ' ', 'c',
    '1', '2', '3', '4', '5', '6', 'g', 'f', 'r',
    'q'
]
MOUSE_BUTTONS = {
    mouse.Button.left: 'mouse_left',
    mouse.Button.right: 'mouse_right'
}

def init_actions():
    k = {key: 0 for key in KEY_INPUTS}
    k.update({btn: 0 for btn in MOUSE_BUTTONS.values()})
    return k
actions = init_actions()
last_mouse_pos = None
new_mouse_pos = None

start = 0
stop = 0
human = 0
start_char = 'z'
stop_char = 'x'
human_char = 'l_shift'

def on_key_press(key):
    if key == Key.shift_l: 
        global human
        human = int(not human)
    elif hasattr(key, 'char'):
        if key.char in actions:
            actions[key.char] = 1
        if key.char == start_char:
            global start
            start = 1
        if key.char == stop_char:
            global stop
            stop = 1

def on_key_release(key):
    if hasattr(key, 'char') and key.char in actions:
        actions[key.char] = 0

def on_mouse_move(x, y):
    global last_mouse_pos, new_mouse_pos
    if last_mouse_pos is None:
        last_mouse_pos = np.array([x, y], dtype=np.float32)
    new_mouse_pos = np.array([x, y], dtype=np.float32)

def on_mouse_press(x, y, button, _):
    if button in MOUSE_BUTTONS:
        if _ == True:
            actions[MOUSE_BUTTONS[button]] = 1
        else:
            actions[MOUSE_BUTTONS[button]] = 0

def on_mouse_release(x, y, button, _):
    pass
    # print(2, _)
    # if button in MOUSE_BUTTONS:
    #    actions[MOUSE_BUTTONS[button]] = 0


class ILGrabber(ScrGrabber):
    def __init__(self):
        super().__init__()
        self.traj_pool: list[trajectory] = []
        self.tick = 0.1
        self.traj_limit = 200
        lprint(self, f"traj time limit: {self.tick * self.traj_limit}")

        self.x_filter = mouse_pos_filter()
        self.y_filter = mouse_pos_filter(D_MAX=200)
    
    def new_traj(self):
        # Create trajectory storage
        traj = trajectory(traj_limit=self.traj_limit, env_id=0)
        global last_mouse_pos; last_mouse_pos = None
        self.x_filter.reset()
        self.y_filter.reset()
        return traj

    def start_dataset_session(self):
        geometry = self.get_scrcpy_window_geometry()
        if not geometry:
            lprint(self, "scrcpy window not found")
            lprint(self, "start_session failed")
            return

        assert GloablStatus.monitor is None
        self.sync_monitor(geometry)

        

        # Initialize listeners
        self.keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        self.mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_press, on_release=on_mouse_release)
        self.keyboard_listener.start()
        self.mouse_listener.start()

        
        try:
            with mss.mss() as sct:
                def grab_screen():
                    screenshot = sct.grab(GloablStatus.monitor)
                    frame = np.array(screenshot)
                    if frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    assert frame.shape[-1] == 3, 'not a BGR format'
                    frame_hw = frame.shape[:2]
                    if not iterable_eq(frame_hw, tuple(reversed(cfg.sz_wh))):
                        if not hasattr(self, "_sz_noticed"):
                            self._sz_noticed = True
                            lprint(self, f"Warning: frame_hw={frame_hw}, resize will be used")
                        frame = cv2.resize(frame, cfg.sz_wh)
                    return frame, time.time_ns()

                traj = self.new_traj()
                self.last_frame, self.last_frame_time = grab_screen()
                time.sleep(self.tick)


                global start, stop, new_mouse_pos, last_mouse_pos, actions
                stop = 1
                while True:
                    if stop:
                        print(end='\n')
                        while not start: 
                            time.sleep(0.25)
                            print靛('\r'+lprint_(self, "paused"), end='')
                        if traj.time_pointer > 0:
                            self.traj_pool.append(traj)
                            traj = self.new_traj()
                        init_actions()
                        new_mouse_pos = None; last_mouse_pos = None
                        stop = 0
                        
                        self.last_frame, self.last_frame_time = grab_screen()
                        time.sleep(self.tick)
                    start = 0

                    if traj.time_pointer == self.traj_limit:
                        self.traj_pool.append(traj)
                        traj = self.new_traj()


                    print绿('\r'+lprint_(self, f"started, traj collected: {len(self.traj_pool)}"), end='')
                    sleeper = Sleeper(tick=self.tick, user=self)
                    
                    assert time.time_ns() - self.last_frame_time < 2 * self.tick * 1e9
                    traj.remember('FRAME_raw', self.last_frame.copy())
                    self.last_frame, self.last_frame_time = grab_screen()

                    # frame = cv2.resize(frame, cfg.sz_wh)
                    # frame = crop(frame)
                    # frame = cv2.resize(frame, (self.sz, self.sz,))
                    # traj.remember('FRAME_cropped', frame.copy())

                    if (new_mouse_pos is not None):
                        last_mouse_pos = new_mouse_pos
                        mouse_movement = np.zeros(2, dtype=np.float32)

                        mouse_movement[0] = self.x_filter.step(new_mouse_pos[0])
                        mouse_movement[1] = self.y_filter.step(new_mouse_pos[1])
                        mouse_movement = mouse_movement.astype(np.float32)
                    else: 
                        mouse_movement = np.array([0., 0.], dtype=np.float32)
                        lprint(self, "Warning: new_mouse_pos is None")
                    act = np.array(list(actions.values()), dtype=np.float32)
                    # if act.any():
                    #     print(actions)
                    if np.max(np.abs(mouse_movement), axis=None) > 50:
                        print(mouse_movement, last_mouse_pos)
                    traj.remember('key', act)
                    traj.remember('mouse', mouse_movement.copy())
                    traj.time_shift()
                    # actions = init_actions()
                    self.sync_monitor_every_n_step(n=100)
                    sleeper.sleep()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            if traj.time_pointer > 0:
                self.traj_pool.append(traj)

            # cv2.destroyAllWindows()
            GloablStatus.monitor = None
            GloablStatus.stop_event.set()
            for i in range(len(self.traj_pool)): self.traj_pool[i].cut_tail()
            pool_name = f"{self.__class__.__name__}-tick={self.tick}-limit={self.traj_limit}-{time.strftime("%Y%m%d-%H:%M:%S")}"
            safe_dump_traj_pool(self.traj_pool, pool_name)
            print亮黄(lprint_(self, "terminated"))

def norm(x: float, lower_side: float=-1.0, upper_side: float=1.0):
    if (x > upper_side): x = upper_side
    if (x < lower_side): x = lower_side
    return x


class RLGrabber(ILGrabber):
    def __init__(self):
        super().__init__()
        from .outputs import OutputUnit
        self.output = OutputUnit()

    @staticmethod
    def get_real_act(wasd, xy):
        limit = 500
        mv_x, mv_y = norm(xy[0], lower_side=-limit, upper_side=limit), norm(xy[1], lower_side=-limit, upper_side=limit)
        act_dict = {
            'w': 0,
            'a': 0,
            's': 0,
            'd': 0,
        }
        if wasd[0] > 0:
            act_dict['w'] = 1
        elif wasd[1] > 0:
            act_dict['a'] = 1
        elif wasd[2] > 0:
            act_dict['s'] = 1
        elif wasd[3] > 0:
            act_dict['d'] = 1
        return act_dict, mv_x, mv_y
    

    def start_rl_session(self, rl_alg):
        geometry = self.get_scrcpy_window_geometry()
        if not geometry:
            lprint(self, "scrcpy window not found")
            lprint(self, "start_session failed")
            return

        assert GloablStatus.monitor is None
        self.sync_monitor(geometry)

        

        # Initialize listeners
        self.keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        self.mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_press, on_release=on_mouse_release)
        self.keyboard_listener.start()
        self.mouse_listener.start()

        print蓝("RL session start...")
        
        try:
            with mss.mss() as sct:
                def grab_screen():
                    screenshot = sct.grab(GloablStatus.monitor)
                    frame = np.array(screenshot)
                    if frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    assert frame.shape[-1] == 3, 'not a BGR format'
                    frame_hw = frame.shape[:2]
                    if not iterable_eq(frame_hw, tuple(reversed(cfg.sz_wh))):
                        if not hasattr(self, "_sz_noticed"):
                            self._sz_noticed = True
                            lprint(self, f"Warning: frame_hw={frame_hw}, resize will be used")
                        frame = cv2.resize(frame, cfg.sz_wh)
                    return frame, time.time_ns()

                traj = self.new_traj()
                time.sleep(self.tick)


                global start, stop
                stop = 1
                real_stop = 1
                while True:
                    if real_stop:
                        print(end='\n')
                        while not start: 
                            rl_alg.train()
                            time.sleep(0.25)
                            print靛('\r'+lprint_(self, "paused"), end='')
                        if traj.time_pointer > 0:
                            self.traj_pool.append(traj)
                            traj = self.new_traj()
                        stop = 0
                        real_stop = 0
                    if stop: real_stop = 1
                    start = 0

                    print绿('\r'+lprint_(self, f"started, traj collected: {len(self.traj_pool)}"), end='')
                    sleeper = Sleeper(tick=self.tick, user=self)
                    
                    frame, frame_time = grab_screen()
                    wasd, xy = rl_alg.interact_with_env({
                        'obs': frame,
                        'done': (traj.time_pointer >= self.traj_limit - 1) or stop,
                    })
                    act_dict, mv_x, mv_y = self.get_real_act(wasd, xy)
                    self.output.output_real_act(act_dict, mv_x, mv_y, sleeper)
                    
                    traj.remember('FRAME_raw', frame.copy())
                    traj.remember('key', wasd)
                    traj.remember('mouse', xy)
                    traj.time_shift()
                    if traj.time_pointer == self.traj_limit:
                        self.traj_pool.append(traj)
                        traj = self.new_traj()

                    self.sync_monitor_every_n_step(n=100)
                    sleeper.sleep()
                    # rl_alg.train()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            if traj.time_pointer > 0:
                self.traj_pool.append(traj)

            # cv2.destroyAllWindows()
            GloablStatus.monitor = None
            GloablStatus.stop_event.set()
            # for i in range(len(self.traj_pool)): self.traj_pool[i].cut_tail()
            # pool_name = f"{self.__class__.__name__}-tick={self.tick}-limit={self.traj_limit}-{time.strftime("%Y%m%d-%H:%M:%S")}"
            # safe_dump_traj_pool(self.traj_pool, pool_name, traj_dir=f"HMP_IL/AIRL/AUTOSAVED/{time.strftime("%Y%m%d-%H:%M:%S")}/")
            print亮黄(lprint_(self, "terminated"))



class DAggrGrabber(RLGrabber):
    def get_human_act_dict(self, key, mouse_xy):
        act_wasd =     key[:4]
        index_jump =   key[4]
        index_crouch = key[5]
        index_reload = key[14]
        index_r =      key[17]
        index_l =      key[16]
        act_mouse_x = mouse_xy[0]
        act_mouse_y = mouse_xy[1]
        act_dict = {
            'act_wasd': act_wasd,
            'xy': mouse_xy,
            'act_mouse_x': act_mouse_x,
            'act_mouse_y': act_mouse_y,
            'index_jump': index_jump,
            'index_crouch': index_crouch,
            'index_reload': index_reload,
            'index_r': index_r,
            'index_l': index_l,
        }
        return act_dict

    def start_rl_session(self, rl_alg):
        geometry = self.get_scrcpy_window_geometry()
        if not geometry:
            lprint(self, "scrcpy window not found")
            lprint(self, "start_session failed")
            return

        assert GloablStatus.monitor is None
        self.sync_monitor(geometry)

        

        # Initialize listeners
        self.keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        self.mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_press, on_release=on_mouse_release)
        self.keyboard_listener.start()
        self.mouse_listener.start()

        print蓝("DAggr session start...")
        
        try:
            with mss.mss() as sct:
                def grab_screen():
                    screenshot = sct.grab(GloablStatus.monitor)
                    frame = np.array(screenshot)
                    if frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    assert frame.shape[-1] == 3, 'not a BGR format'
                    frame_hw = frame.shape[:2]
                    if not iterable_eq(frame_hw, tuple(reversed(cfg.sz_wh))):
                        if not hasattr(self, "_sz_noticed"):
                            self._sz_noticed = True
                            lprint(self, f"Warning: frame_hw={frame_hw}, resize will be used")
                        frame = cv2.resize(frame, cfg.sz_wh)
                    return frame, time.time_ns()

                traj = self.new_traj()
                time.sleep(self.tick)


                global start, stop, new_mouse_pos, last_mouse_pos, actions, human
                stop = 1
                real_stop = 1
                while True:
                    if real_stop:
                        print(end='\n')
                        while not start: 
                            self.save_traj()
                            rl_alg.train()
                            time.sleep(0.25)
                            print靛('\r'+lprint_(self, "paused"), end='')
                        if traj.time_pointer > 0:
                            self.traj_pool.append(traj)
                            traj = self.new_traj()
                        init_actions()
                        new_mouse_pos = None; last_mouse_pos = None
                        stop = 0
                        real_stop = 0
                    if stop: real_stop = 1
                    start = 0

                    print绿('\r'+lprint_(self, f"started, traj collected: {len(self.traj_pool)}"), end='')
                    sleeper = Sleeper(tick=self.tick, user=self)
                    
                    frame, frame_time = grab_screen()
                    if (new_mouse_pos is not None):
                        last_mouse_pos = new_mouse_pos
                        rec_mouse_xy = np.zeros(2, dtype=np.float32)

                        rec_mouse_xy[0] = self.x_filter.step(new_mouse_pos[0])
                        rec_mouse_xy[1] = self.y_filter.step(new_mouse_pos[1])
                        rec_mouse_xy = rec_mouse_xy.astype(np.float32)
                    else: 
                        rec_mouse_xy = np.array([0., 0.], dtype=np.float32)
                        print("\rWarning: new_mouse_pos is None", end='')
                    rec_act = np.array(list(actions.values()), dtype=np.float32)
                    if np.max(np.abs(rec_mouse_xy), axis=None) > 50:
                        print(rec_mouse_xy, last_mouse_pos)
                    # if rec_act.any():
                    #     print(rec_act[:4])  

                    # human_active = rec_act.any() or np.max(np.abs(rec_mouse_xy), axis=None) > 1
                    human_active = human > 0
                    rec_input_dict = self.get_human_act_dict(rec_act, rec_mouse_xy)


                    m_wasd, m_xy = rl_alg.interact_with_env({
                        'obs': frame,
                        'done': (traj.time_pointer >= self.traj_limit - 1) or stop,
                        'rec': rec_input_dict,
                        'human_active': human_active
                    })
                    


                    if human_active:
                        print红("\rhuman active", end='')
                        self.output.release_all()
                    else:
                        act_dict, mv_x, mv_y = self.get_real_act(m_wasd, m_xy)
                        self.output.output_real_act(act_dict, mv_x, mv_y, sleeper)

                        # key = np.concatenate([wasd, np.zeros(14,)])
                        # assert len(key.shape) == 1, len(key.shape)
                        # assert key.shape[0] == 18, key.shape[0]
                        # xy  = xy
                    key = rec_act
                    xy  = rec_mouse_xy
                    traj.remember('FRAME_raw', frame.copy())
                    traj.remember('key', key.copy())
                    traj.remember('mouse', xy.copy())
                    traj.time_shift()
                    if traj.time_pointer == self.traj_limit:
                        self.traj_pool.append(traj)
                        traj = self.new_traj()

                    self.sync_monitor_every_n_step(n=100)
                    sleeper.sleep()
                    # rl_alg.train()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            if traj.time_pointer > 0:
                self.traj_pool.append(traj)

            # cv2.destroyAllWindows()
            GloablStatus.monitor = None
            GloablStatus.stop_event.set()
            self.save_traj()
            print亮黄(lprint_(self, "terminated"))
    
    def save_traj(self):
        if len(self.traj_pool) > 0:
            for i in range(len(self.traj_pool)): self.traj_pool[i].cut_tail()
            pool_name = f"{self.__class__.__name__}-tick={self.tick}-limit={self.traj_limit}-{time.strftime("%Y%m%d-%H:%M:%S")}"
            safe_dump_traj_pool(self.traj_pool, pool_name, traj_dir=f"HMP_IL/DAggr/AUTOSAVED/{time.strftime("%Y%m%d-%H:%M:%S")}/")
            self.traj_pool = []