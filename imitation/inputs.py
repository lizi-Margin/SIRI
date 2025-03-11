import time, cv2, mss, os
import numpy as np
from pynput import keyboard, mouse

from siri.utils.sleeper import Sleeper
from siri.vision.detector import ScrGrabber
from siri.global_config import GloablStatus, GlobalConfig as cfg
from siri.utils.logger import lprint, lprint_
from siri.vision.preprocess import crop
from UTIL.colorful import *
from .traj import trajectory
from .utils import safe_dump_traj_pool


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
start_char = 'j'
stop_char = 'k'

def on_key_press(key):
    if hasattr(key, 'char'):
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


class Grabber(ScrGrabber):
    def __init__(self):
        super().__init__()
        self.traj_pool: list[trajectory] = []
        self.tick = 0.1
        self.traj_limit = 200
        self.sz = 640
        lprint(self, f"traj time limit: {self.tick * self.traj_limit}")
    
    def new_traj(self):
        # Create trajectory storage
        traj = trajectory(traj_limit=self.traj_limit, env_id=0)
        global last_mouse_pos; last_mouse_pos = None
        return traj

    def start_dataset_session(self):
        geometry = self.get_scrcpy_window_geometry()
        if not geometry:
            lprint(self, "scrcpy window not found")
            lprint(self, "start_session failed")
            return

        assert GloablStatus.monitor is None
        left, top, scr_width, scr_height = geometry

        GloablStatus.monitor = {
            "top": top,
            "left": left,
            "width": scr_width,
            "height": scr_height,
        }

        traj = self.new_traj()


        # Initialize listeners
        self.keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
        self.mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_press, on_release=on_mouse_release)
        self.keyboard_listener.start()
        self.mouse_listener.start()

        
        try:
            with mss.mss() as sct:
                global start, stop, new_mouse_pos, last_mouse_pos, actions
                stop = 1
                while True:
                    if stop:
                        if traj.time_pointer > 0:
                            self.traj_pool.append(traj)
                            traj = self.new_traj()
                        print(end='\n')
                        while not start: 
                            time.sleep(0.25)
                            print靛('\r'+lprint_(self, "paused"), end='')
                        init_actions()
                        new_mouse_pos = None; last_mouse_pos = None
                        stop = 0
                    start = 0
                    print绿('\r'+lprint_(self, f"started, traj collected: {len(self.traj_pool)}"), end='')

                    sleeper = Sleeper(tick=self.tick, user=self)

                    if traj.time_pointer == self.traj_limit:
                        self.traj_pool.append(traj)
                        traj = self.new_traj()




                    screenshot = sct.grab(GloablStatus.monitor)
                    frame = np.array(screenshot)
                    if frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    assert frame.shape[-1] == 3, 'not a BGR format'
                    traj.remember('FRAME_raw', frame.copy())

                    # frame = cv2.resize(frame, cfg.sz_wh)
                    # frame = crop(frame)
                    # frame = cv2.resize(frame, (self.sz, self.sz,))
                    # traj.remember('FRAME_cropped', frame.copy())

                    if (new_mouse_pos is not None) and (last_mouse_pos is not None):
                        mouse_movement = new_mouse_pos - last_mouse_pos
                        mouse_movement = mouse_movement.astype(np.float32)
                        last_mouse_pos = new_mouse_pos
                    else: 
                        mouse_movement = np.array([0., 0.], dtype=np.float32)
                        lprint(self, "Warning: new_mouse_pos is None")
                    act = np.array(list(actions.values()), dtype=np.float32)
                    if act.any():
                        print(actions)
                    # if mouse_movement.any():
                    #     print(mouse_movement)
                    traj.remember('key', act)
                    traj.remember('mouse', mouse_movement.copy())
                    traj.time_shift()
                    # actions = init_actions()
                    sleeper.sleep()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            # cv2.destroyAllWindows()
            GloablStatus.monitor = None
            GloablStatus.stop_event.set()
            for i in range(len(self.traj_pool)): self.traj_pool[i].cut_tail()
            pool_name = f"{self.__class__.__name__}-tick={self.tick}-limit={self.traj_limit}-sz={self.sz}-{time.strftime("%Y%m%d-%H:%M:%S")}"
            safe_dump_traj_pool(self.traj_pool, pool_name)
            print亮黄(lprint_(self, "terminated"))




