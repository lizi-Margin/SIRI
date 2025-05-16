from pynput.mouse import Controller as MouseController
from pynput.keyboard import Controller as KBController
from siri.utils.sleeper import Sleeper
class OutputUnit:
    def __init__(self):
        self.in_press = {
            'w': 0,
            'a': 0,
            's': 0,
            'd': 0,
        }
        self.kb = KBController()
        self.mouse = MouseController()
    def move_mouse(self, move_x, move_y):
        if abs(move_x) < 0.1 and abs(move_y) < 0.1:
            return  # abort
        self.mouse.move(move_x, move_y)
    
    def release_all(self):
        for key in self.in_press:
            if self.in_press[key] > 0:
                self.kb.release(key)
                self.in_press[key] = 0

    def output_real_act(self, act_dict, mv_x, mv_y, sleeper: Sleeper):
        for k in ['w', 'a', 's', 'd']:
            if act_dict[k] > 0:
                if self.in_press[k] <= 0: 
                    self.kb.press(k)
                    self.in_press[k] = 1
            else: 
                if self.in_press[k] > 0:
                    self.kb.release(k)
                    self.in_press[k] = 0

        mv_x, mv_y = mv_x/3, mv_y/3
        self.move_mouse(mv_x, mv_y)
        sleeper.sleep_one_of_n(n=4)
        self.move_mouse(mv_x, mv_y)
        sleeper.sleep_one_of_n(n=3)
        self.move_mouse(mv_x, mv_y)